package dev.swipixel.restyler;

import com.mojang.blaze3d.platform.GlStateManager;
import com.mojang.blaze3d.systems.RenderSystem;
import net.minecraft.client.gl.Framebuffer;
import net.minecraft.client.render.BufferRenderer;
import org.joml.Matrix4f;
import org.lwjgl.opengl.GL11;
import org.lwjgl.opengl.GL12;
import org.lwjgl.opengl.GL13;
import org.lwjgl.opengl.GL20;
import org.lwjgl.opengl.GL30;
import org.lwjgl.system.MemoryStack;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;

/**
 * GPU side of the async restyler: re-projection with the game's depth buffer (like TAA / VR async reprojection).
 *
 * Textures (all in GL convention, row 0 = bottom):
 *   world  RG32F  full res   R = distance camera->surface in blocks (1e6 = sky), G = raw depth.  Written at the END
 *                            of world rendering (before the hand clears the depth buffer).
 *   cur    R32F   full res   same distance, but -1 where the HAND/held item is (depth at HUD time < world depth).
 *   capDist R32F  512x288    distance at the moment a frame was sent to the network.
 *   result RGBA8  512x288    the network's newest output.
 *   hist   RGBA8  512x288 x2 accumulated output (ping-pong) + histDist, in the camera of the newest capture.
 *
 * New result:  hist = mix(reproject(hist_old -> new camera), result, alpha)   (where the old pixel is the same surface)
 * Every frame: screen = reproject(hist -> current camera); pixels that can't be matched show the vanilla game.
 */
final class Stabilizer {

    static final int W = 512;
    static final int H = 288;
    private static final float FEATHER = 0.05f;   // fade over the outer 5 % of the look instead of a hard seam

    /** A camera: P * V (V = rotation only, Minecraft renders camera-relative) and its world position. */
    static final class Cam {
        final Matrix4f pv = new Matrix4f();
        final Matrix4f invPv = new Matrix4f();
        double x, y, z;
        float f11;                     // projection[1][1] = 1 / tan(fov / 2)
        boolean valid;

        void set(Matrix4f proj, Matrix4f view, double x, double y, double z) {
            pv.set(proj).mul(view);
            f11 = proj.m11();
            pv.invert(invPv);
            this.x = x;
            this.y = y;
            this.z = z;
            valid = true;
        }

        void set(Cam o) {
            pv.set(o.pv);
            invPv.set(o.invPv);
            f11 = o.f11;
            x = o.x;
            y = o.y;
            z = o.z;
            valid = o.valid;
        }
    }

    final Cam curCam = new Cam();     // this frame (from the END of world rendering)
    final Cam capCam = new Cam();     // the frame in flight to the network
    private final Cam histCam = new Cam();
    private boolean histValid = false;
    private boolean worldReady = false;   // world texture written this frame
    private boolean curReady = false;     // hand mask written this frame

    private int vao = -1;
    private int progWorld, progCur, progCap, progHist, progShow;
    private int fw = -1, fh = -1;
    private int worldFbo = -1, worldTex = -1, curFbo = -1, curTex = -1;
    private int capDistFbo = -1, capDistTex = -1, histDistFbo = -1, histDistTex = -1;
    private int resultTex = -1;
    private int vanFbo = -1, vanTex = -1;      // copy of the current frame (shader can't read the screen it draws on)
    private int vmidFbo = -1, vmidTex = -1;    // current frame crop at 1024x576 ...
    private int vlowFbo = -1, vlowTex = -1;    // ... and 512x288 = the low band the network sees
    private int lutTex = -1;                   // 3D colour LUT: vanilla rgb -> change the network makes
    private boolean haveLut = false;
    static final int LUT = 16;
    private final int[] histFbo = {-1, -1}, histTex = {-1, -1};
    private int histIdx = 0;

    // ------------------------------------------------------------ shaders

    private static final String VS = """
            #version 150
            out vec2 uv;
            void main() {
                vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);   // one triangle covering the viewport
                uv = p;
                gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
            }
            """;

    /** Distance from the camera to the surface behind normalized screen position s (0..1). */
    private static final String COMMON = """
            #version 150
            in vec2 uv;
            out vec4 o;
            const float SKY = 1.0e6;
            vec3 rayDir(mat4 invPv, vec2 s) {
                vec4 f = invPv * vec4(s * 2.0 - 1.0, 1.0, 1.0);
                return normalize(f.xyz / f.w);
            }
            // source history: uSrcDist / uPvSrc / uCropSrc;  uDelta = destination camera pos - source camera pos
            uniform sampler2D uSrcDist;
            uniform mat4 uPvSrc;
            uniform mat4 uInvPvDst;
            uniform vec3 uDelta;
            uniform vec4 uCropSrc;
            uniform float uTol;
            uniform float uFeather;
            const vec2 TEX = vec2(1.0 / 512.0, 1.0 / 288.0);
            // destination pixel at screen pos s with distance d -> where to read the source (uvOut).
            // Returns a weight: 0 = no matching surface (newly visible / outside the capture), up to 1.
            // The source distance is checked in a 3x3 neighbourhood: at edges and on slanted far ground one
            // 512x288 texel spans several blocks, a single sample would reject good pixels (vanilla outlines).
            // pDst = point relative to the DESTINATION camera (world axes)
            float reprojectP(vec3 pDst, bool sky, out vec2 uvOut) {
                uvOut = vec2(-1.0);
                vec3 p = sky ? pDst : pDst + uDelta;
                vec4 c = uPvSrc * vec4(p, 1.0);
                if (c.w <= 1.0e-4) return 0.0;
                vec2 su = ((c.xy / c.w) * 0.5 + 0.5 - uCropSrc.xy) / uCropSrc.zw;
                float edge = min(min(su.x, 1.0 - su.x), min(su.y, 1.0 - su.y));
                if (edge < 0.0) return 0.0;
                float de = sky ? SKY : length(p);
                float tol = uTol * de + 0.15;
                float lo = 1.0e30, hi = -1.0e30, best = 1.0e30;
                vec2 bu = su;
                for (int y = -1; y <= 1; y++) {
                    for (int x = -1; x <= 1; x++) {
                        vec2 t = su + vec2(x, y) * TEX;
                        float ds = texture(uSrcDist, t).r;
                        if (ds < 0.0 || (ds >= SKY * 0.1) != sky) continue;   // hand / sky-vs-ground
                        lo = min(lo, ds);
                        hi = max(hi, ds);
                        float e = abs(ds - de);
                        if (e < best) { best = e; bu = t; }
                    }
                }
                if (hi < 0.0) return 0.0;
                // a neighbour is that surface, or a smooth slope (not an edge) spans that distance
                bool slope = hi - lo < 0.3 * de && de > lo - tol && de < hi + tol;
                if (!sky && best > tol && !slope) return 0.0;                  // newly visible -> vanilla
                float dc = texture(uSrcDist, su).r;
                if (sky || hi - lo > 0.3 * de) {
                    // depth edge (leaf against sky, grass blade against ground) or sky: read exactly ONE texel, the
                    // one on the same surface - bilinear would mix in the other side and the history keeps the ghost
                    uvOut = (floor(bu / TEX) + 0.5) * TEX;
                } else {
                    uvOut = (dc >= 0.0 && abs(dc - de) <= tol) || best > tol ? su : bu;
                }
                return smoothstep(0.0, uFeather, edge);                        // soft edge, no seam
            }
            float reproject(vec2 s, float d, out vec2 uvOut) {
                vec3 dir = rayDir(uInvPvDst, s);
                bool sky = d >= SKY * 0.1;
                return reprojectP(sky ? dir * SKY : dir * d, sky, uvOut);
            }
            """;

    // END of world rendering: main depth -> distance + raw depth
    private static final String FS_WORLD = COMMON + """
            uniform sampler2D uDepth;
            uniform mat4 uInvPv;
            void main() {
                float z = texture(uDepth, uv).r;
                if (z >= 0.99999) { o = vec4(SKY, z, 0.0, 1.0); return; }
                vec4 p = uInvPv * vec4(uv * 2.0 - 1.0, z * 2.0 - 1.0, 1.0);
                o = vec4(length(p.xyz / p.w), z, 0.0, 1.0);
            }
            """;

    // HUD time: depth now holds only the hand -> mark hand pixels -1
    private static final String FS_CUR = COMMON + """
            uniform sampler2D uDepth;
            uniform sampler2D uWorld;
            void main() {
                vec2 w = texture(uWorld, uv).rg;
                float z = texture(uDepth, uv).r;
                o = vec4(z < w.g - 1.0e-6 ? -1.0 : w.r, 0.0, 0.0, 1.0);
            }
            """;

    // capture: full-res cur distance -> 512x288 (crop)
    private static final String FS_CAP = COMMON + """
            uniform sampler2D uCur;
            uniform vec4 uCrop;
            void main() { o = vec4(texture(uCur, uCrop.xy + uv * uCrop.zw).r, 0.0, 0.0, 1.0); }
            """;

    // new result: blend with the re-projected history
    private static final String FS_HIST = COMMON + """
            uniform sampler2D uNew;
            uniform sampler2D uNewDist;
            uniform sampler2D uHist;
            uniform vec4 uCropDst;
            uniform float uAlpha;
            uniform int uUseHist;
            void main() {
                vec4 n = texture(uNew, uv);
                float d = texture(uNewDist, uv).r;
                if (uUseHist == 0 || d < 0.0) { o = n; return; }
                vec2 su;
                float w = reproject(uCropDst.xy + uv * uCropDst.zw, d, su);
                o = w > 0.0 ? mix(n, texture(uHist, su), (1.0 - uAlpha) * w) : n;
            }
            """;

    // every frame: base = vanilla graded with the LUT learned from the last result (covers EVERYTHING, also what
    // the network hasn't seen yet), on top = the detailed look re-projected to the current camera.
    private static final String FS_SHOW = COMMON + """
            uniform sampler2D uCur;
            uniform sampler2D uHist;
            uniform sampler2D uVanilla;
            uniform sampler2D uLow;        // vanilla at 512x288 = what the network saw (the low band)
            uniform float uDetail;         // 0 = off, 1 = full game texture detail on top of the look
            uniform sampler3D uLut;
            uniform int uHaveLut;
            uniform vec4 uCropDst;
            uniform float uStrength;
            uniform int uReproject;
            uniform int uTexelSnap;
            uniform vec3 uCamFrac;        // camera position mod 16 (world): block grid without float precision loss
            uniform float uFocalPx;       // pixels per unit at distance 1
            // full-block faces lie on whole-number planes -> which 16x16 texel is this pixel on? Read the look once,
            // at the texel's centre, for all of its pixels: paint locked to the block grid (pixel-art crisp, no
            // sliding inside a texel). Mobs, hand, water, slabs, plants are not on the grid -> unchanged.
            float snapped(vec2 s, float d, out vec2 su) {
                vec3 p = rayDir(uInvPvDst, s) * d;
                vec3 w = uCamFrac + p;
                vec3 fr = abs(w - floor(w + 0.5));
                int k = fr.x < fr.y ? (fr.x < fr.z ? 0 : 2) : (fr.y < fr.z ? 1 : 2);
                if (fr[k] > 0.01 + 0.002 * d) return reproject(s, d, su);      // not on the block grid
                vec3 c = floor(w * 16.0) / 16.0 + 1.0 / 32.0;                    // texel centre ...
                c[k] = floor(w[k] + 0.5);                                         // ... on the face plane
                return reprojectP(c - uCamFrac, false, su);
            }
            void main() {
                vec2 s = uCropDst.xy + uv * uCropDst.zw;
                vec3 v = texture(uVanilla, s).rgb;
                vec3 g = uHaveLut != 0 ? clamp(v + texture(uLut, v).rgb, 0.0, 1.0) : v;
                float w = 1.0;
                vec2 su = uv;
                if (uReproject != 0) {
                    float d = texture(uCur, s).r;
                    if (d < 0.0) w = texture(uSrcDist, uv).r < 0.0 ? 1.0 : 0.0;    // hand: same place on screen
                    else if (uTexelSnap != 0 && d < SKY * 0.1 && uFocalPx / (16.0 * d) > 1.5) w = snapped(s, d, su);
                    else w = reproject(s, d, su);
                }
                // frequency separation: the network paints the low band, the game supplies the high band
                vec3 vl = texture(uLow, uv).rgb;
                vec3 detail = pow(clamp((v + 0.03) / (vl + 0.03), 0.25, 4.0), vec3(uDetail));
                vec3 col = w > 0.0 ? mix(g, clamp(texture(uHist, su).rgb * detail, 0.0, 1.0), w) : g;
                o = vec4(col, uStrength);
            }
            """;

    // ------------------------------------------------------------ setup

    private void ensureBase() {
        if (vao >= 0) {
            return;
        }
        vao = GL30.glGenVertexArrays();
        progWorld = program(FS_WORLD);
        progCur = program(FS_CUR);
        progCap = program(FS_CAP);
        progHist = program(FS_HIST);
        progShow = program(FS_SHOW);
        int[] t = target(W, H, GL30.GL_R32F, GL11.GL_RED, GL11.GL_FLOAT, false);
        capDistFbo = t[0];
        capDistTex = t[1];
        t = target(W, H, GL30.GL_R32F, GL11.GL_RED, GL11.GL_FLOAT, false);
        histDistFbo = t[0];
        histDistTex = t[1];
        for (int i = 0; i < 2; i++) {
            t = target(W, H, GL11.GL_RGBA8, GL11.GL_RGBA, GL11.GL_UNSIGNED_BYTE, true);
            histFbo[i] = t[0];
            histTex[i] = t[1];
        }
        resultTex = target(W, H, GL11.GL_RGBA8, GL11.GL_RGBA, GL11.GL_UNSIGNED_BYTE, true)[1];
        t = target(W * 2, H * 2, GL11.GL_RGBA8, GL11.GL_RGBA, GL11.GL_UNSIGNED_BYTE, true);
        vmidFbo = t[0];
        vmidTex = t[1];
        t = target(W, H, GL11.GL_RGBA8, GL11.GL_RGBA, GL11.GL_UNSIGNED_BYTE, true);
        vlowFbo = t[0];
        vlowTex = t[1];
        lutTex = GL11.glGenTextures();
        GL11.glBindTexture(GL12.GL_TEXTURE_3D, lutTex);
        GL11.glTexParameteri(GL12.GL_TEXTURE_3D, GL11.GL_TEXTURE_MIN_FILTER, GL11.GL_LINEAR);
        GL11.glTexParameteri(GL12.GL_TEXTURE_3D, GL11.GL_TEXTURE_MAG_FILTER, GL11.GL_LINEAR);
        GL11.glTexParameteri(GL12.GL_TEXTURE_3D, GL11.GL_TEXTURE_WRAP_S, GL12.GL_CLAMP_TO_EDGE);
        GL11.glTexParameteri(GL12.GL_TEXTURE_3D, GL11.GL_TEXTURE_WRAP_T, GL12.GL_CLAMP_TO_EDGE);
        GL11.glTexParameteri(GL12.GL_TEXTURE_3D, GL12.GL_TEXTURE_WRAP_R, GL12.GL_CLAMP_TO_EDGE);
        GL12.glTexImage3D(GL12.GL_TEXTURE_3D, 0, GL30.GL_RGBA16F, LUT, LUT, LUT, 0, GL11.GL_RGBA, GL11.GL_FLOAT,
                (ByteBuffer) null);
        GL11.glBindTexture(GL12.GL_TEXTURE_3D, 0);
        Restyler.LOG.info("Restyler: stabilizer shaders ready");
    }

    private void ensureSize(int w, int h) {
        ensureBase();
        if (w == fw && h == fh) {
            return;
        }
        delete(worldFbo, worldTex);
        delete(curFbo, curTex);
        delete(vanFbo, vanTex);
        int[] t = target(w, h, GL11.GL_RGBA8, GL11.GL_RGBA, GL11.GL_UNSIGNED_BYTE, false);
        vanFbo = t[0];
        vanTex = t[1];
        t = target(w, h, GL30.GL_RG32F, GL30.GL_RG, GL11.GL_FLOAT, false);
        worldFbo = t[0];
        worldTex = t[1];
        t = target(w, h, GL30.GL_R32F, GL11.GL_RED, GL11.GL_FLOAT, false);
        curFbo = t[0];
        curTex = t[1];
        fw = w;
        fh = h;
        histValid = false;
    }

    void reset() {
        histValid = false;
    }

    boolean hasHistory() {
        return histValid;
    }

    // ------------------------------------------------------------ passes

    /** WorldRenderEvents.END: remember the camera and turn the depth buffer into distances. */
    void onWorldEnd(Framebuffer main, Matrix4f proj, Matrix4f view, double x, double y, double z) {
        ensureSize(main.textureWidth, main.textureHeight);
        curCam.set(proj, view, x, y, z);
        try (Pass p = new Pass(worldFbo, 0, 0, fw, fh, progWorld)) {
            p.tex("uDepth", 0, main.getDepthAttachment());
            p.mat("uInvPv", curCam.invPv);
            p.draw();
        }
        main.beginWrite(true);
        worldReady = true;
    }

    /**
     * End of GameRenderer.renderWorld: the depth buffer was cleared and holds ONLY the hand / held item now
     * (before the GUI clears it again) -> hand mask. Hand pixels get distance -1 and are never re-projected.
     */
    void handPass(Framebuffer main) {
        if (!worldReady || main.textureWidth != fw || main.textureHeight != fh) {
            return;
        }
        try (Pass p = new Pass(curFbo, 0, 0, fw, fh, progCur)) {
            p.tex("uDepth", 0, main.getDepthAttachment());
            p.tex("uWorld", 1, worldTex);
            p.draw();
        }
        main.beginWrite(true);
        curReady = true;
    }

    /** HUD time: true if this frame's distances + hand mask are ready. */
    boolean ready(Framebuffer main) {
        return curReady && main.textureWidth == fw && main.textureHeight == fh;
    }

    void endFrame() {
        worldReady = false;
        curReady = false;
    }

    /** Remember distances + camera of the frame that is being sent to the network. */
    void captureDist(Framebuffer main, float[] crop) {
        try (Pass p = new Pass(capDistFbo, 0, 0, W, H, progCap)) {
            p.tex("uCur", 0, curTex);
            p.vec4("uCrop", crop);
            p.draw();
        }
        main.beginWrite(true);
        capCam.set(curCam);
    }

    /** Network result (RGBA8, rows bottom-up) arrived: upload and blend into the history. */
    void addResult(Framebuffer main, ByteBuffer rgba, FloatBuffer lut, float[] crop, float alpha, boolean reproject) {
        GlStateManager._bindTexture(resultTex);
        GlStateManager._pixelStore(GL11.GL_UNPACK_ROW_LENGTH, 0);
        GlStateManager._pixelStore(GL11.GL_UNPACK_SKIP_ROWS, 0);
        GlStateManager._pixelStore(GL11.GL_UNPACK_SKIP_PIXELS, 0);
        GlStateManager._pixelStore(GL11.GL_UNPACK_ALIGNMENT, 4);
        GL11.glTexSubImage2D(GL11.GL_TEXTURE_2D, 0, 0, 0, W, H, GL11.GL_RGBA, GL11.GL_UNSIGNED_BYTE, rgba);
        GlStateManager._bindTexture(0);
        if (lut != null) {
            GL11.glBindTexture(GL12.GL_TEXTURE_3D, lutTex);
            GL12.glTexSubImage3D(GL12.GL_TEXTURE_3D, 0, 0, 0, 0, LUT, LUT, LUT, GL11.GL_RGBA, GL11.GL_FLOAT, lut);
            GL11.glBindTexture(GL12.GL_TEXTURE_3D, 0);
            haveLut = true;
        }

        int dst = 1 - histIdx;
        try (Pass p = new Pass(histFbo[dst], 0, 0, W, H, progHist)) {
            p.tex("uNew", 0, resultTex);
            p.tex("uNewDist", 1, capDistTex);
            p.tex("uHist", 2, histTex[histIdx]);
            p.tex("uSrcDist", 3, histDistTex);
            p.vec4("uCropDst", crop);
            p.vec4("uCropSrc", crop);
            p.mat("uInvPvDst", capCam.invPv);
            p.mat("uPvSrc", histCam.pv);
            p.delta("uDelta", capCam, histCam);
            p.f("uTol", 0.05f);
            p.f("uFeather", FEATHER);
            p.f("uAlpha", alpha);
            p.i("uUseHist", histValid && reproject && alpha < 0.999f ? 1 : 0);
            p.draw();
        }
        // the history now lives in the capture's camera
        GL30.glBindFramebuffer(GL30.GL_READ_FRAMEBUFFER, capDistFbo);
        GL30.glBindFramebuffer(GL30.GL_DRAW_FRAMEBUFFER, histDistFbo);
        GL30.glBlitFramebuffer(0, 0, W, H, 0, 0, W, H, GL11.GL_COLOR_BUFFER_BIT, GL11.GL_NEAREST);
        main.beginWrite(true);
        histIdx = dst;
        histCam.set(capCam);
        histValid = true;
    }

    /** Draw the history over the world (main framebuffer, crop rectangle). */
    void show(Framebuffer main, int cx, int cy, int cw, int ch, float[] crop, float strength, boolean reproject,
              float detail, boolean texelSnap) {
        if (!histValid) {
            return;
        }
        // the shader can't read the framebuffer it draws into -> copy the current (vanilla) frame first
        GL30.glBindFramebuffer(GL30.GL_READ_FRAMEBUFFER, main.fbo);
        GL30.glBindFramebuffer(GL30.GL_DRAW_FRAMEBUFFER, vanFbo);
        GL30.glBlitFramebuffer(0, 0, fw, fh, 0, 0, fw, fh, GL11.GL_COLOR_BUFFER_BIT, GL11.GL_NEAREST);
        if (detail > 0f) {   // the low band exactly like the capture: crop -> 1024x576 -> 512x288
            GL30.glBindFramebuffer(GL30.GL_READ_FRAMEBUFFER, vanFbo);
            GL30.glBindFramebuffer(GL30.GL_DRAW_FRAMEBUFFER, vmidFbo);
            GL30.glBlitFramebuffer(cx, cy, cx + cw, cy + ch, 0, 0, W * 2, H * 2, GL11.GL_COLOR_BUFFER_BIT, GL11.GL_LINEAR);
            GL30.glBindFramebuffer(GL30.GL_READ_FRAMEBUFFER, vmidFbo);
            GL30.glBindFramebuffer(GL30.GL_DRAW_FRAMEBUFFER, vlowFbo);
            GL30.glBlitFramebuffer(0, 0, W * 2, H * 2, 0, 0, W, H, GL11.GL_COLOR_BUFFER_BIT, GL11.GL_LINEAR);
        }
        main.beginWrite(false);
        RenderSystem.disableDepthTest();
        RenderSystem.enableBlend();
        RenderSystem.blendFuncSeparate(GlStateManager.SrcFactor.SRC_ALPHA, GlStateManager.DstFactor.ONE_MINUS_SRC_ALPHA,
                GlStateManager.SrcFactor.ZERO, GlStateManager.DstFactor.ONE);
        try (Pass p = new Pass(main.fbo, cx, cy, cw, ch, progShow)) {
            p.tex("uCur", 0, curTex);
            p.tex("uHist", 1, histTex[histIdx]);
            p.tex("uSrcDist", 2, histDistTex);
            p.tex("uVanilla", 3, vanTex);
            p.tex("uLow", 5, vlowTex);
            p.f("uDetail", detail);
            GlStateManager._activeTexture(GL13.GL_TEXTURE4);
            GL11.glBindTexture(GL12.GL_TEXTURE_3D, lutTex);
            GL20.glUniform1i(GL20.glGetUniformLocation(progShow, "uLut"), 4);
            p.i("uHaveLut", haveLut ? 1 : 0);
            p.vec4("uCropDst", crop);
            p.vec4("uCropSrc", crop);
            p.mat("uInvPvDst", curCam.invPv);
            p.mat("uPvSrc", histCam.pv);
            p.delta("uDelta", curCam, histCam);
            p.f("uTol", 0.05f);
            p.f("uFeather", FEATHER);
            p.f("uStrength", strength);
            p.i("uReproject", reproject ? 1 : 0);
            p.i("uTexelSnap", texelSnap ? 1 : 0);
            GL20.glUniform3f(GL20.glGetUniformLocation(progShow, "uCamFrac"), (float) (curCam.x - 16 * Math.floor(curCam.x / 16)),
                    (float) (curCam.y - 16 * Math.floor(curCam.y / 16)), (float) (curCam.z - 16 * Math.floor(curCam.z / 16)));
            p.f("uFocalPx", curCam.f11 * fh / 2f);
            p.draw();
            GlStateManager._activeTexture(GL13.GL_TEXTURE4);
            GL11.glBindTexture(GL12.GL_TEXTURE_3D, 0);
        }
        RenderSystem.defaultBlendFunc();
        RenderSystem.disableBlend();
        main.beginWrite(true);
    }

    // ------------------------------------------------------------ GL helpers

    /** One full-viewport triangle into fbo with program prog; restores the game's GL state on close. */
    private final class Pass implements AutoCloseable {
        private final int prog;
        private int units = 0;
        private final boolean blend;
        private final boolean depth;

        Pass(int fbo, int x, int y, int w, int h, int prog) {
            this.prog = prog;
            blend = GL11.glIsEnabled(GL11.GL_BLEND);
            depth = GL11.glIsEnabled(GL11.GL_DEPTH_TEST);
            GlStateManager._glBindFramebuffer(GL30.GL_FRAMEBUFFER, fbo);
            GlStateManager._viewport(x, y, w, h);
            if (prog != progShow) {
                RenderSystem.disableBlend();
            }
            RenderSystem.disableDepthTest();
            BufferRenderer.reset();               // the game's cached vertex array is no longer bound after this
            GlStateManager._glBindVertexArray(vao);
            GlStateManager._glUseProgram(prog);
        }

        void tex(String name, int unit, int tex) {
            GlStateManager._activeTexture(GL13.GL_TEXTURE0 + unit);
            GlStateManager._bindTexture(tex);
            GL20.glUniform1i(GL20.glGetUniformLocation(prog, name), unit);
            units = Math.max(units, unit + 1);
        }

        void mat(String name, Matrix4f m) {
            try (MemoryStack s = MemoryStack.stackPush()) {
                FloatBuffer fb = s.mallocFloat(16);
                m.get(fb);
                GL20.glUniformMatrix4fv(GL20.glGetUniformLocation(prog, name), false, fb);
            }
        }

        void vec4(String name, float[] v) {
            GL20.glUniform4f(GL20.glGetUniformLocation(prog, name), v[0], v[1], v[2], v[3]);
        }

        /** destination camera position - source camera position (doubles: world coordinates can be large) */
        void delta(String name, Cam dst, Cam src) {
            GL20.glUniform3f(GL20.glGetUniformLocation(prog, name),
                    (float) (dst.x - src.x), (float) (dst.y - src.y), (float) (dst.z - src.z));
        }

        void f(String name, float v) {
            GL20.glUniform1f(GL20.glGetUniformLocation(prog, name), v);
        }

        void i(String name, int v) {
            GL20.glUniform1i(GL20.glGetUniformLocation(prog, name), v);
        }

        void draw() {
            GL11.glDrawArrays(GL11.GL_TRIANGLES, 0, 3);
        }

        @Override
        public void close() {
            for (int u = units - 1; u >= 0; u--) {
                GlStateManager._activeTexture(GL13.GL_TEXTURE0 + u);
                GlStateManager._bindTexture(0);
            }
            GlStateManager._activeTexture(GL13.GL_TEXTURE0);
            GlStateManager._glUseProgram(0);
            GlStateManager._glBindVertexArray(0);
            if (blend) {
                RenderSystem.enableBlend();
            } else {
                RenderSystem.disableBlend();
            }
            if (depth) {
                RenderSystem.enableDepthTest();
            } else {
                RenderSystem.disableDepthTest();
            }
        }
    }

    private static int program(String fs) {
        int vs = shader(GL20.GL_VERTEX_SHADER, VS);
        int f = shader(GL20.GL_FRAGMENT_SHADER, fs);
        int p = GL20.glCreateProgram();
        GL20.glAttachShader(p, vs);
        GL20.glAttachShader(p, f);
        GL30.glBindFragDataLocation(p, 0, "o");
        GL20.glLinkProgram(p);
        if (GL20.glGetProgrami(p, GL20.GL_LINK_STATUS) == GL11.GL_FALSE) {
            throw new IllegalStateException("Restyler shader link failed: " + GL20.glGetProgramInfoLog(p));
        }
        GL20.glDeleteShader(vs);
        GL20.glDeleteShader(f);
        return p;
    }

    private static int shader(int type, String src) {
        int s = GL20.glCreateShader(type);
        GL20.glShaderSource(s, src);
        GL20.glCompileShader(s);
        if (GL20.glGetShaderi(s, GL20.GL_COMPILE_STATUS) == GL11.GL_FALSE) {
            throw new IllegalStateException("Restyler shader compile failed: " + GL20.glGetShaderInfoLog(s));
        }
        return s;
    }

    /** texture + framebuffer; linear = bilinear filter (colour), else nearest (distances). */
    private static int[] target(int w, int h, int internal, int format, int type, boolean linear) {
        int tex = GL11.glGenTextures();
        GlStateManager._bindTexture(tex);
        int filter = linear ? GL11.GL_LINEAR : GL11.GL_NEAREST;
        GL11.glTexParameteri(GL11.GL_TEXTURE_2D, GL11.GL_TEXTURE_MIN_FILTER, filter);
        GL11.glTexParameteri(GL11.GL_TEXTURE_2D, GL11.GL_TEXTURE_MAG_FILTER, filter);
        GL11.glTexParameteri(GL11.GL_TEXTURE_2D, GL11.GL_TEXTURE_WRAP_S, GL12.GL_CLAMP_TO_EDGE);
        GL11.glTexParameteri(GL11.GL_TEXTURE_2D, GL11.GL_TEXTURE_WRAP_T, GL12.GL_CLAMP_TO_EDGE);
        GL11.glTexImage2D(GL11.GL_TEXTURE_2D, 0, internal, w, h, 0, format, type, (ByteBuffer) null);
        GlStateManager._bindTexture(0);
        int fbo = GL30.glGenFramebuffers();
        GL30.glBindFramebuffer(GL30.GL_FRAMEBUFFER, fbo);
        GL30.glFramebufferTexture2D(GL30.GL_FRAMEBUFFER, GL30.GL_COLOR_ATTACHMENT0, GL11.GL_TEXTURE_2D, tex, 0);
        int status = GL30.glCheckFramebufferStatus(GL30.GL_FRAMEBUFFER);
        if (status != GL30.GL_FRAMEBUFFER_COMPLETE) {
            throw new IllegalStateException("Restyler: framebuffer incomplete, status " + status);
        }
        return new int[]{fbo, tex};
    }

    private static void delete(int fbo, int tex) {
        if (fbo >= 0) {
            GL30.glDeleteFramebuffers(fbo);
        }
        if (tex >= 0) {
            GlStateManager._bindTexture(0);
            GL11.glDeleteTextures(tex);
        }
    }
}
