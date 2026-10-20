package dev.swipixel.restyler;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.mojang.blaze3d.platform.GlStateManager;
import com.mojang.blaze3d.systems.RenderSystem;
import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.gl.Framebuffer;
import net.minecraft.client.gui.DrawContext;
import net.minecraft.client.texture.NativeImage;
import net.minecraft.client.texture.NativeImageBackedTexture;
import net.minecraft.client.util.math.MatrixStack;
import net.minecraft.client.world.ClientWorld;
import net.minecraft.text.Text;
import net.minecraft.util.Identifier;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.biome.Biome;
import org.joml.Matrix4f;
import org.lwjgl.opengl.GL11;
import org.lwjgl.opengl.GL12;
import org.lwjgl.opengl.GL15;
import org.lwjgl.opengl.GL21;
import org.lwjgl.opengl.GL30;
import org.lwjgl.opengl.GL32;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Reader;
import java.io.Writer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Properties;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/**
 * Restyler v3: one network (restyler_v3_epXX.onnx, inputs "frame" [1,3,288,512] + "style" [1,512]), two looks.
 *
 *   NORMAL  = style vector of "photorealistic"   (realistic materials)
 *   WETCRAFT= style vector of "rain"             (wet, glossy, reflections)
 *   in between: style = normalize(lerp(normal, wet, t))   (the student was trained on these mixes)
 *
 * Look AUTO: t follows the rain, but only in biomes where it rains (not deserts, not snowy biomes, not the
 * Nether/End). t rises with the rain (~5 s) and dries after it stops (dry_seconds, default 20 s).
 *
 * ASYNC (default): the game never waits for the network.
 *   capture (crop -> 1024x576 -> 512x288, read back through a PBO, no stall) -> worker thread runs the network
 *   -> result is blended into a history and, EVERY game frame, re-projected to the current camera with the depth
 *   buffer (Stabilizer). The look updates ~20x/s, the camera moves at full FPS, highlights stay on their blocks.
 * SYNC (async=false in the config): the old way - capture, run, draw, every frame, the game waits.
 */
public final class Restyler {

    public static final Restyler INSTANCE = new Restyler();
    public static final Logger LOG = LoggerFactory.getLogger("restyler");

    private static final int W = 512;
    private static final int H = 288;
    private static final int PLANE = W * H;
    private static final Identifier OUTPUT_ID = Identifier.of("restyler", "output");

    public enum Look {
        AUTO("Auto (weather)"), NORMAL("Normal"), WET("WetCraft");
        public final String title;

        Look(String title) {
            this.title = title;
        }

        Look next() {
            return values()[(ordinal() + 1) % values().length];
        }
    }

    // ---- settings: config/restyler/restyler.properties ----
    private boolean enabled = true;
    private Look look = Look.AUTO;
    private float strength = 1.0f;
    private boolean keepDark = true;     // dark frames (night, caves): no brightening beyond 1.2x the game
    private int dehaze = 0;              // 0 off, 1 levels (stretch the darkest 1 % back to black)
    private int runEveryNFrames = 1;
    private boolean useGpu = true;
    private boolean autoCuda = true;
    private String modelFile = ModelFetcher.MODEL;   // the release model (empty = newest restyler_v3_epXX.onnx)
    private String normalLabel = "photorealistic";
    private String wetLabel = "rain";
    private float autoMaxWet = 1.0f;     // t at full rain in AUTO
    private float wetSeconds = 5f;       // AUTO: time to get fully wet once rain is at full strength
    private float drySeconds = 20f;      // AUTO: time to dry from fully wet after the rain stopped
    private boolean profile = false;
    private boolean async = false;       // PERFORMANCE MODE: network on its own thread + re-projection (lower quality)
    private boolean stabilizer = true;   // re-project + blend with history (false = newest result, no correction)
    private float stabNew = 0.5f;        // weight of a new network result in the history (1 = no smoothing)
    private boolean devignette = true;   // divide out the model's screen-fixed vignette (vignette.json)
    private boolean lightLock = true;    // large-scale brightness stays within [lockMin, lockMax] x the game's
    private boolean texelSnap = true;    // look read once per block texel (paint locked to the block grid)
    private float detail = 1.0f;         // full-res game texture detail on top of the look (0 = off, 1 = full)
    private float lockMin = 0.7f, lockMax = 1.25f;
    private volatile float[] vigNormal, vigWet;   // log2 correction per cell, VIG_GX x VIG_GY
    private static final int VIG_GX = 32, VIG_GY = 18;

    private Path dir;
    private Path configFile;

    // ---- look state ----
    private float wet = 0f;              // current mix t, 0 = normal, 1 = WetCraft
    private boolean raining = false;     // AUTO: rain falls at the player's biome right now
    private float[] vNormal, vWet;
    private FloatBuffer styleBuf;

    // ---- ONNX Runtime ----
    private OrtEnvironment env;
    private volatile OrtSession session;
    private String frameName, styleName;
    private String provider = "-";
    private String loadedModelName = "-";
    private boolean loadFailed = false;
    private volatile boolean cudaDownloading = false;
    private volatile boolean modelDownloading = false;
    private int profiledRuns = 0;

    // ---- OpenGL / buffers ----
    private int midFbo = -1, midTex = -1, midW, midH;
    private int smallFbo = -1, smallTex = -1;
    private final ByteBuffer readBuf = ByteBuffer.allocateDirect(PLANE * 4).order(ByteOrder.nativeOrder());
    private final FloatBuffer inputBuf =
            ByteBuffer.allocateDirect(3 * PLANE * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
    private final float[] rgbOut = new float[3 * PLANE];
    private final int[] hist = new int[256];
    private float blackEma = -1f;
    private NativeImageBackedTexture outputTex;
    private boolean haveOutput = false;

    // ---- stats (smoothed ms): capture, network, post, draw ----
    private long frameCounter = 0;
    private final double[] tAvg = new double[4];
    private volatile long tNetNs, tPostNs;

    // ---- async pipeline ----
    private final Stabilizer stab = new Stabilizer();
    private final Object sessionLock = new Object();
    private final ExecutorService worker = Executors.newSingleThreadExecutor(r -> {
        Thread t = new Thread(r, "restyler-network");
        t.setDaemon(true);
        return t;
    });
    private enum Stage { IDLE, READBACK, RUNNING }
    private Stage stage = Stage.IDLE;
    private int pbo = -1;
    private long fence = 0;
    private final byte[] capBytes = new byte[PLANE * 4];             // RGBA, rows bottom-up
    private Future<ByteBuffer> pending;
    private final FloatBuffer wLut = ByteBuffer.allocateDirect(Stabilizer.LUT * Stabilizer.LUT * Stabilizer.LUT * 16)
            .order(ByteOrder.nativeOrder()).asFloatBuffer();
    private final float[] lutAcc = new float[Stabilizer.LUT * Stabilizer.LUT * Stabilizer.LUT * 3];
    private final int[] lutCnt = new int[Stabilizer.LUT * Stabilizer.LUT * Stabilizer.LUT];
    private final float[] lutPrev = new float[Stabilizer.LUT * Stabilizer.LUT * Stabilizer.LUT * 3];
    private boolean lutPrevValid = false;
    private final float[] crop = new float[4];                        // crop rect in 0..1 screen coords
    private long lastResultNs = 0;
    private double resultHz = 0;
    // owned by the worker thread
    private final FloatBuffer wIn = ByteBuffer.allocateDirect(3 * PLANE * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
    private final float[] wOut = new float[3 * PLANE];
    private final ByteBuffer wRgba = ByteBuffer.allocateDirect(PLANE * 4).order(ByteOrder.nativeOrder());
    private FloatBuffer wStyle;

    private Restyler() {
    }

    // ------------------------------------------------------------
    // settings
    // ------------------------------------------------------------

    public void init() {
        dir = FabricLoader.getInstance().getConfigDir().resolve("restyler");
        configFile = dir.resolve("restyler.properties");
        try {
            Files.createDirectories(dir);
            if (Files.exists(configFile)) {
                Properties p = new Properties();
                try (Reader r = Files.newBufferedReader(configFile)) {
                    p.load(r);
                }
                enabled = bool(p, "enabled", enabled);
                look = parseLook(p.getProperty("look", look.name()));
                strength = clamp01(num(p, "strength", strength));
                keepDark = bool(p, "keep_dark", keepDark);
                dehaze = Math.max(0, Math.min(1, (int) num(p, "dehaze", dehaze)));
                runEveryNFrames = Math.max(1, Math.min(4, (int) num(p, "run_every_n_frames", runEveryNFrames)));
                useGpu = bool(p, "use_gpu", useGpu);
                autoCuda = bool(p, "auto_cuda", autoCuda);
                modelFile = p.getProperty("model", modelFile).trim();
                normalLabel = p.getProperty("normal_style", normalLabel).trim();
                wetLabel = p.getProperty("wet_style", wetLabel).trim();
                autoMaxWet = clamp01(num(p, "auto_max_wet", autoMaxWet));
                wetSeconds = Math.max(0.5f, num(p, "wet_seconds", wetSeconds));
                drySeconds = Math.max(0.5f, num(p, "dry_seconds", drySeconds));
                profile = bool(p, "profile", profile);
                async = bool(p, "async", async);
                stabilizer = bool(p, "stabilizer", stabilizer);
                stabNew = Math.max(0.05f, Math.min(1f, num(p, "stabilizer_new", stabNew)));
                devignette = bool(p, "devignette", devignette);
                lightLock = bool(p, "light_lock", lightLock);
                detail = Math.max(0f, Math.min(1.5f, num(p, "detail", detail)));
                texelSnap = bool(p, "texel_snap", texelSnap);
                lockMin = Math.max(0.05f, Math.min(1f, num(p, "light_lock_min", lockMin)));
                lockMax = Math.max(1f, Math.min(4f, num(p, "light_lock_max", lockMax)));
            }
            save();
        } catch (Exception e) {
            LOG.warn("Restyler: could not read {}, using defaults", configFile, e);
        }
        LOG.info("Restyler: put restyler_v3_epXX.onnx + styles.json into {}", dir);
    }

    private void save() {
        try {
            Properties p = new Properties();
            p.setProperty("enabled", Boolean.toString(enabled));
            p.setProperty("look", look.name());
            p.setProperty("strength", fmt(strength));
            p.setProperty("keep_dark", Boolean.toString(keepDark));
            p.setProperty("dehaze", Integer.toString(dehaze));
            p.setProperty("run_every_n_frames", Integer.toString(runEveryNFrames));
            p.setProperty("use_gpu", Boolean.toString(useGpu));
            p.setProperty("auto_cuda", Boolean.toString(autoCuda));
            p.setProperty("model", modelFile);
            p.setProperty("normal_style", normalLabel);
            p.setProperty("wet_style", wetLabel);
            p.setProperty("auto_max_wet", fmt(autoMaxWet));
            p.setProperty("wet_seconds", fmt(wetSeconds));
            p.setProperty("dry_seconds", fmt(drySeconds));
            p.setProperty("profile", Boolean.toString(profile));
            p.setProperty("async", Boolean.toString(async));
            p.setProperty("stabilizer", Boolean.toString(stabilizer));
            p.setProperty("stabilizer_new", fmt(stabNew));
            p.setProperty("devignette", Boolean.toString(devignette));
            p.setProperty("light_lock", Boolean.toString(lightLock));
            p.setProperty("detail", fmt(detail));
            p.setProperty("texel_snap", Boolean.toString(texelSnap));
            p.setProperty("light_lock_min", fmt(lockMin));
            p.setProperty("light_lock_max", fmt(lockMax));
            try (Writer w = Files.newBufferedWriter(configFile)) {
                p.store(w, "Restyler v3. look=AUTO|NORMAL|WET. model= empty -> newest restyler_v3_epXX.onnx here. "
                        + "keep_dark: no night vision in dark frames. dehaze=0|1. run_every_n_frames=1..4. "
                        + "AUTO: wet follows rain in rainy biomes; wet_seconds/dry_seconds = how fast it gets wet/dries; "
                        + "auto_max_wet = wetness at full rain. normal_style/wet_style = labels in styles.json. "
                        + "async = Performance mode: network on its own thread, full FPS but lower quality (false = full quality, "
                        + "uses run_every_n_frames). stabilizer: re-project the look with the depth buffer; "
                        + "stabilizer_new = weight of each new network result (lower = calmer, slower to react).");
            }
        } catch (Exception e) {
            LOG.warn("Restyler: could not save {}", configFile, e);
        }
    }

    private static boolean bool(Properties p, String k, boolean def) {
        return Boolean.parseBoolean(p.getProperty(k, Boolean.toString(def)).trim());
    }

    private static float num(Properties p, String k, float def) {
        try {
            return Float.parseFloat(p.getProperty(k, Float.toString(def)).trim());
        } catch (NumberFormatException e) {
            return def;
        }
    }

    private static String fmt(float v) {
        return String.format(Locale.ROOT, "%.2f", v);
    }

    private static Look parseLook(String s) {
        String t = s == null ? "" : s.trim().toUpperCase(Locale.ROOT);
        if (t.startsWith("WET")) {
            return Look.WET;
        }
        if (t.startsWith("NORM") || t.startsWith("REAL")) {
            return Look.NORMAL;
        }
        return Look.AUTO;
    }

    // ------------------------------------------------------------
    // controls (keys, menu, /restyler)
    // ------------------------------------------------------------

    public void toggle() {
        setEnabled(!enabled);
        say("Restyler " + (enabled ? "ON" : "OFF") + status(), true);
    }

    public boolean isEnabled() {
        return enabled;
    }

    public void setEnabled(boolean on) {
        enabled = on;
        if (on) {
            loadFailed = false;   // allow another load attempt, e.g. after copying the model
            stab.reset();         // the old history is from another place / time
        }
        save();
    }

    public boolean isTexelSnap() {
        return texelSnap;
    }

    public void setTexelSnap(boolean on) {
        texelSnap = on;
        save();
    }

    public float getDetail() {
        return detail;
    }

    public void setDetail(float d) {
        detail = Math.max(0f, Math.min(1.5f, d));
        save();
    }

    public boolean isLightLock() {
        return lightLock;
    }

    public void setLightLock(boolean on) {
        lightLock = on;
        haveOutput = false;
        save();
    }

    public boolean isDevignette() {
        return devignette;
    }

    public boolean hasVignette() {
        return vigNormal != null;
    }

    public void setDevignette(boolean on) {
        devignette = on;
        haveOutput = false;
        save();
    }

    public boolean isAsync() {
        return async;
    }

    /** Menu "Stabilizer": ON = async + re-projection (60 FPS), OFF = the clean synchronous mode (network every frame). */
    public boolean isStabilizer() {
        return async;
    }

    public void setStabilizer(boolean on) {
        async = on;
        stabilizer = true;
        stab.reset();
        haveOutput = false;
        save();
    }

    public Look getLook() {
        return look;
    }

    public void cycleLook() {
        setLook(look.next());
        say("Look: " + lookText() + status(), true);
    }

    public void setLook(Look l) {
        look = l;
        // switching by hand is instant; only the weather in AUTO fades
        wet = switch (l) {
            case NORMAL -> 0f;
            case WET -> 1f;
            case AUTO -> autoTarget(MinecraftClient.getInstance());
        };
        haveOutput = false;
        save();
    }

    public String lookText() {
        if (look != Look.AUTO) {
            return look.title;
        }
        return String.format(Locale.ROOT, "Auto - %s, wet %d%%", raining ? "raining" : "dry", Math.round(wet * 100));
    }

    public float getStrength() {
        return strength;
    }

    public void setStrength(float v) {
        strength = clamp01(v);
        save();
    }

    public void changeStrength(float d) {
        setStrength(Math.round((strength + d) * 10f) / 10f);
        say("Restyler strength " + Math.round(strength * 100) + "%", true);
    }

    public boolean isKeepDark() {
        return keepDark;
    }

    public void setKeepDark(boolean on) {
        keepDark = on;
        haveOutput = false;
        save();
    }

    public int getDehaze() {
        return dehaze;
    }

    public void setDehaze(int d) {
        dehaze = Math.max(0, Math.min(1, d));
        blackEma = -1f;
        haveOutput = false;
        save();
    }

    public int getRunEvery() {
        return runEveryNFrames;
    }

    public void setRunEvery(int n) {
        runEveryNFrames = Math.max(1, Math.min(4, n));
        save();
    }

    public boolean isDownloading() {
        return cudaDownloading;
    }

    public String providerName() {
        return provider;
    }

    public double frameMs() {
        return tAvg[0] + tAvg[1] + tAvg[2] + tAvg[3];
    }

    public String loadedModel() {
        return session == null ? (modelDownloading ? "downloading model" : cudaDownloading ? "downloading GPU runtime"
                : "not loaded") : loadedModelName;
    }

    public List<String> modelFiles() {
        try (Stream<Path> files = Files.list(dir)) {
            return files.map(f -> f.getFileName().toString()).filter(f -> f.endsWith(".onnx")).sorted().toList();
        } catch (Exception e) {
            return List.of();
        }
    }

    /** Switch the network without restarting ("" = newest restyler_v3_epXX.onnx). */
    public void useModel(String name) {
        String n = name == null ? "" : name.trim();
        if (!n.isEmpty() && !Files.exists(dir.resolve(n))) {
            say("No file " + n + " in config/restyler. Models: " + String.join(", ", modelFiles()), false);
            return;
        }
        modelFile = n;
        closeSession();
        loadFailed = false;
        haveOutput = false;
        save();
        say("Model: " + (n.isEmpty() ? "newest" : n) + " - loading on the next frame", true);
    }

    public String statusLine() {
        if (cudaDownloading) {
            return "downloading GPU runtime...";
        }
        if (async) {
            return String.format(Locale.ROOT, "%s  network %.0f ms, look %.0fx/s  |  %s", provider, tAvg[1], resultHz,
                    lookText());
        }
        return String.format(Locale.ROOT, "%s  %.0f ms  |  %s", provider, frameMs(), lookText());
    }

    private String status() {
        if (session == null) {
            return "";
        }
        return async ? String.format(Locale.ROOT, "  (%s, network %.0f ms, %.0fx/s)", provider, tAvg[1], resultHz)
                : String.format(Locale.ROOT, "  (%s, %.1f ms)", provider, frameMs());
    }

    // ------------------------------------------------------------
    // per tick: AUTO weather
    // ------------------------------------------------------------

    private float autoTarget(MinecraftClient mc) {
        raining = false;
        if (mc == null || mc.world == null || mc.player == null) {
            return 0f;
        }
        ClientWorld w = mc.world;
        BlockPos pos = mc.player.getBlockPos();
        boolean rainBiome = w.getBiome(pos).value().getPrecipitation(pos) == Biome.Precipitation.RAIN;
        float rain = rainBiome ? w.getRainGradient(1f) : 0f;   // 0..1, Nether/End always 0
        raining = rain > 0.01f;
        return rain * autoMaxWet;
    }

    public void tick(MinecraftClient mc) {
        if (look != Look.AUTO || mc.world == null || mc.player == null || mc.isPaused()) {
            return;
        }
        boolean wasRaining = raining;
        float target = autoTarget(mc);
        if (raining != wasRaining && enabled) {
            say(raining ? "Restyler: rain - getting wet"
                    : String.format(Locale.ROOT, "Restyler: rain stopped - drying in %.0f s", wet * drySeconds), true);
        }
        if (target > wet) {
            wet = Math.min(target, wet + 1f / (20f * wetSeconds));
        } else {
            wet = Math.max(target, wet - 1f / (20f * drySeconds));
        }
    }

    /** style = normalize((1 - t) * normal + t * wet) */
    private void updateStyle() {
        float t = clamp01(wet);
        double n2 = 0;
        for (int i = 0; i < vNormal.length; i++) {
            double v = (1f - t) * vNormal[i] + t * vWet[i];
            n2 += v * v;
        }
        float inv = (float) (1.0 / Math.max(1e-8, Math.sqrt(n2)));
        for (int i = 0; i < vNormal.length; i++) {
            styleBuf.put(i, ((1f - t) * vNormal[i] + t * vWet[i]) * inv);
        }
    }

    // ------------------------------------------------------------
    // per frame
    // ------------------------------------------------------------

    public void onBeforeHud(DrawContext context) {
        if (!enabled || strength <= 0f) {
            return;
        }
        MinecraftClient mc = MinecraftClient.getInstance();
        if (mc.world == null || !ensureLoaded(mc)) {
            return;
        }
        Framebuffer main = mc.getFramebuffer();
        int fw = main.textureWidth;
        int fh = main.textureHeight;

        // centre 16:9 crop in framebuffer pixels (the crop the network was trained on)
        int cw = fw;
        int ch = fh;
        if ((long) fw * H > (long) fh * W) {
            cw = Math.round(fh * (float) W / H);
        } else {
            ch = Math.round(fw * (float) H / W);
        }
        int cx = (fw - cw) / 2;
        int cy = (fh - ch) / 2;

        if (async) {
            try {
                frameAsync(main, cx, cy, cw, ch, fw, fh);
            } catch (Throwable t) {
                asyncFailed(t);
            }
            return;
        }
        if (pending != null) {   // just switched from async: let the worker's last run finish before using the model here
            try {
                pending.get();
            } catch (Exception ignored) {
            }
            pending = null;
        }
        stage = Stage.IDLE;
        try {
            if (frameCounter++ % runEveryNFrames == 0 || !haveOutput) {
                long t0 = System.nanoTime();
                capture(main, cx, cy, cw, ch);
                long t1 = System.nanoTime();
                runNetwork();
                tAdd(0, (t1 - t0) / 1e6);
                tAdd(1, tNetNs / 1e6);
                tAdd(2, tPostNs / 1e6);
            }
            long t2 = System.nanoTime();
            draw(context, cx, cy, cw, ch, fw, fh);
            tAdd(3, (System.nanoTime() - t2) / 1e6);
            if (frameCounter % 600 == 1) {
                LOG.info(String.format(Locale.ROOT,
                        "Restyler timing (ms): capture %.1f | network %.1f | post %.1f | draw %.1f | total %.1f (%s, %dx%d, wet %.2f)",
                        tAvg[0], tAvg[1], tAvg[2], tAvg[3], frameMs(), provider, fw, fh, wet));
            }
        } catch (Throwable t) {
            LOG.error("Restyler: turned off after an error", t);
            enabled = false;
            say("Restyler turned off after an error, see the log: " + t, false);
        }
    }

    private void tAdd(int k, double ms) {
        tAvg[k] = tAvg[k] == 0 ? ms : 0.95 * tAvg[k] + 0.05 * ms;
    }

    // ------------------------------------------------------------
    // ASYNC: the game never waits for the network
    // ------------------------------------------------------------

    /** WorldRenderEvents.END (world drawn, hand not yet): camera + depth -> distances. */
    public void onWorldEnd(Matrix4f proj, Matrix4f view, double x, double y, double z) {
        if (!enabled || !async || session == null) {
            return;
        }
        try {
            stab.onWorldEnd(MinecraftClient.getInstance().getFramebuffer(), proj, view, x, y, z);
        } catch (Throwable t) {
            asyncFailed(t);
        }
    }

    /** End of GameRenderer.renderWorld (hand drawn, GUI not yet): hand mask for the stabilizer. */
    public void afterHand() {
        if (!enabled || !async || session == null) {
            return;
        }
        try {
            stab.handPass(MinecraftClient.getInstance().getFramebuffer());
        } catch (Throwable t) {
            asyncFailed(t);
        }
    }

    private void frameAsync(Framebuffer main, int cx, int cy, int cw, int ch, int fw, int fh) throws Exception {
        crop[0] = cx / (float) fw;
        crop[1] = cy / (float) fh;
        crop[2] = cw / (float) fw;
        crop[3] = ch / (float) fh;
        if (!stab.ready(main)) {
            stab.endFrame();
            return;   // world/hand pass didn't run this frame (e.g. first frame, resize)
        }
        long t0 = System.nanoTime();

        // 1. pixels of the last capture arrived on the CPU -> network (worker thread)
        if (stage == Stage.READBACK && fenceDone()) {
            readPbo();
            updateStyle();
            float[] style = new float[styleBuf.capacity()];
            styleBuf.get(0, style);
            final float tJob = wet;
            pending = worker.submit(() -> runJob(style, tJob));
            stage = Stage.RUNNING;
        }
        // 2. network finished -> blend into the history
        if (stage == Stage.RUNNING && pending.isDone()) {
            ByteBuffer rgba = null;
            try {
                rgba = pending.get();
            } catch (ExecutionException e) {
                LOG.error("Restyler: network run failed", e.getCause());
            }
            pending = null;
            stage = Stage.IDLE;
            if (rgba != null) {
                stab.addResult(main, rgba, wLut.rewind(), crop, stabilizer ? stabNew : 1f, stabilizer);
                long now = System.nanoTime();
                if (lastResultNs > 0) {
                    double hz = 1e9 / (now - lastResultNs);
                    resultHz = resultHz == 0 ? hz : 0.9 * resultHz + 0.1 * hz;
                }
                lastResultNs = now;
                tAdd(1, tNetNs / 1e6);
                tAdd(2, tPostNs / 1e6);
            }
        }
        // 3. nothing in flight -> capture this frame
        if (stage == Stage.IDLE) {
            captureAsync(main, cx, cy, cw, ch);
            stab.captureDist(main, crop);
            stage = Stage.READBACK;
        }
        long t1 = System.nanoTime();
        tAdd(0, (t1 - t0) / 1e6);

        // 4. every frame: the look, re-projected to where the camera is now
        stab.show(main, cx, cy, cw, ch, crop, strength, stabilizer, detail, texelSnap);
        stab.endFrame();
        tAdd(3, (System.nanoTime() - t1) / 1e6);

        if (frameCounter++ % 600 == 1) {
            LOG.info(String.format(Locale.ROOT,
                    "Restyler timing (ms): render thread %.1f | network %.1f | post %.1f | show %.1f | look %.1fx/s "
                            + "(%s, %dx%d, stabilizer %s, wet %.2f)",
                    tAvg[0], tAvg[1], tAvg[2], tAvg[3], resultHz, provider, fw, fh, stabilizer ? "on" : "off", wet));
        }
    }

    /** crop -> 1024x576 -> 512x288 on the GPU, then glReadPixels into a PBO: returns at once, no stall. */
    private void captureAsync(Framebuffer main, int cx, int cy, int cw, int ch) {
        blitToSmall(main, cx, cy, cw, ch);
        if (pbo < 0) {
            pbo = GL15.glGenBuffers();
            GL15.glBindBuffer(GL21.GL_PIXEL_PACK_BUFFER, pbo);
            GL15.glBufferData(GL21.GL_PIXEL_PACK_BUFFER, (long) PLANE * 4, GL15.GL_STREAM_READ);
        }
        GL15.glBindBuffer(GL21.GL_PIXEL_PACK_BUFFER, pbo);
        GlStateManager._pixelStore(GL11.GL_PACK_ROW_LENGTH, 0);
        GlStateManager._pixelStore(GL11.GL_PACK_SKIP_ROWS, 0);
        GlStateManager._pixelStore(GL11.GL_PACK_SKIP_PIXELS, 0);
        GlStateManager._pixelStore(GL11.GL_PACK_ALIGNMENT, 4);
        GL11.glReadPixels(0, 0, W, H, GL11.GL_RGBA, GL11.GL_UNSIGNED_BYTE, 0L);
        GL15.glBindBuffer(GL21.GL_PIXEL_PACK_BUFFER, 0);
        fence = GL32.glFenceSync(GL32.GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        main.beginWrite(true);
    }

    private boolean fenceDone() {
        if (fence == 0) {
            return false;
        }
        int r = GL32.glClientWaitSync(fence, 0, 0L);
        if (r == GL32.GL_ALREADY_SIGNALED || r == GL32.GL_CONDITION_SATISFIED) {
            GL32.glDeleteSync(fence);
            fence = 0;
            return true;
        }
        return false;
    }

    private void readPbo() {
        GL15.glBindBuffer(GL21.GL_PIXEL_PACK_BUFFER, pbo);
        ByteBuffer m = GL30.glMapBufferRange(GL21.GL_PIXEL_PACK_BUFFER, 0, (long) PLANE * 4, GL30.GL_MAP_READ_BIT);
        if (m != null) {
            m.get(0, capBytes);
            GL15.glUnmapBuffer(GL21.GL_PIXEL_PACK_BUFFER);
        }
        GL15.glBindBuffer(GL21.GL_PIXEL_PACK_BUFFER, 0);
    }

    /** Worker thread: RGBA bottom-up -> network -> post -> RGBA bottom-up (GL row order, ready to upload). */
    private ByteBuffer runJob(float[] style, float tWet) throws Exception {
        for (int y = 0; y < H; y++) {
            int src = (H - 1 - y) * W * 4;
            int dst = y * W;
            for (int x = 0; x < W; x++) {
                int i = src + x * 4;
                int o = dst + x;
                wIn.put(o, (capBytes[i] & 0xFF) / 255f);
                wIn.put(PLANE + o, (capBytes[i + 1] & 0xFF) / 255f);
                wIn.put(2 * PLANE + o, (capBytes[i + 2] & 0xFF) / 255f);
            }
        }
        if (wStyle == null || wStyle.capacity() != style.length) {
            wStyle = ByteBuffer.allocateDirect(style.length * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
        }
        wStyle.put(0, style);
        long n0;
        long n1;
        synchronized (sessionLock) {
            if (session == null) {
                return null;   // model is being switched
            }
            Map<String, OnnxTensor> feeds = new HashMap<>();
            try (OnnxTensor frame = OnnxTensor.createTensor(env, wIn.rewind(), new long[]{1, 3, H, W});
                 OnnxTensor st = OnnxTensor.createTensor(env, wStyle.rewind(), new long[]{1, style.length})) {
                feeds.put(frameName, frame);
                feeds.put(styleName, st);
                n0 = System.nanoTime();
                try (OrtSession.Result result = session.run(feeds)) {
                    n1 = System.nanoTime();
                    if (profile && ++profiledRuns == 300) {
                        LOG.info("Restyler: ONNX layer profile written to {}", session.endProfiling());
                        profile = false;
                    }
                    FloatBuffer out = ((OnnxTensor) result.get(0)).getFloatBuffer();
                    for (int i = 0; i < 3 * PLANE; i++) {
                        float v = out.get(i);
                        wOut[i] = v >= 0f ? (v > 1f ? 1f : v) : 0f;   // NaN -> 0
                    }
                }
            }
        }
        applyDevignette(wOut, tWet);
        applyLightLock(wIn, wOut);
        if (keepDark) {
            keepDarkGuard(wIn, wOut);
        }
        if (dehaze >= 1) {
            levels(wOut);
        }
        for (int y = 0; y < H; y++) {
            int row = (H - 1 - y) * W * 4;
            for (int x = 0; x < W; x++) {
                int o = y * W + x;
                int i = row + x * 4;
                wRgba.put(i, (byte) toByte(wOut[o]));
                wRgba.put(i + 1, (byte) toByte(wOut[PLANE + o]));
                wRgba.put(i + 2, (byte) toByte(wOut[2 * PLANE + o]));
                wRgba.put(i + 3, (byte) 255);
            }
        }
        buildLut();
        tNetNs = n1 - n0;
        tPostNs = System.nanoTime() - n1;
        return wRgba.rewind();
    }

    /**
     * Worker: 16x16x16 colour LUT "vanilla rgb -> what the network adds", from this result's 147k pixel pairs.
     * Empty cells are filled from their neighbours, then smoothed over results (no flicker). In game it grades
     * every pixel the detailed look doesn't cover yet (fast turns, newly visible areas).
     */
    private void buildLut() {
        final int L = Stabilizer.LUT;
        final int n = L * L * L;
        java.util.Arrays.fill(lutAcc, 0f);
        java.util.Arrays.fill(lutCnt, 0);
        double gr = 0, gg = 0, gb = 0;
        for (int o = 0; o < PLANE; o++) {
            float r = wIn.get(o), g = wIn.get(PLANE + o), b = wIn.get(2 * PLANE + o);
            float dr = wOut[o] - r, dg = wOut[PLANE + o] - g, db = wOut[2 * PLANE + o] - b;
            int c = (Math.min(L - 1, (int) (b * L)) * L + Math.min(L - 1, (int) (g * L))) * L + Math.min(L - 1, (int) (r * L));
            lutAcc[3 * c] += dr;
            lutAcc[3 * c + 1] += dg;
            lutAcc[3 * c + 2] += db;
            lutCnt[c]++;
            gr += dr;
            gg += dg;
            gb += db;
        }
        float[] cur = new float[3 * n];
        boolean[] has = new boolean[n];
        for (int c = 0; c < n; c++) {
            if (lutCnt[c] >= 4) {
                for (int k = 0; k < 3; k++) {
                    cur[3 * c + k] = lutAcc[3 * c + k] / lutCnt[c];
                }
                has[c] = true;
            }
        }
        for (int pass = 0; pass < 6; pass++) {   // grow into empty cells from filled neighbours
            boolean[] next = has.clone();
            for (int c = 0; c < n; c++) {
                if (has[c]) {
                    continue;
                }
                int x = c % L, y = (c / L) % L, z = c / (L * L);
                float sr = 0, sg = 0, sb = 0;
                int m = 0;
                int[][] nb = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
                for (int[] d : nb) {
                    int xx = x + d[0], yy = y + d[1], zz = z + d[2];
                    if (xx < 0 || yy < 0 || zz < 0 || xx >= L || yy >= L || zz >= L) {
                        continue;
                    }
                    int q = (zz * L + yy) * L + xx;
                    if (has[q]) {
                        sr += cur[3 * q];
                        sg += cur[3 * q + 1];
                        sb += cur[3 * q + 2];
                        m++;
                    }
                }
                if (m > 0) {
                    cur[3 * c] = sr / m;
                    cur[3 * c + 1] = sg / m;
                    cur[3 * c + 2] = sb / m;
                    next[c] = true;
                }
            }
            has = next;
        }
        float mr = (float) (gr / PLANE), mg = (float) (gg / PLANE), mb = (float) (gb / PLANE);
        for (int c = 0; c < n; c++) {
            if (!has[c]) {   // colours far from anything on screen: the average change
                cur[3 * c] = mr;
                cur[3 * c + 1] = mg;
                cur[3 * c + 2] = mb;
            }
            for (int k = 0; k < 3; k++) {
                float v = lutPrevValid ? 0.5f * lutPrev[3 * c + k] + 0.5f * cur[3 * c + k] : cur[3 * c + k];
                lutPrev[3 * c + k] = v;
                wLut.put(4 * c + k, v);
            }
            wLut.put(4 * c + 3, 0f);
        }
        lutPrevValid = true;
    }

    /** Something in the async path broke: log it and fall back to the old synchronous mode. */
    private void asyncFailed(Throwable t) {
        LOG.error("Restyler: async mode failed, back to the simple mode", t);
        async = false;
        stage = Stage.IDLE;
        pending = null;
        fence = 0;
        say("Restyler: stabilizer failed (" + t + ") - back to the simple mode. Send me the log.", false);
    }

    // ---- 1 + 2: shrink on the GPU (crop -> 1024x576 -> 512x288, like the training pack) and read back ----

    private void capture(Framebuffer main, int cx, int cy, int cw, int ch) {
        blitToSmall(main, cx, cy, cw, ch);
        readBuf.clear();
        GL11.glReadPixels(0, 0, W, H, GL11.GL_RGBA, GL11.GL_UNSIGNED_BYTE, readBuf);
        main.beginWrite(false);   // give the screen back to the game

        // RGBA bytes, rows bottom-up -> float planes [1,3,H,W], rows top-down, 0..1
        for (int y = 0; y < H; y++) {
            int src = (H - 1 - y) * W * 4;
            int dst = y * W;
            for (int x = 0; x < W; x++) {
                int i = src + x * 4;
                int o = dst + x;
                inputBuf.put(o, (readBuf.get(i) & 0xFF) / 255f);
                inputBuf.put(PLANE + o, (readBuf.get(i + 1) & 0xFF) / 255f);
                inputBuf.put(2 * PLANE + o, (readBuf.get(i + 2) & 0xFF) / 255f);
            }
        }
    }

    /** crop -> 1024x576 -> 512x288 (same as the training pack); leaves smallFbo bound for reading. */
    private void blitToSmall(Framebuffer main, int cx, int cy, int cw, int ch) {
        int wantMidW = Math.min(cw, W * 2);
        int wantMidH = Math.min(ch, H * 2);
        if (midFbo < 0 || midW != wantMidW || midH != wantMidH) {
            deleteTarget(midFbo, midTex);
            int[] t = createTarget(wantMidW, wantMidH);
            midFbo = t[0];
            midTex = t[1];
            midW = wantMidW;
            midH = wantMidH;
        }
        if (smallFbo < 0) {
            int[] t = createTarget(W, H);
            smallFbo = t[0];
            smallTex = t[1];
        }
        GL30.glBindFramebuffer(GL30.GL_READ_FRAMEBUFFER, main.fbo);
        GL30.glBindFramebuffer(GL30.GL_DRAW_FRAMEBUFFER, midFbo);
        GL30.glBlitFramebuffer(cx, cy, cx + cw, cy + ch, 0, 0, midW, midH, GL11.GL_COLOR_BUFFER_BIT, GL11.GL_LINEAR);
        GL30.glBindFramebuffer(GL30.GL_READ_FRAMEBUFFER, midFbo);
        GL30.glBindFramebuffer(GL30.GL_DRAW_FRAMEBUFFER, smallFbo);
        GL30.glBlitFramebuffer(0, 0, midW, midH, 0, 0, W, H, GL11.GL_COLOR_BUFFER_BIT, GL11.GL_LINEAR);
        GL30.glBindFramebuffer(GL30.GL_READ_FRAMEBUFFER, smallFbo);
    }

    private static int[] createTarget(int w, int h) {
        int tex = GL11.glGenTextures();
        GlStateManager._bindTexture(tex);
        GL11.glTexParameteri(GL11.GL_TEXTURE_2D, GL11.GL_TEXTURE_MIN_FILTER, GL11.GL_LINEAR);
        GL11.glTexParameteri(GL11.GL_TEXTURE_2D, GL11.GL_TEXTURE_MAG_FILTER, GL11.GL_LINEAR);
        GL11.glTexParameteri(GL11.GL_TEXTURE_2D, GL11.GL_TEXTURE_WRAP_S, GL12.GL_CLAMP_TO_EDGE);
        GL11.glTexParameteri(GL11.GL_TEXTURE_2D, GL11.GL_TEXTURE_WRAP_T, GL12.GL_CLAMP_TO_EDGE);
        GL11.glTexImage2D(GL11.GL_TEXTURE_2D, 0, GL11.GL_RGBA8, w, h, 0, GL11.GL_RGBA, GL11.GL_UNSIGNED_BYTE,
                (ByteBuffer) null);
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

    private static void deleteTarget(int fbo, int tex) {
        if (fbo >= 0) {
            GL30.glDeleteFramebuffers(fbo);
        }
        if (tex >= 0) {
            GlStateManager._bindTexture(0);
            GL11.glDeleteTextures(tex);
        }
    }

    // ---- 3 + 4: network, post-processing, upload ----

    private void runNetwork() throws Exception {
        updateStyle();
        Map<String, OnnxTensor> feeds = new HashMap<>();
        try (OnnxTensor frame = OnnxTensor.createTensor(env, inputBuf.rewind(), new long[]{1, 3, H, W});
             OnnxTensor style = OnnxTensor.createTensor(env, styleBuf.rewind(), new long[]{1, styleBuf.capacity()})) {
            feeds.put(frameName, frame);
            feeds.put(styleName, style);
            long n0 = System.nanoTime();
            try (OrtSession.Result result = session.run(feeds)) {
                long n1 = System.nanoTime();
                if (profile && ++profiledRuns == 300) {
                    LOG.info("Restyler: ONNX layer profile written to {}", session.endProfiling());
                    profile = false;
                }
                fillOutput(((OnnxTensor) result.get(0)).getFloatBuffer());
                tNetNs = n1 - n0;
                tPostNs = System.nanoTime() - n1;
            }
        }
    }

    private void fillOutput(FloatBuffer out) {
        for (int i = 0; i < 3 * PLANE; i++) {
            float v = out.get(i);
            rgbOut[i] = v >= 0f ? (v > 1f ? 1f : v) : 0f;   // NaN -> 0
        }
        applyDevignette(rgbOut, wet);
        applyLightLock(inputBuf, rgbOut);
        if (keepDark) {
            keepDarkGuard(inputBuf, rgbOut);
        }
        if (dehaze >= 1) {
            levels(rgbOut);
        }
        NativeImage img = outputTex.getImage();
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                int o = y * W + x;
                int r = toByte(rgbOut[o]);
                int g = toByte(rgbOut[PLANE + o]);
                int b = toByte(rgbOut[2 * PLANE + o]);
                img.setColor(x, y, 0xFF000000 | (b << 16) | (g << 8) | r);   // ABGR
            }
        }
        outputTex.upload();
        haveOutput = true;
    }

    /**
     * The darker the whole frame (night, caves), the less a pixel may get brighter than the game drew it:
     * at most 1.2x + a little. Bright daytime frames are not affected.
     */
    private static void keepDarkGuard(FloatBuffer inputBuf, float[] rgbOut) {
        double sum = 0;
        for (int o = 0; o < PLANE; o += 7) {
            sum += 0.299f * inputBuf.get(o) + 0.587f * inputBuf.get(PLANE + o) + 0.114f * inputBuf.get(2 * PLANE + o);
        }
        float mean = (float) (sum / ((PLANE + 6) / 7));
        float guard = Math.max(0f, Math.min(1f, (0.25f - mean) / (0.25f - 0.08f)));   // 1 = dark, 0 = day
        if (guard <= 0f) {
            return;
        }
        for (int o = 0; o < PLANE; o++) {
            float lin = 0.299f * inputBuf.get(o) + 0.587f * inputBuf.get(PLANE + o) + 0.114f * inputBuf.get(2 * PLANE + o);
            float lout = 0.299f * rgbOut[o] + 0.587f * rgbOut[PLANE + o] + 0.114f * rgbOut[2 * PLANE + o];
            float cap = lin * 1.2f + 0.015f;
            float allowed = cap + (1f - guard) * Math.max(0f, lout - cap);
            if (lout > allowed && lout > 1e-5f) {
                float k = allowed / lout;
                rgbOut[o] *= k;
                rgbOut[PLANE + o] *= k;
                rgbOut[2 * PLANE + o] *= k;
            }
        }
    }

    /** Stretch the darkest 1 % (smoothed over frames, no flicker) back to black. */
    private void levels(float[] rgbOut) {
        java.util.Arrays.fill(hist, 0);
        for (int o = 0; o < PLANE; o++) {
            float l = 0.299f * rgbOut[o] + 0.587f * rgbOut[PLANE + o] + 0.114f * rgbOut[2 * PLANE + o];
            hist[Math.min(255, (int) (l * 255f + 0.5f))]++;
        }
        int target = PLANE / 100;
        int acc = 0;
        int b = 0;
        for (; b < 255; b++) {
            acc += hist[b];
            if (acc >= target) {
                break;
            }
        }
        blackEma = blackEma < 0f ? b / 255f : 0.9f * blackEma + 0.1f * (b / 255f);
        float black = Math.min(blackEma, 0.6f);
        float gain = 1f / (1f - black);
        for (int i = 0; i < 3 * PLANE; i++) {
            float v = (rgbOut[i] - black) * gain;
            rgbOut[i] = v < 0f ? 0f : (v > 1f ? 1f : v);
        }
    }

    private static int toByte(float v) {
        int i = Math.round(v * 255f);
        return i < 0 ? 0 : Math.min(i, 255);
    }

    // ---- 5: draw over the world ----

    private void draw(DrawContext context, int cx, int cy, int cw, int ch, int fw, int fh) {
        if (!haveOutput) {
            return;
        }
        MatrixStack m = context.getMatrices();
        m.push();
        m.scale(context.getScaledWindowWidth() / (float) fw, context.getScaledWindowHeight() / (float) fh, 1f);
        RenderSystem.disableDepthTest();
        RenderSystem.enableBlend();
        RenderSystem.defaultBlendFunc();
        RenderSystem.setShaderColor(1f, 1f, 1f, strength);
        context.drawTexture(OUTPUT_ID, cx, cy, cw, ch, 0f, 0f, W, H, W, H);
        RenderSystem.setShaderColor(1f, 1f, 1f, 1f);
        RenderSystem.disableBlend();
        m.pop();
    }

    // ------------------------------------------------------------
    // loading
    // ------------------------------------------------------------

    private Path pickModel() {
        if (!modelFile.isEmpty() && Files.exists(dir.resolve(modelFile))) {
            return dir.resolve(modelFile);
        }
        // newest epoch of restyler_v3_epXX.onnx, else any .onnx
        Pattern ep = Pattern.compile("restyler_v3_ep(\\d+)\\.onnx");
        Path best = null;
        int bestEp = -1;
        for (String f : modelFiles()) {
            Matcher m = ep.matcher(f);
            int e = m.matches() ? Integer.parseInt(m.group(1)) : -1;
            if (best == null || e > bestEp) {
                best = dir.resolve(f);
                bestEp = e;
            }
        }
        return best;
    }

    /** styles.json from the trainer's v3_out (same 48 labels as before, so config/snowfilter's copy also works). */
    private boolean loadStyles() {
        Path json = dir.resolve("styles.json");
        if (!Files.exists(json)) {
            json = dir.getParent().resolve("snowfilter").resolve("styles.json");
        }
        if (!Files.exists(json)) {
            say("Restyler: copy styles.json (from v3_out) into " + dir, false);
            return false;
        }
        try (Reader r = Files.newBufferedReader(json, StandardCharsets.UTF_8)) {
            JsonObject o = JsonParser.parseReader(r).getAsJsonObject();
            JsonArray labels = o.getAsJsonArray("labels");
            JsonArray vecs = o.getAsJsonArray("vectors");
            vNormal = vWet = null;
            for (int i = 0; i < labels.size(); i++) {
                String l = labels.get(i).getAsString();
                if (l.equalsIgnoreCase(normalLabel)) {
                    vNormal = floats(vecs.get(i).getAsJsonArray());
                }
                if (l.equalsIgnoreCase(wetLabel)) {
                    vWet = floats(vecs.get(i).getAsJsonArray());
                }
            }
            if (vNormal == null || vWet == null) {
                say("Restyler: styles.json has no \"" + (vNormal == null ? normalLabel : wetLabel) + "\"", false);
                return false;
            }
            styleBuf = ByteBuffer.allocateDirect(vNormal.length * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
            LOG.info("Restyler: style vectors from {} ({} = normal, {} = wet, dim {})", json, normalLabel, wetLabel,
                    vNormal.length);
            return true;
        } catch (Exception e) {
            LOG.error("Restyler: could not read {}", json, e);
            say("Restyler: could not read styles.json, see the log: " + e, false);
            return false;
        }
    }

    /** vignette.json from vignette_v3.py: per-look log2 brightness bias by screen cell (absent = no fix). */
    private void loadVignette() {
        vigNormal = vigWet = null;
        Path json = dir.resolve("vignette.json");
        if (!Files.exists(json)) {
            return;
        }
        try (Reader r = Files.newBufferedReader(json, StandardCharsets.UTF_8)) {
            JsonObject looks = JsonParser.parseReader(r).getAsJsonObject().getAsJsonObject("looks");
            float[] n = looks.has(normalLabel) ? floats(looks.getAsJsonArray(normalLabel)) : null;
            float[] w = looks.has(wetLabel) ? floats(looks.getAsJsonArray(wetLabel)) : n;
            if (n != null && n.length == VIG_GX * VIG_GY && w.length == n.length) {
                vigNormal = n;
                vigWet = w;
                LOG.info("Restyler: vignette correction loaded from {}", json);
            }
        } catch (Exception e) {
            LOG.warn("Restyler: could not read {}", json, e);
        }
    }

    /** out /= 2^bias(x, y): bias bilinear from the 32x18 grid, normal/wet maps mixed by t. */
    private void applyDevignette(float[] out, float t) {
        float[] vn = vigNormal, vw = vigWet;
        if (!devignette || vn == null) {
            return;
        }
        float[] cell = new float[vn.length];
        for (int i = 0; i < cell.length; i++) {
            cell[i] = -((1f - t) * vn[i] + t * vw[i]);
        }
        applyCellLog2(out, cell);
    }

    /**
     * Light lock: per 16x16 px patch, the network's brightness may be at most lockMin..lockMax x the game's.
     * Materials, texture and colour stay; WHERE it is light stays the game's (torches, sky light), so the model
     * can't light the cave like a flashlight held by the player.
     */
    private void applyLightLock(FloatBuffer in, float[] out) {
        if (!lightLock) {
            return;
        }
        final int cw = W / VIG_GX, ch = H / VIG_GY;   // 16 x 16
        final float eps = 0.02f;
        float[] cell = new float[VIG_GX * VIG_GY];
        for (int cy = 0; cy < VIG_GY; cy++) {
            for (int cx = 0; cx < VIG_GX; cx++) {
                double li = 0, lo = 0;
                for (int y = cy * ch; y < (cy + 1) * ch; y++) {
                    int o = y * W + cx * cw;
                    for (int x = 0; x < cw; x++, o++) {
                        li += 0.299f * in.get(o) + 0.587f * in.get(PLANE + o) + 0.114f * in.get(2 * PLANE + o);
                        lo += 0.299f * out[o] + 0.587f * out[PLANE + o] + 0.114f * out[2 * PLANE + o];
                    }
                }
                double n = cw * ch;
                double r = (lo / n + eps) / (li / n + eps);
                double rc = Math.max(lockMin, Math.min(lockMax, r));
                cell[cy * VIG_GX + cx] = (float) (Math.log(rc / r) / Math.log(2.0));
            }
        }
        applyCellLog2(out, cell);
    }

    /** out *= 2^cell(x, y), cell bilinear from the 32x18 grid; clamps to 1. */
    private static void applyCellLog2(float[] out, float[] cell) {
        float[] rowGain = new float[W];
        for (int y = 0; y < H; y++) {
            float fy = (y + 0.5f) * VIG_GY / H - 0.5f;
            int y0 = Math.max(0, Math.min(VIG_GY - 1, (int) Math.floor(fy)));
            int y1 = Math.min(VIG_GY - 1, y0 + 1);
            float wy = Math.max(0f, Math.min(1f, fy - y0));
            for (int x = 0; x < W; x++) {
                float fx = (x + 0.5f) * VIG_GX / W - 0.5f;
                int x0 = Math.max(0, Math.min(VIG_GX - 1, (int) Math.floor(fx)));
                int x1 = Math.min(VIG_GX - 1, x0 + 1);
                float wx = Math.max(0f, Math.min(1f, fx - x0));
                float top = cell[y0 * VIG_GX + x0] * (1 - wx) + cell[y0 * VIG_GX + x1] * wx;
                float bot = cell[y1 * VIG_GX + x0] * (1 - wx) + cell[y1 * VIG_GX + x1] * wx;
                rowGain[x] = (float) Math.pow(2.0, top * (1 - wy) + bot * wy);
            }
            int o = y * W;
            for (int c = 0; c < 3; c++) {
                int base = c * PLANE + o;
                for (int x = 0; x < W; x++) {
                    float v = out[base + x] * rowGain[x];
                    out[base + x] = v > 1f ? 1f : v;
                }
            }
        }
    }

    private static float[] floats(JsonArray a) {
        float[] f = new float[a.size()];
        for (int i = 0; i < f.length; i++) {
            f[i] = a.get(i).getAsFloat();
        }
        return f;
    }

    private boolean ensureLoaded(MinecraftClient mc) {
        if (session != null) {
            return true;
        }
        if (loadFailed || cudaDownloading || modelDownloading) {
            return false;
        }
        loadFailed = true;   // until proven otherwise

        // first start: fetch the model + style vectors (a few MB), then continue
        if (!ModelFetcher.present(dir) && pickModel() == null) {
            if (ModelFetcher.configured()) {
                startModelDownload(mc);
            } else {
                say("Restyler: copy " + ModelFetcher.MODEL + " and styles.json into " + dir + ", then press K twice", false);
            }
            return false;
        }

        if (FabricLoader.getInstance().isModLoaded("snowfilter")) {
            say("Restyler: the old Snow Filter / Restyler 0.1 jar is also in your mods folder - remove "
                    + "snowfilter-0.1.0.jar, otherwise both restyle the same frame (and both use K)", false);
        }
        Path model = pickModel();
        if (model == null) {
            say("Restyler: copy restyler_v3_epXX.onnx and styles.json (from v3_out) into " + dir
                    + ", then press K twice", false);
            return false;
        }
        if (!loadStyles()) {
            return false;
        }
        loadVignette();
        try {
            env = OrtEnvironment.getEnvironment();
        } catch (Throwable t) {
            LOG.error("Restyler: ONNX Runtime could not start", t);
            say("Restyler: ONNX Runtime could not start, see the log: " + t, false);
            return false;
        }

        // NVIDIA runtime: our own copy (downloaded once), or the one the old mod already downloaded
        Path cudaDir = dir.resolve("cuda");
        Path oldCuda = dir.getParent().resolve("snowfilter").resolve("cuda");
        if (!CudaFetcher.present(cudaDir) && CudaFetcher.present(oldCuda)) {
            cudaDir = oldCuda;
        }
        if (useGpu && CudaFetcher.isWindows() && CudaFetcher.present(cudaDir)) {
            CudaFetcher.preload(cudaDir);
        }
        if (useGpu) {
            try {
                OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
                opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
                if (profile) {
                    opts.enableProfiling(dir.resolve("ort_profile").toString());
                }
                opts.addCUDA(0);
                session = env.createSession(model.toString(), opts);
                provider = "GPU";
            } catch (Throwable t) {
                session = null;
                String gpu = GL11.glGetString(GL11.GL_VENDOR) + " " + GL11.glGetString(GL11.GL_RENDERER);
                if (autoCuda && CudaFetcher.isWindows() && !CudaFetcher.present(cudaDir)
                        && gpu.toLowerCase(Locale.ROOT).contains("nvidia")) {
                    LOG.warn("Restyler: CUDA not found - downloading the NVIDIA runtime once", t);
                    startCudaDownload(mc, dir.resolve("cuda"));
                    return false;
                }
                LOG.warn("Restyler: GPU (CUDA) not available, falling back to the CPU", t);
                say("Restyler: GPU not available (CUDA 12 / cuDNN 9 missing?), using the CPU - see the log", false);
            }
        }
        if (session == null) {
            try {
                session = env.createSession(model.toString(), new OrtSession.SessionOptions());
                provider = "CPU";
            } catch (Throwable t) {
                LOG.error("Restyler: could not load the model", t);
                say("Restyler: could not load " + model.getFileName() + ", see the log: " + t, false);
                return false;
            }
        }

        try {
            frameName = null;
            styleName = null;
            for (String n : session.getInputNames()) {
                if (n.equals("style")) {
                    styleName = n;
                } else if (frameName == null) {
                    frameName = n;
                }
            }
            if (styleName == null || frameName == null) {
                say("Restyler: " + model.getFileName() + " is not a style model (inputs " + session.getInputNames()
                        + "). Use restyler_v3_epXX.onnx.", false);
                closeSession();
                return false;
            }
            loadedModelName = model.getFileName().toString();
            if (outputTex == null) {
                outputTex = new NativeImageBackedTexture(new NativeImage(NativeImage.Format.RGBA, W, H, false));
                outputTex.setFilter(true, false);   // smooth when stretched to full screen
                mc.getTextureManager().registerTexture(OUTPUT_ID, outputTex);
            }
        } catch (Throwable t) {
            LOG.error("Restyler: setup failed", t);
            say("Restyler: setup failed, see the log: " + t, false);
            closeSession();
            return false;
        }
        if (look != Look.AUTO) {
            wet = look == Look.WET ? 1f : 0f;
        } else {
            wet = autoTarget(mc);
        }
        loadFailed = false;
        haveOutput = false;
        LOG.info("Restyler: loaded {} on {} (inputs {})", loadedModelName, provider, session.getInputNames());
        say("Restyler: " + loadedModelName + " on " + provider + ". O menu, U look (Auto/Normal/WetCraft), "
                + "[ ] strength, K on/off   |   " + lookText(), true);
        return true;
    }

    private void closeSession() {
        synchronized (sessionLock) {
            closeSessionLocked();
        }
        stab.reset();
    }

    private void closeSessionLocked() {
        if (session != null) {
            try {
                session.close();
            } catch (Exception e) {
                LOG.warn("Restyler: closing the model failed", e);
            }
        }
        session = null;
    }

    private void startModelDownload(MinecraftClient mc) {
        modelDownloading = true;
        say("Restyler: first start - downloading the model (a few MB)...", true);
        Thread t = new Thread(() -> {
            try {
                ModelFetcher.download(dir, msg -> mc.execute(() -> say(msg, true)));
                mc.execute(() -> {
                    modelDownloading = false;
                    loadFailed = false;   // next frame: load it (and fetch the GPU runtime if needed)
                    say("Restyler: model ready", true);
                });
            } catch (Throwable e) {
                LOG.error("Restyler: model download failed", e);
                mc.execute(() -> {
                    modelDownloading = false;
                    say("Restyler: model download failed (" + e.getMessage() + "). Check your internet and press K "
                            + "twice, or copy " + ModelFetcher.MODEL + " + styles.json into " + dir, false);
                });
            }
        }, "restyler-model-download");
        t.setDaemon(true);
        t.start();
    }

    private void startCudaDownload(MinecraftClient mc, Path cudaDir) {
        if (cudaDownloading) {
            return;
        }
        cudaDownloading = true;
        say("Restyler: first start on this PC - downloading NVIDIA's GPU runtime once (~1.6 GB). "
                + "Keep playing; the filter turns on by itself when it's done.", false);
        Thread t = new Thread(() -> {
            try {
                CudaFetcher.download(cudaDir, msg -> mc.execute(() -> say(msg, true)));
                mc.execute(() -> {
                    cudaDownloading = false;
                    loadFailed = false;
                    say("Restyler: GPU runtime ready - starting the filter", false);
                });
            } catch (Throwable e) {
                LOG.error("Restyler: CUDA download failed", e);
                mc.execute(() -> {
                    cudaDownloading = false;
                    autoCuda = false;   // no retry loop; CPU on the next try
                    loadFailed = false;
                    save();
                    say("Restyler: GPU runtime download failed (" + e.getMessage() + ") - using the CPU. "
                            + "Set auto_cuda=true in restyler.properties to retry.", false);
                });
            }
        }, "restyler-cuda-download");
        t.setDaemon(true);
        t.start();
    }

    // ------------------------------------------------------------

    static void say(String msg, boolean actionBar) {
        MinecraftClient mc = MinecraftClient.getInstance();
        if (mc.player != null) {
            mc.player.sendMessage(Text.literal(msg), actionBar);
        }
        LOG.info(msg);
    }

    private static float clamp01(float v) {
        return Math.max(0f, Math.min(1f, v));
    }
}
