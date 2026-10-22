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
    private boolean async = true;        // PERFORMANCE MODE: network on its own thread + re-projection (lower quality)
    private boolean stabilizer = true;   // re-project + blend with history (false = newest result, no correction)
    private final float stabNew = 1f;    // newest network result only (mixing with older ones only blurred motion)
    private boolean devignette = true;   // divide out the model's screen-fixed vignette (vignette.json)
    private boolean lightLock = true;    // large-scale brightness stays within [lockMin, lockMax] x the game's
    private float detail = 0f;           // full-res game texture detail on top of the look (0 = off, 1 = full)
    private float lockMin = 0.7f, lockMax = 1.25f;
    private boolean partialUpdate = true; // Performance mode: repaint only the part of the screen that changed
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

    // ---- partial update: repaint only what changed (needs <model>_partial.onnx, made by make_partial.py) ----
    private static final int TILE = 32, TILES_X = W / TILE, TILES_Y = H / TILE;   // 16 x 9 tiles of 32 px
    private static final int MARGIN = 48;              // context the network gets around the repainted part (px)
    private static final float DIRTY_FRAC = 0.06f;     // a tile is repainted if > 6 % of its pixels changed
    private static final float DIRTY_THRESH = 0.12f;   // vanilla colour change that counts as "changed"
    private static final float FULL_AREA = 0.55f;      // larger than this -> just repaint everything
    private static final long FULL_EVERY_NS = 2_000_000_000L;   // full repaint at least every 2 s ...
    private static final int FULL_EVERY_JOBS = 10;               // ... and after 10 partial ones (re-projection blurs)
    private volatile boolean partialModel = false;     // the loaded model takes norm_stats / use_stats
    private int statsRows = 0;                         // 2 x number of GroupNorms
    private volatile boolean forceFull = true;         // next job repaints everything (history reset, model change)
    private volatile float[] jobRegion;                // last job: null = whole frame, else x0,y0,x1,y1 (0..1, GL)
    private float[] wStats;                            // worker: GroupNorm statistics of the last full repaint
    private long wLastFullNs = 0;
    private int wSinceFull = 0;
    private float[] wLastStyle;
    private FloatBuffer wCrop;
    private FloatBuffer wStatsBuf;
    private volatile double areaEma = 1.0, skipEma = 0.0;   // share of the frame repainted per run / runs skipped
    private boolean syncPartialFailed = false;

    // ---- sharpen while still: when nothing changes, repaint the screen at DOUBLE resolution, tile by tile ----
    private boolean sharpen = true;
    private static final int RT_W = 128, RT_H = 96, RT_X = W / RT_W, RT_Y = H / RT_H;   // 4 x 3 tiles (1x px)
    private static final int REF_W = 1280, REF_H = 720; // the screen at 2.5x the network's normal resolution
    private static final int RC_W = 488, RC_H = 408;   // 2.5x network input per tile: tile + fade border + context
    private static final long FADE_NS = 200_000_000L;  // a sharpened tile fades in over 0.2 s
    private int refFbo = -1, refTex = -1;              // the 1280x720 capture (only while sharpening)
    private final float[] fadeP = new float[12];       // render thread: fade progress per tile (-1 = none)
    private final long[] fadeStart = new long[12];
    private final int[] fadeW2 = new int[12], fadeH2 = new int[12];
    private final float[][] fadeRect = new float[12][], fadeInner = new float[12][];
    private static final int R_FEATHER = 8;            // 1x px around a tile where it fades in
    private static final float SHARPEN_MAX_CHANGE = 0.25f;  // sharpen between repaints while < 25 % changes
    private boolean capHaveRef = false;                // the capture in flight brought its 1280x720 version
    private final byte[] refBytes = new byte[REF_W * REF_H * 4];
    private final ByteBuffer refReadBuf = ByteBuffer.allocateDirect(REF_W * REF_H * 4).order(ByteOrder.nativeOrder());
    private int pbo2 = -1;
    private volatile boolean jobKeepSharp = false;     // last job (1x): keep the sharp pixels that didn't change
    private boolean wLastWasPartial = false;           // worker: alternate 1x repaint / sharpening while moving
    private volatile double sharpFrac = 0;             // share of the screen that is sharp (from the last capture)
    private volatile int jobRefTile = -1;              // last job: >= 0 = it was a 2x tile
    private volatile int jobCandidates = 0;            // last job: tiles that still wanted sharpening (incl. its own)
    // FOCUS PULL: sharpened tiles are staged (shown out of focus = the 1x look) until the whole screen is done, then
    // the screen focuses at once, like an eye - no tile-by-tile "tetris"
    private static final long REVEAL_NS = 1_000_000_000L;      // the focus pull takes 1 s (slow = unnoticed)
    private static final long REVEAL_WAIT_NS = 1_000_000_000L; // moving: focus what is staged at least every 1 s
    private long stagedSinceNs = -1;                   // render thread: first staged tile not yet in focus
    private long revealStartNs = -1;                   // focus pull running since (-1 = none)
    private boolean debugView = false;                 // colour what is cached (not saved)
    private volatile int jobRefW2, jobRefH2;
    private volatile int[] jobRefRect;                 // its pixels: 1x px, top-down {x0, y0, x1, y1}
    private FloatBuffer wRefIn;
    private final ByteBuffer wRefRgba = ByteBuffer.allocateDirect(Stabilizer.REF_MAX_W * Stabilizer.REF_MAX_H * 4)
            .order(ByteOrder.nativeOrder());

    {
        java.util.Arrays.fill(fadeP, -1f);
    }


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
                devignette = bool(p, "devignette", devignette);
                lightLock = bool(p, "light_lock", lightLock);
                detail = Math.max(0f, Math.min(1.5f, num(p, "detail", detail)));
                partialUpdate = bool(p, "partial_update", partialUpdate);
                sharpen = bool(p, "sharpen_when_still", sharpen);
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
            p.setProperty("devignette", Boolean.toString(devignette));
            p.setProperty("light_lock", Boolean.toString(lightLock));
            p.setProperty("detail", fmt(detail));
            p.setProperty("partial_update", Boolean.toString(partialUpdate));
            p.setProperty("sharpen_when_still", Boolean.toString(sharpen));
            p.setProperty("light_lock_min", fmt(lockMin));
            p.setProperty("light_lock_max", fmt(lockMax));
            try (Writer w = Files.newBufferedWriter(configFile)) {
                p.store(w, "Restyler v3. look=AUTO|NORMAL|WET. model= empty -> newest restyler_v3_epXX.onnx here. "
                        + "keep_dark: no night vision in dark frames. dehaze=0|1. run_every_n_frames=1..4. "
                        + "AUTO: wet follows rain in rainy biomes; wet_seconds/dry_seconds = how fast it gets wet/dries; "
                        + "auto_max_wet = wetness at full rain. normal_style/wet_style = labels in styles.json. "
                        + "async = Performance mode: network on its own thread, full FPS but lower quality (false = full quality, "
                        + "uses run_every_n_frames). stabilizer: re-project the look with the depth buffer. "
                        + "partial_update: repaint only what changed. sharpen_when_still: 2.5x tiles while nothing changes.");
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
            forceFull = true;
        }
        save();
    }

    public boolean isPartialUpdate() {
        return partialUpdate;
    }

    public boolean isSharpen() {
        return sharpen;
    }

    public void setSharpen(boolean on) {
        sharpen = on;
        forceFull = true;     // back to the normal resolution everywhere
        save();
    }

    public boolean isDebugView() {
        return debugView;
    }

    public void setDebugView(boolean on) {
        debugView = on;
    }

    public boolean hasPartialModel() {
        return partialModel;
    }

    public void setPartialUpdate(boolean on) {
        partialUpdate = on;
        syncPartialFailed = false;
        stab.reset();         // the history may be old (it isn't kept up to date in the normal quality mode)
        forceFull = true;
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
        forceFull = true;
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
            return files.map(f -> f.getFileName().toString())
                    .filter(f -> f.endsWith(".onnx") && !f.endsWith("_partial.onnx")).sorted().toList();
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
        if (syncPartial()) {
            try {
                if (frameSyncPartial(main, cx, cy, cw, ch, fw, fh)) {
                    return;
                }
            } catch (Throwable t) {
                LOG.error("Restyler: partial update (quality mode) failed, back to full repaints", t);
                syncPartialFailed = true;
                stab.endFrame();
                say("Restyler: partial update failed (" + t + ") - repainting the full frame again. Send me the log.",
                        false);
            }
        }
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
    /** Quality mode (synchronous) with partial update: needs the depth / camera passes too. */
    private boolean syncPartial() {
        return !async && partialUpdate && partialModel && !syncPartialFailed;
    }

    public void onWorldEnd(Matrix4f proj, Matrix4f view, double x, double y, double z) {
        if (!enabled || !(async || syncPartial()) || session == null) {
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
        if (!enabled || !(async || syncPartial()) || session == null) {
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
            final boolean partialOk = partialUpdate && partialModel && stabilizer && stab.hasHistory();
            final boolean haveRef = capHaveRef;
            final int fading = fadingMask();
            pending = worker.submit(() -> runJob(style, tJob, partialOk, haveRef, fading));
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
            boolean added = integrateResult(main, rgba, stabilizer ? stabNew : 1f, stabilizer);
            afterJob();
            if (added) {
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
            stab.captureDist(main, crop);          // distances + camera first: the dirty mask needs them
            captureAsync(main, cx, cy, cw, ch);
            stage = Stage.READBACK;
        }
        long t1 = System.nanoTime();
        tAdd(0, (t1 - t0) / 1e6);

        // 4. every frame: the look, re-projected to where the camera is now
        stepFades(main);
        float focus = stepReveal(main);
        stab.show(main, cx, cy, cw, ch, crop, strength, stabilizer, detail, focus, debugView);
        stab.endFrame();
        tAdd(3, (System.nanoTime() - t1) / 1e6);

        if (frameCounter++ % 600 == 1) {
            LOG.info(String.format(Locale.ROOT,
                    "Restyler timing (ms): render thread %.1f | network %.1f | post %.1f | show %.1f | look %.1fx/s "
                            + "| repainted %.0f %% per run, %.0f %% of runs unchanged | sharp %.0f %% "
                            + "(%s, %dx%d, partial %s, wet %.2f)",
                    tAvg[0], tAvg[1], tAvg[2], tAvg[3], resultHz, areaEma * 100, skipEma * 100,
                    sharpFrac * 100, provider, fw, fh, partialModel && partialUpdate ? "on" : "off", wet));
        }
    }

    /**
     * QUALITY mode + partial update. Like Performance mode, but the frame waits for the network: capture, find what
     * changed, repaint just that (or nothing), keep the rest from the previous output moved along with the camera.
     * Returns false on the first frame (no depth yet) -> the normal full repaint draws it.
     */
    private boolean frameSyncPartial(Framebuffer main, int cx, int cy, int cw, int ch, int fw, int fh) throws Exception {
        crop[0] = cx / (float) fw;
        crop[1] = cy / (float) fh;
        crop[2] = cw / (float) fw;
        crop[3] = ch / (float) fh;
        if (!stab.ready(main)) {
            stab.endFrame();
            return false;
        }
        long t0 = System.nanoTime();
        stab.captureDist(main, crop);
        blitToSmall(main, cx, cy, cw, ch);
        stab.markDirty(main, smallFbo, crop, DIRTY_THRESH);
        wantRef(cw, ch);
        GL30.glBindFramebuffer(GL30.GL_READ_FRAMEBUFFER, smallFbo);
        GlStateManager._pixelStore(GL11.GL_PACK_ROW_LENGTH, 0);
        GlStateManager._pixelStore(GL11.GL_PACK_SKIP_ROWS, 0);
        GlStateManager._pixelStore(GL11.GL_PACK_SKIP_PIXELS, 0);
        GlStateManager._pixelStore(GL11.GL_PACK_ALIGNMENT, 4);
        readBuf.clear();
        GL11.glReadPixels(0, 0, W, H, GL11.GL_RGBA, GL11.GL_UNSIGNED_BYTE, readBuf);
        readRefine(main, cx, cy, cw, ch, false);
        main.beginWrite(true);
        readBuf.get(0, capBytes);
        long t1 = System.nanoTime();

        updateStyle();
        float[] style = new float[styleBuf.capacity()];
        styleBuf.get(0, style);
        ByteBuffer rgba = runJob(style, wet, stab.hasHistory(), capHaveRef, fadingMask());
        boolean added = integrateResult(main, rgba, 1f, true);
        afterJob();
        if (added) {
            tAdd(1, tNetNs / 1e6);
            tAdd(2, tPostNs / 1e6);
        } else {
            tAdd(1, 0);
            tAdd(2, 0);
        }
        tAdd(0, (t1 - t0) / 1e6);
        long t2 = System.nanoTime();
        stepFades(main);
        float focus = stepReveal(main);
        stab.show(main, cx, cy, cw, ch, crop, strength, true, 0f, focus, debugView);   // no detail layer: like the normal quality mode
        stab.endFrame();
        tAdd(3, (System.nanoTime() - t2) / 1e6);
        if (frameCounter++ % 600 == 1) {
            LOG.info(String.format(Locale.ROOT,
                    "Restyler timing (ms, quality + partial): capture %.1f | network %.1f | post %.1f | show %.1f "
                            + "| total %.1f | repainted %.0f %% per frame, %.0f %% of frames unchanged | sharp %.0f %% "
                            + "(%s, %dx%d, wet %.2f)",
                    tAvg[0], tAvg[1], tAvg[2], tAvg[3], frameMs(), areaEma * 100, skipEma * 100,
                    sharpFrac * 100, provider, fw, fh, wet));
        }
        return true;
    }

    /** Render thread, at capture: bring the 1280x720 version along? (the worker picks the tile to sharpen) */
    private void wantRef(int cw, int ch) {
        // not in WetCraft: at 2.5x its big puddles turn into small glints - keep it as it is
        capHaveRef = sharpen && partialUpdate && partialModel && wet <= 0.3f && cw >= REF_W && ch >= REF_H
                && stab.hasHistory();
    }

    /** The screen crop at 1280x720 (2.5x the network's normal input) -> PBO or refBytes. */
    private void readRefine(Framebuffer main, int cx, int cy, int cw, int ch, boolean toPbo) {
        if (!capHaveRef) {
            return;
        }
        if (refFbo < 0) {
            int[] t = createTarget(REF_W, REF_H);
            refFbo = t[0];
            refTex = t[1];
        }
        GL30.glBindFramebuffer(GL30.GL_READ_FRAMEBUFFER, main.fbo);
        GL30.glBindFramebuffer(GL30.GL_DRAW_FRAMEBUFFER, refFbo);
        GL30.glBlitFramebuffer(cx, cy, cx + cw, cy + ch, 0, 0, REF_W, REF_H, GL11.GL_COLOR_BUFFER_BIT, GL11.GL_LINEAR);
        GL30.glBindFramebuffer(GL30.GL_READ_FRAMEBUFFER, refFbo);
        if (toPbo) {
            if (pbo2 < 0) {
                pbo2 = GL15.glGenBuffers();
                GL15.glBindBuffer(GL21.GL_PIXEL_PACK_BUFFER, pbo2);
                GL15.glBufferData(GL21.GL_PIXEL_PACK_BUFFER, (long) REF_W * REF_H * 4, GL15.GL_STREAM_READ);
            }
            GL15.glBindBuffer(GL21.GL_PIXEL_PACK_BUFFER, pbo2);
            GL11.glReadPixels(0, 0, REF_W, REF_H, GL11.GL_RGBA, GL11.GL_UNSIGNED_BYTE, 0L);
            GL15.glBindBuffer(GL21.GL_PIXEL_PACK_BUFFER, 0);
        } else {
            refReadBuf.clear();
            GL11.glReadPixels(0, 0, REF_W, REF_H, GL11.GL_RGBA, GL11.GL_UNSIGNED_BYTE, refReadBuf);
            refReadBuf.get(0, refBytes);
        }
    }

    /** Tiles that are fading in right now (the worker must not pick them again). */
    private int fadingMask() {
        if (revealStartNs >= 0) {
            return (1 << 12) - 1;    // focusing right now: no new tiles until it's done
        }
        int m = 0;
        for (int t = 0; t < fadeP.length; t++) {
            if (fadeP[t] >= 0f) {
                m |= 1 << t;
            }
        }
        return m;
    }

    private static float[] uvGl(int x0, int y0, int x1, int y1) {   // top-down px -> 0..1 GL (y up)
        return new float[]{x0 / (float) W, (H - y1) / (float) H, x1 / (float) W, (H - y0) / (float) H};
    }

    /** Render thread: a finished job into the history. False = nothing was added. */
    private boolean integrateResult(Framebuffer main, ByteBuffer rgba, float alpha, boolean reproject) {
        if (rgba == null) {
            return false;
        }
        int refT = jobRefTile;
        float[] region = jobRegion;
        if ((refT >= 0 || region != null) && !stab.hasHistory()) {
            forceFull = true;     // the history was reset meanwhile: this piece has nothing to fit into
            return false;
        }
        if (refT >= 0) {
            int[] r = jobRefRect;
            int tx = refT % RT_X, ty = refT / RT_X;
            float[] rect = uvGl(r[0], r[1], r[2], r[3]);
            float[] inner = uvGl(tx * RT_W, ty * RT_H, (tx + 1) * RT_W, (ty + 1) * RT_H);
            // into the STAGE (not on screen yet): it appears with all the others in one slow focus pull
            stab.addRefine(main, rgba, jobRefW2, jobRefH2, rect, inner, crop, refT, 0f, false);
            if (stagedSinceNs < 0) {
                stagedSinceNs = System.nanoTime();
            }
            return true;
        }
        finishFades(main);       // the picture moves with this result: running fades finish now (nothing is lost)
        stab.addResult(main, rgba, region == null ? wLut.rewind() : null, crop, alpha, reproject, region, jobKeepSharp);
        if (region == null && !jobKeepSharp) {   // everything repainted at 1x (style change ...): nothing staged
            stagedSinceNs = -1;
            revealStartNs = -1;
        }
        return true;
    }

    /** After every job: no tile left to sharpen and something staged -> focus now. */
    private void afterJob() {
        if (stagedSinceNs < 0 || revealStartNs >= 0) {
            return;
        }
        int left = jobRefTile >= 0 ? jobCandidates - 1 : jobCandidates;
        if (left <= 0) {
            revealStartNs = System.nanoTime();
        }
    }

    /**
     * Every frame: the focus pull progress (smoothstep 0..1). The staged 2.5x paint's low band (shape, colour, light)
     * fades in with it; its fine detail with it^2.5, because seeing detail is compressive (roughly the square root of
     * its contrast): linear would show most of the change in the first ~20 % - an "event".
     */
    private float stepReveal(Framebuffer main) {
        if (stagedSinceNs < 0 || !stab.hasHistory()) {
            stagedSinceNs = -1;
            revealStartNs = -1;
            return 0f;
        }
        long now = System.nanoTime();
        if (revealStartNs < 0 && now - stagedSinceNs > REVEAL_WAIT_NS) {
            revealStartNs = now;     // moving: don't keep the sharp parts waiting forever
        }
        if (revealStartNs < 0) {
            return 0f;
        }
        float p = Math.min(1f, (now - revealStartNs) / (float) REVEAL_NS);
        if (p >= 1f) {
            stab.commit(main);       // staged detail -> history, sharp
            revealStartNs = -1;
            stagedSinceNs = -1;
            return 1f;
        }
        return p * p * (3f - 2f * p);   // the show shader: low band x this, detail x this^2.5
    }

    /** Every frame: sharpened tiles fade in (current look -> 2.5x) over FADE_NS. */
    private void stepFades(Framebuffer main) {
        long now = System.nanoTime();
        for (int t = 0; t < fadeP.length; t++) {
            float p0 = fadeP[t];
            if (p0 < 0f) {
                continue;
            }
            if (!stab.hasHistory()) {
                fadeP[t] = -1f;
                continue;
            }
            float p1 = Math.min(1f, (now - fadeStart[t]) / (float) FADE_NS);
            p1 = p1 * p1 * (3f - 2f * p1);            // smooth start and end
            if (p1 > p0) {
                float a = (p1 - p0) / (1f - p0);      // linear blend step from what is there now
                stab.fadeRefine(main, t, fadeW2[t], fadeH2[t], fadeRect[t], fadeInner[t], crop, a, p1 >= 1f);
                fadeP[t] = p1;
            }
            if (p1 >= 1f) {
                fadeP[t] = -1f;
            }
        }
    }

    /** Before the history moves (new result): finish every running fade at once, still in the right place. */
    private void finishFades(Framebuffer main) {
        for (int t = 0; t < fadeP.length; t++) {
            if (fadeP[t] >= 0f) {
                if (stab.hasHistory()) {
                    stab.fadeRefine(main, t, fadeW2[t], fadeH2[t], fadeRect[t], fadeInner[t], crop, 1f, true);
                }
                fadeP[t] = -1f;
            }
        }
    }

    /** crop -> 1024x576 -> 512x288 on the GPU, then glReadPixels into a PBO: returns at once, no stall. */
    private void captureAsync(Framebuffer main, int cx, int cy, int cw, int ch) {
        blitToSmall(main, cx, cy, cw, ch);
        stab.markDirty(main, smallFbo, crop, DIRTY_THRESH);   // alpha: changed / not sharp yet / sharp
        wantRef(cw, ch);
        GL30.glBindFramebuffer(GL30.GL_READ_FRAMEBUFFER, smallFbo);
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
        readRefine(main, cx, cy, cw, ch, true);
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
        if (capHaveRef) {
            GL15.glBindBuffer(GL21.GL_PIXEL_PACK_BUFFER, pbo2);
            ByteBuffer m2 = GL30.glMapBufferRange(GL21.GL_PIXEL_PACK_BUFFER, 0, (long) REF_W * REF_H * 4,
                    GL30.GL_MAP_READ_BIT);
            if (m2 != null) {
                m2.get(0, refBytes);
                GL15.glUnmapBuffer(GL21.GL_PIXEL_PACK_BUFFER);
            } else {
                capHaveRef = false;
            }
        }
        GL15.glBindBuffer(GL21.GL_PIXEL_PACK_BUFFER, 0);
    }

    /**
     * Worker thread: RGBA bottom-up -> network -> post -> RGBA bottom-up (GL row order, ready to upload).
     * PARTIAL UPDATE: the capture's alpha says which pixels the history can't supply (new surfaces, or the game's
     * image changed there). Only the 32 px tiles with enough of those are repainted: the network runs on that
     * rectangle + MARGIN of context, normalised with the GroupNorm statistics of the last full repaint (so the piece
     * matches the rest of the picture). Nothing changed -> no network run at all (returns null, jobRegion = null).
     */
    private ByteBuffer runJob(float[] style, float tWet, boolean partialOk, boolean haveRef, int fading)
            throws Exception {
        jobRegion = null;
        jobRefTile = -1;
        jobKeepSharp = false;
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

        // ---- full or partial?
        long now = System.nanoTime();
        boolean styleChanged = wLastStyle == null || wLastStyle.length != style.length;
        for (int i = 0; !styleChanged && i < style.length; i++) {
            styleChanged = Math.abs(style[i] - wLastStyle[i]) > 1e-4f;
        }
        boolean hardFull = !partialOk || wStats == null || forceFull || styleChanged;   // everything new
        boolean full = hardFull || wSinceFull >= FULL_EVERY_JOBS;    // routine refresh: keeps sharp pixels
        jobKeepSharp = !hardFull;
        int bx0 = 0, by0 = 0, bx1 = W, by1 = H;      // part to repaint, top-down pixels
        // the capture's alpha: 255 = changed (repaint), 128 = same but not sharp yet, 0 = same and sharp
        int[] cnt = new int[TILES_X * TILES_Y];
        int[] rUnsharp = new int[RT_X * RT_Y], rDirty = new int[RT_X * RT_Y];
        int sharpPx = 0;
        for (int y = 0; y < H; y++) {                 // GL rows: y = 0 is the bottom
            int yt = H - 1 - y;
            int ty = yt / TILE;
            int row = y * W * 4 + 3;
            for (int x = 0; x < W; x++) {
                int a = capBytes[row + x * 4] & 0xFF;
                int rt = (yt / RT_H) * RT_X + x / RT_W;
                if (a > 191) {
                    cnt[ty * TILES_X + x / TILE]++;
                    rDirty[rt]++;
                } else if (a > 63) {
                    rUnsharp[rt]++;
                } else {
                    sharpPx++;
                }
            }
        }
        sharpFrac = sharpPx / (double) PLANE;
        // best tile to sharpen: unchanged, not fading in, most pixels still at 1x; the centre counts more
        int refT = -1;
        int nCand = 0;
        if (haveRef && sharpen && wStats != null && !hardFull) {
            double best = 0;
            for (int t = 0; t < RT_X * RT_Y; t++) {
                if ((fading & (1 << t)) != 0 || rDirty[t] > RT_W * RT_H / 50 || rUnsharp[t] < RT_W * RT_H / 8) {
                    continue;
                }
                nCand++;
                double dx = ((t % RT_X) + 0.5) / RT_X - 0.5, dy = ((t / RT_X) + 0.5) / RT_Y - 0.5;
                double score = rUnsharp[t] * (1.5 - Math.sqrt(dx * dx + dy * dy));
                if (score > best) {
                    best = score;
                    refT = t;
                }
            }
        }
        jobCandidates = nCand;
        if (!full) {
            bx0 = W;
            by0 = H;
            bx1 = 0;
            by1 = 0;
            int need = (int) (TILE * TILE * DIRTY_FRAC);
            for (int ty = 0; ty < TILES_Y; ty++) {
                for (int tx = 0; tx < TILES_X; tx++) {
                    if (cnt[ty * TILES_X + tx] > need) {
                        bx0 = Math.min(bx0, tx * TILE);
                        by0 = Math.min(by0, ty * TILE);
                        bx1 = Math.max(bx1, (tx + 1) * TILE);
                        by1 = Math.max(by1, (ty + 1) * TILE);
                    }
                }
            }
            if (bx1 <= bx0) {                         // nothing changed: the history is still right ...
                skipEma = 0.9 * skipEma + 0.1;
                wLastWasPartial = false;
                return refT >= 0 ? runRefine(refT) : null;   // ... so spend the time on sharpening
            }
            boolean small = (bx1 - bx0) * (by1 - by0) < SHARPEN_MAX_CHANGE * PLANE;
            if (small && refT >= 0 && wLastWasPartial) {      // little changes: alternate repaint / sharpen
                wLastWasPartial = false;
                return runRefine(refT);
            }
            wLastWasPartial = true;
            if ((bx1 - bx0) * (by1 - by0) > FULL_AREA * PLANE) {
                full = true;
                bx0 = 0;
                by0 = 0;
                bx1 = W;
                by1 = H;
            }
        }
        skipEma = 0.9 * skipEma;

        long n0;
        long n1;
        synchronized (sessionLock) {
            if (session == null) {
                return null;   // model is being switched
            }
            if (full) {
                n0 = System.nanoTime();
                float[] out = run(wIn.rewind(), H, W, 0f);
                n1 = System.nanoTime();
                for (int i = 0; i < 3 * PLANE; i++) {
                    float v = out[i];
                    wOut[i] = v >= 0f ? (v > 1f ? 1f : v) : 0f;   // NaN -> 0
                }
                wLastFullNs = now;
                wSinceFull = 0;
                wLastStyle = style.clone();
                forceFull = false;
            } else {
                // network input: the part + MARGIN of context, sides on multiples of 8 (the U-Net halves 3 times)
                // sizes rounded up to multiples of 64 x 48: few different shapes, so cuDNN picks its algorithms
                // once per shape (the first run of a new size is slow - on this background thread that's harmless)
                int cx0 = Math.max(0, (bx0 - MARGIN) / 8 * 8), cy0 = Math.max(0, (by0 - MARGIN) / 8 * 8);
                int cx1 = Math.min(W, (bx1 + MARGIN + 7) / 8 * 8), cy1 = Math.min(H, (by1 + MARGIN + 7) / 8 * 8);
                int cw = Math.min(W, (cx1 - cx0 + 63) / 64 * 64), ch = Math.min(H, (cy1 - cy0 + 47) / 48 * 48);
                cx0 = Math.min(cx0, W - cw);
                cy0 = Math.min(cy0, H - ch);
                cx1 = cx0 + cw;
                cy1 = cy0 + ch;
                if (wCrop == null) {
                    wCrop = ByteBuffer.allocateDirect(3 * PLANE * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
                }
                wCrop.clear();
                for (int c = 0; c < 3; c++) {
                    for (int y = cy0; y < cy1; y++) {
                        for (int x = cx0; x < cx1; x++) {
                            wCrop.put(wIn.get(c * PLANE + y * W + x));
                        }
                    }
                }
                wCrop.flip();
                n0 = System.nanoTime();
                float[] out = run(wCrop.slice(), ch, cw, 1f);
                n1 = System.nanoTime();
                for (int c = 0; c < 3; c++) {
                    for (int y = by0; y < by1; y++) {
                        for (int x = bx0; x < bx1; x++) {
                            float v = out[c * ch * cw + (y - cy0) * cw + (x - cx0)];
                            wOut[c * PLANE + y * W + x] = v >= 0f ? (v > 1f ? 1f : v) : 0f;
                        }
                    }
                }
                wSinceFull++;
            }
        }
        areaEma = 0.9 * areaEma + 0.1 * ((bx1 - bx0) * (double) (by1 - by0) / PLANE);
        int[] box = full ? null : new int[]{bx0, by0, bx1, by1};   // post-processing only where repainted
        applyDevignette(wOut, tWet, box);
        applyLightLock(wIn, wOut, box);
        if (keepDark) {
            keepDarkGuard(wIn, wOut, box);
        }
        if (dehaze >= 1) {
            levels(wOut, full, box);
        }
        for (int y = by0; y < by1; y++) {
            int row = (H - 1 - y) * W * 4;
            for (int x = bx0; x < bx1; x++) {
                int o = y * W + x;
                int i = row + x * 4;
                wRgba.put(i, (byte) toByte(wOut[o]));
                wRgba.put(i + 1, (byte) toByte(wOut[PLANE + o]));
                wRgba.put(i + 2, (byte) toByte(wOut[2 * PLANE + o]));
                wRgba.put(i + 3, (byte) 255);
            }
        }
        if (full) {
            buildLut();
        } else {   // top-down pixels -> GL 0..1 (y up)
            jobRegion = new float[]{bx0 / (float) W, (H - by1) / (float) H, bx1 / (float) W, (H - by0) / (float) H};
        }
        tNetNs = n1 - n0;
        tPostNs = System.nanoTime() - n1;
        return wRgba.rewind();
    }

    /**
     * Worker: one tile at 2.5x resolution (the network gets the 1280x720 picture there, normalised with the full
     * frame's statistics so it fits the rest). Post-processing like the normal path, then the tile (+ fade border)
     * as RGBA rows bottom-up.
     */
    private ByteBuffer runRefine(int t) throws Exception {
        int ox = Math.max(0, Math.min(REF_W - RC_W, 320 * (t % RT_X) - 88));   // 320 = 128 x 2.5
        int oy = Math.max(0, Math.min(REF_H - RC_H, 240 * (t / RT_X) - 88));   // 240 = 96 x 2.5
        int n = RC_W * RC_H;
        if (wRefIn == null) {
            wRefIn = ByteBuffer.allocateDirect(3 * n * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
        }
        for (int y = 0; y < RC_H; y++) {
            int src = ((REF_H - 1 - (oy + y)) * REF_W + ox) * 4;   // the 1280x720 frame, GL rows bottom-up
            for (int x = 0; x < RC_W; x++) {
                int i = src + x * 4;
                int o = y * RC_W + x;
                wRefIn.put(o, (refBytes[i] & 0xFF) / 255f);
                wRefIn.put(n + o, (refBytes[i + 1] & 0xFF) / 255f);
                wRefIn.put(2 * n + o, (refBytes[i + 2] & 0xFF) / 255f);
            }
        }
        long n0;
        long n1;
        float[] out;
        synchronized (sessionLock) {
            if (session == null) {
                return null;
            }
            n0 = System.nanoTime();
            out = run(wRefIn.rewind(), RC_H, RC_W, 1f);
            n1 = System.nanoTime();
        }
        int tx = t % RT_X, ty = t / RT_X;
        int ex0 = Math.max(0, tx * RT_W - R_FEATHER), ey0 = Math.max(0, ty * RT_H - R_FEATHER);
        int ex1 = Math.min(W, (tx + 1) * RT_W + R_FEATHER), ey1 = Math.min(H, (ty + 1) * RT_H + R_FEATHER);
        int w2 = 5 * (ex1 - ex0) / 2, h2 = 5 * (ey1 - ey0) / 2, m = w2 * h2;   // 2.5x (all sides are even)
        int x0 = 5 * ex0 / 2 - ox, y0 = 5 * ey0 / 2 - oy;                     // the tile in the network input
        float[] pr = new float[3 * m];
        float[] vr = new float[3 * m];
        for (int c = 0; c < 3; c++) {
            for (int y = 0; y < h2; y++) {
                for (int x = 0; x < w2; x++) {
                    int si = c * n + (y0 + y) * RC_W + (x0 + x);
                    float v = out[si];
                    pr[c * m + y * w2 + x] = v >= 0f ? (v > 1f ? 1f : v) : 0f;
                    vr[c * m + y * w2 + x] = wRefIn.get(si);
                }
            }
        }
        if (lightLock) {
            localLightLock(pr, vr, w2, h2, 40);       // 40 px here = the 16 px cells at normal resolution
        }
        if (keepDark) {
            float guard = darkGuard(wIn);
            if (guard > 0f) {
                for (int i = 0; i < m; i++) {
                    float lin = 0.299f * vr[i] + 0.587f * vr[m + i] + 0.114f * vr[2 * m + i];
                    float lout = 0.299f * pr[i] + 0.587f * pr[m + i] + 0.114f * pr[2 * m + i];
                    float cap = lin * 1.2f + 0.015f;
                    float allowed = cap + (1f - guard) * Math.max(0f, lout - cap);
                    if (lout > allowed && lout > 1e-5f) {
                        float k = allowed / lout;
                        pr[i] *= k;
                        pr[m + i] *= k;
                        pr[2 * m + i] *= k;
                    }
                }
            }
        }
        if (dehaze >= 1 && blackEma >= 0f) {
            float black = Math.min(blackEma, 0.6f);
            float gain = 1f / (1f - black);
            for (int i = 0; i < 3 * m; i++) {
                float v = (pr[i] - black) * gain;
                pr[i] = v < 0f ? 0f : (v > 1f ? 1f : v);
            }
        }
        for (int y = 0; y < h2; y++) {
            int row = (h2 - 1 - y) * w2;
            for (int x = 0; x < w2; x++) {
                int o = y * w2 + x;
                int i = (row + x) * 4;
                wRefRgba.put(i, (byte) toByte(pr[o]));
                wRefRgba.put(i + 1, (byte) toByte(pr[m + o]));
                wRefRgba.put(i + 2, (byte) toByte(pr[2 * m + o]));
                wRefRgba.put(i + 3, (byte) 255);
            }
        }
        jobRefW2 = w2;
        jobRefH2 = h2;
        jobRefRect = new int[]{ex0, ey0, ex1, ey1};
        jobRefTile = t;
        tNetNs = n1 - n0;
        tPostNs = System.nanoTime() - n1;
        return wRefRgba.rewind();
    }

    /** Light lock on a w x h picture (planes): per cs x cs cell, brightness within lockMin..lockMax x vanilla. */
    private void localLightLock(float[] p, float[] v, int w, int h, int cs) {
        int nx = (w + cs - 1) / cs, ny = (h + cs - 1) / cs, m = w * h;
        float eps = 0.02f;
        float[] g = new float[nx * ny];
        for (int cy = 0; cy < ny; cy++) {
            for (int cx = 0; cx < nx; cx++) {
                double li = 0, lo = 0;
                int cnt = 0;
                for (int y = cy * cs; y < Math.min(h, (cy + 1) * cs); y++) {
                    for (int x = cx * cs; x < Math.min(w, (cx + 1) * cs); x++) {
                        int o = y * w + x;
                        li += 0.299f * v[o] + 0.587f * v[m + o] + 0.114f * v[2 * m + o];
                        lo += 0.299f * p[o] + 0.587f * p[m + o] + 0.114f * p[2 * m + o];
                        cnt++;
                    }
                }
                double r = (lo / cnt + eps) / (li / cnt + eps);
                double rc = Math.max(lockMin, Math.min(lockMax, r));
                g[cy * nx + cx] = (float) (Math.log(rc / r) / Math.log(2.0));
            }
        }
        for (int y = 0; y < h; y++) {
            float fy = (y + 0.5f) / cs - 0.5f;
            int ya = Math.max(0, Math.min(ny - 1, (int) Math.floor(fy)));
            int yb = Math.min(ny - 1, ya + 1);
            float wy = Math.max(0f, Math.min(1f, fy - ya));
            for (int x = 0; x < w; x++) {
                float fx = (x + 0.5f) / cs - 0.5f;
                int xa = Math.max(0, Math.min(nx - 1, (int) Math.floor(fx)));
                int xb = Math.min(nx - 1, xa + 1);
                float wx = Math.max(0f, Math.min(1f, fx - xa));
                float top = g[ya * nx + xa] * (1 - wx) + g[ya * nx + xb] * wx;
                float bot = g[yb * nx + xa] * (1 - wx) + g[yb * nx + xb] * wx;
                float gain = (float) Math.pow(2.0, top * (1 - wy) + bot * wy);
                int o = y * w + x;
                for (int c = 0; c < 3; c++) {
                    float q = p[c * m + o] * gain;
                    p[c * m + o] = q > 1f ? 1f : q;
                }
            }
        }
    }

    /** Keep dark: 1 = dark frame (no brightening allowed), 0 = day. */
    private static float darkGuard(FloatBuffer in) {
        double sum = 0;
        for (int o = 0; o < PLANE; o += 7) {
            sum += 0.299f * in.get(o) + 0.587f * in.get(PLANE + o) + 0.114f * in.get(2 * PLANE + o);
        }
        float mean = (float) (sum / ((PLANE + 6) / 7));
        return Math.max(0f, Math.min(1f, (0.25f - mean) / (0.25f - 0.08f)));
    }

    /**
     * One network run on a [1,3,h,w] frame (worker, under sessionLock). Partial models also get / give the GroupNorm
     * statistics: use = 0 measures them (stored for later partial runs), use = 1 normalises with the stored ones.
     */
    private float[] run(FloatBuffer frame, int h, int w, float use) throws Exception {
        Map<String, OnnxTensor> feeds = new HashMap<>();
        java.util.List<OnnxTensor> made = new java.util.ArrayList<>();
        try {
            OnnxTensor f = OnnxTensor.createTensor(env, frame, new long[]{1, 3, h, w});
            made.add(f);
            feeds.put(frameName, f);
            OnnxTensor st = OnnxTensor.createTensor(env, wStyle.rewind(), new long[]{1, wStyle.capacity()});
            made.add(st);
            feeds.put(styleName, st);
            if (partialModel) {
                if (wStatsBuf == null) {
                    wStatsBuf = ByteBuffer.allocateDirect(statsRows * 8 * 4).order(ByteOrder.nativeOrder())
                            .asFloatBuffer();
                }
                wStatsBuf.clear();
                if (wStats != null) {
                    wStatsBuf.put(wStats);
                } else {
                    wStatsBuf.put(new float[statsRows * 8]);
                }
                wStatsBuf.flip();
                OnnxTensor ns = OnnxTensor.createTensor(env, wStatsBuf, new long[]{statsRows, 8});
                made.add(ns);
                feeds.put("norm_stats", ns);
                FloatBuffer u = ByteBuffer.allocateDirect(4).order(ByteOrder.nativeOrder()).asFloatBuffer();
                u.put(0, use);
                OnnxTensor ut = OnnxTensor.createTensor(env, u, new long[]{1});
                made.add(ut);
                feeds.put("use_stats", ut);
            }
            try (OrtSession.Result result = session.run(feeds)) {
                if (profile && ++profiledRuns == 300) {
                    LOG.info("Restyler: ONNX layer profile written to {}", session.endProfiling());
                    profile = false;
                }
                FloatBuffer ob = ((OnnxTensor) result.get(0)).getFloatBuffer();
                float[] out = new float[3 * h * w];
                ob.get(out);
                if (partialModel && use == 0f) {
                    java.util.Optional<ai.onnxruntime.OnnxValue> so = result.get("norm_stats_out");
                    if (so.isPresent()) {
                        FloatBuffer sb = ((OnnxTensor) so.get()).getFloatBuffer();
                        float[] stats = new float[statsRows * 8];
                        sb.get(stats);
                        wStats = stats;
                    }
                }
                return out;
            }
        } finally {
            for (OnnxTensor t : made) {
                t.close();
            }
        }
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
            if (partialModel) {   // full frame, measure its own statistics
                feeds.put("norm_stats", OnnxTensor.createTensor(env, new float[statsRows][8]));
                feeds.put("use_stats", OnnxTensor.createTensor(env, new float[]{0f}));
            }
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
            } finally {
                if (partialModel) {
                    feeds.get("norm_stats").close();
                    feeds.get("use_stats").close();
                }
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
        keepDarkGuard(inputBuf, rgbOut, null);
    }

    private static void keepDarkGuard(FloatBuffer inputBuf, float[] rgbOut, int[] box) {
        double sum = 0;
        for (int o = 0; o < PLANE; o += 7) {
            sum += 0.299f * inputBuf.get(o) + 0.587f * inputBuf.get(PLANE + o) + 0.114f * inputBuf.get(2 * PLANE + o);
        }
        float mean = (float) (sum / ((PLANE + 6) / 7));
        float guard = Math.max(0f, Math.min(1f, (0.25f - mean) / (0.25f - 0.08f)));   // 1 = dark, 0 = day
        if (guard <= 0f) {
            return;
        }
        int xa = box == null ? 0 : box[0], xb = box == null ? W : box[2];
        int ya = box == null ? 0 : box[1], yb = box == null ? H : box[3];
        for (int yy = ya; yy < yb; yy++) for (int xx = xa; xx < xb; xx++) {
            int o = yy * W + xx;
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
        levels(rgbOut, true);
    }

    /** measure = false: keep the current black point (partial repaints must match the rest of the picture). */
    private void levels(float[] rgbOut, boolean measure) {
        levels(rgbOut, measure, null);
    }

    private void levels(float[] rgbOut, boolean measure, int[] box) {
        if (!measure && blackEma >= 0f) {
            float black = Math.min(blackEma, 0.6f);
            float gain = 1f / (1f - black);
            int xa = box == null ? 0 : box[0], xb = box == null ? W : box[2];
            int ya = box == null ? 0 : box[1], yb = box == null ? H : box[3];
            for (int c = 0; c < 3; c++) {
                for (int y = ya; y < yb; y++) {
                    for (int x = xa; x < xb; x++) {
                        int i = c * PLANE + y * W + x;
                        float v = (rgbOut[i] - black) * gain;
                        rgbOut[i] = v < 0f ? 0f : (v > 1f ? 1f : v);
                    }
                }
            }
            return;
        }
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
        applyDevignette(out, t, null);
    }

    private void applyDevignette(float[] out, float t, int[] box) {
        float[] vn = vigNormal, vw = vigWet;
        if (!devignette || vn == null) {
            return;
        }
        float[] cell = new float[vn.length];
        for (int i = 0; i < cell.length; i++) {
            cell[i] = -((1f - t) * vn[i] + t * vw[i]);
        }
        applyCellLog2(out, cell, box);
    }

    /**
     * Light lock: per 16x16 px patch, the network's brightness may be at most lockMin..lockMax x the game's.
     * Materials, texture and colour stay; WHERE it is light stays the game's (torches, sky light), so the model
     * can't light the cave like a flashlight held by the player.
     */
    private void applyLightLock(FloatBuffer in, float[] out) {
        applyLightLock(in, out, null);
    }

    /** box = {x0, y0, x1, y1} (top-down px) of a partial repaint: cells outside copy the nearest inside cell. */
    private void applyLightLock(FloatBuffer in, float[] out, int[] box) {
        if (!lightLock) {
            return;
        }
        final int cw = W / VIG_GX, ch = H / VIG_GY;   // 16 x 16
        final float eps = 0.02f;
        float[] cell = new float[VIG_GX * VIG_GY];
        int gxa = box == null ? 0 : box[0] / cw, gxb = box == null ? VIG_GX : (box[2] + cw - 1) / cw;
        int gya = box == null ? 0 : box[1] / ch, gyb = box == null ? VIG_GY : (box[3] + ch - 1) / ch;
        for (int cy = gya; cy < gyb; cy++) {
            for (int cx = gxa; cx < gxb; cx++) {
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
        if (box != null) {   // no seam at the edge of the repainted part: outside cells = nearest inside cell
            int gx0 = box[0] / cw, gy0 = box[1] / ch, gx1 = (box[2] - 1) / cw, gy1 = (box[3] - 1) / ch;
            for (int cy = 0; cy < VIG_GY; cy++) {
                for (int cx = 0; cx < VIG_GX; cx++) {
                    int sx = Math.max(gx0, Math.min(gx1, cx)), sy = Math.max(gy0, Math.min(gy1, cy));
                    cell[cy * VIG_GX + cx] = cell[sy * VIG_GX + sx];
                }
            }
        }
        applyCellLog2(out, cell, box);
    }

    /** out *= 2^cell(x, y), cell bilinear from the 32x18 grid; clamps to 1. */
    private static void applyCellLog2(float[] out, float[] cell) {
        applyCellLog2(out, cell, null);
    }

    /** box = {x0, y0, x1, y1} (top-down px): only there (partial repaint), null = everywhere. */
    private static void applyCellLog2(float[] out, float[] cell, int[] box) {
        int xa = box == null ? 0 : box[0], xb = box == null ? W : box[2];
        int ya = box == null ? 0 : box[1], yb = box == null ? H : box[3];
        float[] rowGain = new float[W];
        for (int y = ya; y < yb; y++) {
            float fy = (y + 0.5f) * VIG_GY / H - 0.5f;
            int y0 = Math.max(0, Math.min(VIG_GY - 1, (int) Math.floor(fy)));
            int y1 = Math.min(VIG_GY - 1, y0 + 1);
            float wy = Math.max(0f, Math.min(1f, fy - y0));
            for (int x = xa; x < xb; x++) {
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
                for (int x = xa; x < xb; x++) {
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

        // first start: fetch the model + style vectors (~11 MB), then continue
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
        if (model != null) {   // same network + partial-update inputs (make_partial.py): use it when it's there
            Path partial = model.resolveSibling(model.getFileName().toString().replace(".onnx", "_partial.onnx"));
            if (Files.exists(partial)) {
                model = partial;
            }
        }
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
            partialModel = false;
            for (String n : session.getInputNames()) {
                if (n.equals("style")) {
                    styleName = n;
                } else if (n.equals("norm_stats")) {
                    partialModel = true;
                    long[] shape = ((ai.onnxruntime.TensorInfo) session.getInputInfo().get(n).getInfo()).getShape();
                    statsRows = (int) shape[0];
                } else if (!n.equals("use_stats") && (frameName == null || n.equals("frame"))) {
                    frameName = n;
                }
            }
            wStats = null;
            forceFull = true;
            if (styleName == null || frameName == null) {
                say("Restyler: " + model.getFileName() + " is not a style model (inputs " + session.getInputNames()
                        + "). Use restyler_v3_epXX.onnx.", false);
                closeSession();
                return false;
            }
            loadedModelName = model.getFileName().toString().replace("_partial.onnx", ".onnx");
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
        say("Restyler: first start - downloading the model (~11 MB)...", true);
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
