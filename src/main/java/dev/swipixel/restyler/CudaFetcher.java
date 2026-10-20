package dev.swipixel.restyler;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.io.File;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HexFormat;
import java.util.List;
import java.util.Locale;
import java.util.function.Consumer;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

/**
 * Windows without CUDA 12 / cuDNN 9 installed: downloads NVIDIA's official runtime DLLs once (from the
 * nvidia-* packages on PyPI, ~1.6 GB download, ~2 GB on disk) into config/restyler/cuda/ and loads them
 * before ONNX Runtime needs them - the same trick as onnxruntime.preload_dlls() in Python.
 * Nothing is installed system-wide; delete the folder to remove it.
 */
final class CudaFetcher {

    // package, required version prefix (ONNX Runtime 1.20 = CUDA 12 + cuDNN 9)
    private static final String[][] PACKAGES = {
            {"nvidia-cuda-runtime-cu12", "12."},
            {"nvidia-cublas-cu12", "12."},
            {"nvidia-cufft-cu12", "11."},
            {"nvidia-cuda-nvrtc-cu12", "12."},
            {"nvidia-cudnn-cu12", "9."},
    };
    private static final String[] NEEDED = {"cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll",
            "cufft64_11.dll", "cudnn64_9.dll"};
    private static final String DONE = "download_complete.txt";

    private CudaFetcher() {
    }

    static boolean isWindows() {
        return System.getProperty("os.name", "").toLowerCase(Locale.ROOT).contains("win");
    }

    static boolean present(Path dir) {
        if (!Files.exists(dir.resolve(DONE))) {
            return false;
        }
        for (String n : NEEDED) {
            if (!Files.exists(dir.resolve(n))) {
                return false;
            }
        }
        return true;
    }

    /** Loads every DLL in the folder; retries in rounds so dependency order sorts itself out. */
    static void preload(Path dir) {
        List<File> todo = new ArrayList<>();
        File[] files = dir.toFile().listFiles((d, n) -> n.toLowerCase(Locale.ROOT).endsWith(".dll"));
        if (files == null) {
            return;
        }
        // cudart / cublasLt first, cudnn64_9 (which loads its sub-libraries) last
        for (File f : files) {
            todo.add(f);
        }
        todo.sort((a, b) -> Integer.compare(order(a.getName()), order(b.getName())));
        int loaded = 0;
        boolean progress = true;
        while (!todo.isEmpty() && progress) {
            progress = false;
            for (int i = 0; i < todo.size(); i++) {
                try {
                    System.load(todo.get(i).getAbsolutePath());
                    todo.remove(i--);
                    loaded++;
                    progress = true;
                } catch (Throwable ignored) {
                    // a dependency is not loaded yet - next round
                }
            }
        }
        Restyler.LOG.info("Restyler: preloaded {} CUDA DLLs from {}{}", loaded, dir,
                todo.isEmpty() ? "" : " (could not load: " + todo + ")");
    }

    private static int order(String n) {
        n = n.toLowerCase(Locale.ROOT);
        if (n.startsWith("cudart")) return 0;
        if (n.startsWith("cublaslt")) return 1;
        if (n.startsWith("cublas")) return 2;
        if (n.startsWith("cufft")) return 3;
        if (n.startsWith("nvrtc")) return 4;
        if (n.startsWith("cudnn_")) return 5;
        if (n.startsWith("cudnn")) return 6;
        return 7;
    }

    /** Blocking. Reports progress through say ("CUDA download: cudnn 45% ..."). */
    static void download(Path dir, Consumer<String> say) throws Exception {
        Files.createDirectories(dir);
        HttpClient http = HttpClient.newBuilder().followRedirects(HttpClient.Redirect.NORMAL)
                .connectTimeout(Duration.ofSeconds(20)).build();
        for (String[] pkg : PACKAGES) {
            JsonObject wheel = findWheel(http, pkg[0], pkg[1]);
            String url = wheel.get("url").getAsString();
            long size = wheel.get("size").getAsLong();
            String sha = wheel.getAsJsonObject("digests").get("sha256").getAsString();
            String shortName = pkg[0].replace("nvidia-", "").replace("-cu12", "");
            Path tmp = dir.resolve(pkg[0] + ".whl.part");

            HttpResponse<InputStream> resp = http.send(HttpRequest.newBuilder(URI.create(url)).build(),
                    HttpResponse.BodyHandlers.ofInputStream());
            if (resp.statusCode() != 200) {
                throw new IllegalStateException(url + " -> HTTP " + resp.statusCode());
            }
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            try (InputStream in = resp.body(); OutputStream out = Files.newOutputStream(tmp)) {
                byte[] buf = new byte[1 << 20];
                long done = 0;
                int lastPct = -1;
                for (int r; (r = in.read(buf)) > 0; ) {
                    out.write(buf, 0, r);
                    md.update(buf, 0, r);
                    done += r;
                    int pct = (int) (done * 100 / Math.max(1, size));
                    if (pct / 5 != lastPct / 5) {
                        lastPct = pct;
                        say.accept(String.format(Locale.ROOT, "Restyler: downloading GPU runtime - %s %d%% (%d MB)",
                                shortName, pct, size >> 20));
                    }
                }
            }
            if (!HexFormat.of().formatHex(md.digest()).equalsIgnoreCase(sha)) {
                Files.deleteIfExists(tmp);
                throw new IllegalStateException("checksum mismatch for " + pkg[0]);
            }
            say.accept("Restyler: unpacking " + shortName + " ...");
            try (ZipFile zip = new ZipFile(tmp.toFile())) {
                Enumeration<? extends ZipEntry> en = zip.entries();
                while (en.hasMoreElements()) {
                    ZipEntry e = en.nextElement();
                    String name = e.getName();
                    if (!e.isDirectory() && name.toLowerCase(Locale.ROOT).endsWith(".dll")) {
                        String base = name.substring(name.lastIndexOf('/') + 1);
                        try (InputStream in = zip.getInputStream(e)) {
                            Files.copy(in, dir.resolve(base), StandardCopyOption.REPLACE_EXISTING);
                        }
                    }
                }
            }
            Files.deleteIfExists(tmp);
        }
        Files.writeString(dir.resolve(DONE), "NVIDIA CUDA 12 / cuDNN 9 runtime DLLs from PyPI (nvidia-* packages). "
                + "Delete this folder to remove them.\n");
    }

    /** Newest win_amd64 wheel of the package whose version starts with prefix. */
    private static JsonObject findWheel(HttpClient http, String pkg, String prefix) throws Exception {
        HttpResponse<String> r = http.send(HttpRequest.newBuilder(URI.create("https://pypi.org/pypi/" + pkg + "/json"))
                .timeout(Duration.ofSeconds(30)).build(), HttpResponse.BodyHandlers.ofString());
        if (r.statusCode() != 200) {
            throw new IllegalStateException("PyPI " + pkg + " -> HTTP " + r.statusCode());
        }
        JsonObject releases = JsonParser.parseString(r.body()).getAsJsonObject().getAsJsonObject("releases");
        String bestVer = null;
        JsonObject best = null;
        for (String ver : releases.keySet()) {
            if (!ver.startsWith(prefix) || !ver.matches("[0-9.]+")) {
                continue;
            }
            for (JsonElement f : releases.getAsJsonArray(ver)) {
                JsonObject o = f.getAsJsonObject();
                if (o.get("filename").getAsString().endsWith("win_amd64.whl")
                        && (bestVer == null || newer(ver, bestVer))) {
                    bestVer = ver;
                    best = o;
                }
            }
        }
        if (best == null) {
            throw new IllegalStateException("no Windows wheel for " + pkg + " " + prefix + "x");
        }
        return best;
    }

    private static boolean newer(String a, String b) {
        String[] x = a.split("\\.");
        String[] y = b.split("\\.");
        for (int i = 0; i < Math.max(x.length, y.length); i++) {
            long p = i < x.length ? Long.parseLong(x[i]) : 0;
            long q = i < y.length ? Long.parseLong(y[i]) : 0;
            if (p != q) {
                return p > q;
            }
        }
        return false;
    }
}
