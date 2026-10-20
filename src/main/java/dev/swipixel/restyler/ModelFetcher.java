package dev.swipixel.restyler;

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
import java.util.HexFormat;
import java.util.Locale;
import java.util.function.Consumer;

/**
 * First start: downloads the model + style vectors into config/restyler/ (a few MB), so players only install the jar.
 *
 * Files live in the PUBLIC Hugging Face repo SwiPixel/RestylerV3 (upload: publish.bat). A new model = upload it,
 * change MODEL and paste its sha256 (empty = not checked) below.
 */
final class ModelFetcher {

    static final String BASE_URL = "https://huggingface.co/SwiPixel/RestylerV3/resolve/main/";
    static final String MODEL = "restyler_v3_ep60.onnx";
    static final String[][] FILES = {
            {MODEL, "996f916a007c7be6c974234810e23e87cdbae2917ef80f03504cd3f6129ce692"},   // file name, sha256
            {"styles.json", "2552531dba5216fdaf356987d4491709bd0690da6017c8223b4c2bed7bf6fe57"},
    };

    private ModelFetcher() {
    }

    static boolean configured() {
        return !BASE_URL.isEmpty();
    }

    static boolean present(Path dir) {
        for (String[] f : FILES) {
            try {
                if (!Files.exists(dir.resolve(f[0])) || Files.size(dir.resolve(f[0])) == 0) {
                    return false;
                }
            } catch (Exception e) {
                return false;
            }
        }
        return true;
    }

    /** Blocking (call from a background thread). Skips files that are already there. */
    static void download(Path dir, Consumer<String> say) throws Exception {
        Files.createDirectories(dir);
        HttpClient http = HttpClient.newBuilder().followRedirects(HttpClient.Redirect.NORMAL)
                .connectTimeout(Duration.ofSeconds(20)).build();
        for (String[] f : FILES) {
            Path dst = dir.resolve(f[0]);
            if (Files.exists(dst) && Files.size(dst) > 0) {
                continue;
            }
            String url = BASE_URL + f[0];
            HttpResponse<InputStream> r = http.send(HttpRequest.newBuilder(URI.create(url))
                    .header("User-Agent", "SwiPixel/restyler").build(), HttpResponse.BodyHandlers.ofInputStream());
            if (r.statusCode() != 200) {
                throw new IllegalStateException(url + " -> HTTP " + r.statusCode());
            }
            long size = r.headers().firstValueAsLong("Content-Length").orElse(-1);
            Path tmp = dir.resolve(f[0] + ".part");
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            try (InputStream in = r.body(); OutputStream out = Files.newOutputStream(tmp)) {
                byte[] buf = new byte[1 << 16];
                long done = 0;
                int last = -1;
                for (int n; (n = in.read(buf)) > 0; ) {
                    out.write(buf, 0, n);
                    md.update(buf, 0, n);
                    done += n;
                    int pct = size > 0 ? (int) (done * 100 / size) : -1;
                    if (pct >= 0 && pct / 10 != last / 10) {
                        last = pct;
                        say.accept(String.format(Locale.ROOT, "Restyler: downloading %s %d%%", f[0], pct));
                    }
                }
            }
            String sha = HexFormat.of().formatHex(md.digest());
            if (!f[1].isEmpty() && !f[1].equalsIgnoreCase(sha)) {
                Files.deleteIfExists(tmp);
                throw new IllegalStateException("checksum mismatch for " + f[0]);
            }
            Files.move(tmp, dst, StandardCopyOption.REPLACE_EXISTING);
            Restyler.LOG.info("Restyler: downloaded {} (sha256 {})", f[0], sha);
        }
    }
}
