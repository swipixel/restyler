package dev.swipixel.restyler;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.client.MinecraftClient;
import net.minecraft.text.ClickEvent;
import net.minecraft.text.Text;
import net.minecraft.util.Formatting;

import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;

/**
 * Once per game start: asks Modrinth for the newest Restyler version for this Minecraft version + Fabric and,
 * if it is newer, shows a chat line with a clickable [Download] when you join a world.
 * (Players who installed it through the Modrinth App / a pack get the update button there as well.)
 * Set PROJECT to your Modrinth project slug after publishing; empty = no check.
 */
final class UpdateCheck {

    static final String PROJECT = "";      // e.g. "restyler" once https://modrinth.com/mod/restyler exists
    private static volatile Text message;
    private static boolean shown = false;

    private UpdateCheck() {
    }

    static void start() {
        if (PROJECT.isEmpty()) {
            return;
        }
        Thread t = new Thread(UpdateCheck::run, "restyler-update-check");
        t.setDaemon(true);
        t.start();
    }

    private static void run() {
        try {
            String current = FabricLoader.getInstance().getModContainer("restyler")
                    .map(c -> c.getMetadata().getVersion().getFriendlyString()).orElse("0");
            String mc = FabricLoader.getInstance().getModContainer("minecraft")
                    .map(c -> c.getMetadata().getVersion().getFriendlyString()).orElse("1.21.1");
            String q = "?loaders=" + enc("[\"fabric\"]") + "&game_versions=" + enc("[\"" + mc + "\"]");
            HttpResponse<String> r = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(8)).build().send(
                    HttpRequest.newBuilder(URI.create("https://api.modrinth.com/v2/project/" + PROJECT + "/version" + q))
                            .header("User-Agent", "SwiPixel/restyler/" + current).timeout(Duration.ofSeconds(10)).build(),
                    HttpResponse.BodyHandlers.ofString());
            if (r.statusCode() != 200) {
                return;
            }
            JsonArray versions = JsonParser.parseString(r.body()).getAsJsonArray();
            if (versions.isEmpty()) {
                return;
            }
            JsonObject newest = versions.get(0).getAsJsonObject();   // Modrinth lists newest first
            String latest = newest.get("version_number").getAsString();
            if (!newer(latest, current)) {
                return;
            }
            String url = "https://modrinth.com/mod/" + PROJECT + "/version/" + newest.get("id").getAsString();
            message = Text.literal("Restyler " + latest + " is out (you have " + current + ")  ")
                    .formatted(Formatting.YELLOW)
                    .append(Text.literal("[Download]").styled(s -> s.withColor(Formatting.AQUA).withUnderline(true)
                            .withClickEvent(new ClickEvent(ClickEvent.Action.OPEN_URL, url))));
            Restyler.LOG.info("Restyler: update available {} -> {}", current, latest);
        } catch (Exception e) {
            Restyler.LOG.info("Restyler: update check skipped ({})", e.toString());
        }
    }

    /** Called when joining a world. */
    static void showIfAny(MinecraftClient mc) {
        if (message != null && !shown && mc.player != null) {
            shown = true;
            mc.player.sendMessage(message, false);
        }
    }

    private static String enc(String s) {
        return URLEncoder.encode(s, StandardCharsets.UTF_8);
    }

    static boolean newer(String a, String b) {
        String[] x = a.replaceAll("[^0-9.]", "").split("\\.");
        String[] y = b.replaceAll("[^0-9.]", "").split("\\.");
        for (int i = 0; i < Math.max(x.length, y.length); i++) {
            int p = i < x.length && !x[i].isEmpty() ? Integer.parseInt(x[i]) : 0;
            int q = i < y.length && !y[i].isEmpty() ? Integer.parseInt(y[i]) : 0;
            if (p != q) {
                return p > q;
            }
        }
        return false;
    }
}
