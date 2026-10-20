package dev.swipixel.restyler;

import net.minecraft.client.gui.DrawContext;
import net.minecraft.client.gui.screen.Screen;
import net.minecraft.client.gui.tooltip.Tooltip;
import net.minecraft.client.gui.widget.ButtonWidget;
import net.minecraft.client.gui.widget.SliderWidget;
import net.minecraft.text.Text;

import java.util.List;
import java.util.function.DoubleConsumer;

/** Menu (key O). The game keeps running behind it, so every change is visible immediately. */
public class RestylerScreen extends Screen {

    private static final int BW = 200;
    private static final int BH = 20;
    private static final int GAP = 4;

    private final Restyler r = Restyler.INSTANCE;

    public RestylerScreen() {
        super(Text.literal("Restyler"));
    }

    @Override
    protected void init() {
        int x = (width - BW) / 2;
        int y = 34;

        addDrawableChild(ButtonWidget.builder(lookText(), b -> {
            r.setLook(r.getLook().next());
            b.setMessage(lookText());
        }).dimensions(x, y, BW, BH).tooltip(Tooltip.of(Text.literal(
                "Auto (weather): normal when dry, WetCraft while it rains (only in biomes where it rains). "
                        + "Dries over about a minute after the rain stops.\n"
                        + "Normal: realistic materials.\nWetCraft: always wet."))).build());
        y += BH + GAP;

        addDrawableChild(new Slider(x, y, BW, BH, "Strength", r.getStrength(), v -> r.setStrength((float) v)));
        y += BH + GAP;

        addDrawableChild(ButtonWidget.builder(onOff("Restyler", r.isEnabled()), b -> {
            r.setEnabled(!r.isEnabled());
            b.setMessage(onOff("Restyler", r.isEnabled()));
        }).dimensions(x, y, BW, BH).build());
        y += BH + GAP + 6;

        int hw = (BW - GAP) / 2;
        addDrawableChild(ButtonWidget.builder(onOff("Keep dark", r.isKeepDark()), b -> {
            r.setKeepDark(!r.isKeepDark());
            b.setMessage(onOff("Keep dark", r.isKeepDark()));
        }).dimensions(x, y, hw, BH).tooltip(Tooltip.of(Text.literal(
                "ON: at night and in caves the filter can't brighten pitch black (no free night vision)"))).build());
        addDrawableChild(ButtonWidget.builder(dehazeText(), b -> {
            r.setDehaze(1 - r.getDehaze());
            b.setMessage(dehazeText());
        }).dimensions(x + hw + GAP, y, hw, BH).tooltip(Tooltip.of(Text.literal(
                "levels: stretch the darkest 1% back to black (more contrast)"))).build());
        y += BH + GAP;

        addDrawableChild(ButtonWidget.builder(onOff("Performance", r.isStabilizer()), b -> {
            r.setStabilizer(!r.isStabilizer());
            b.setMessage(onOff("Performance", r.isStabilizer()));
        }).dimensions(x, y, hw, BH).tooltip(Tooltip.of(Text.literal(
                "Performance mode: 60 FPS - the network runs in the background and its look is moved along with "
                        + "the camera.\n!! Greatly reduces quality: rain reflections, gloss and fine details suffer.\n"
                        + "OFF (recommended): full quality, the network runs on every frame (lower FPS)."))).build());
        addDrawableChild(ButtonWidget.builder(Text.literal("Model: " + shortModel(r.loadedModel())), b -> {
            List<String> files = r.modelFiles();
            if (!files.isEmpty()) {
                int cur = files.indexOf(r.loadedModel());
                r.useModel(files.get((cur + 1) % files.size()));
                b.setMessage(Text.literal("Model: " + shortModel(files.get((cur + 1) % files.size()))));
            }
        }).dimensions(x + hw + GAP, y, hw, BH).tooltip(Tooltip.of(Text.literal(
                "Cycle through the .onnx files in config/restyler"))).build());
        y += BH + GAP;

        if (r.isAsync()) {
            addDrawableChild(new Slider(x, y, hw, BH, "Detail", r.getDetail() / 1.5,
                    v -> r.setDetail((float) (v * 1.5))));
            addDrawableChild(ButtonWidget.builder(onOff("Texel snap", r.isTexelSnap()), b -> {
                r.setTexelSnap(!r.isTexelSnap());
                b.setMessage(onOff("Texel snap", r.isTexelSnap()));
            }).dimensions(x + hw + GAP, y, hw, BH).tooltip(Tooltip.of(Text.literal(
                    "ON: the look is read once per block texel - locked to the block grid, pixel-art crisp.\n"
                            + "OFF: smooth, as before."))).build());
            y += BH + GAP;
        }
        addDrawableChild(ButtonWidget.builder(onOff("Light lock", r.isLightLock()), b -> {
            r.setLightLock(!r.isLightLock());
            b.setMessage(onOff("Light lock", r.isLightLock()));
        }).dimensions(x, y, hw, BH).tooltip(Tooltip.of(Text.literal(
                "ON: the game decides where it is light (torches, sky); the model only changes materials.\n"
                        + "Stops far walls going dark and lighting up as you walk closer."))).build());
        addDrawableChild(ButtonWidget.builder(vigText(), b -> {
            r.setDevignette(!r.isDevignette());
            b.setMessage(vigText());
        }).dimensions(x + hw + GAP, y, hw, BH).tooltip(Tooltip.of(Text.literal(
                "Removes the model's screen-fixed darkening (the 'shadow' that follows the middle of the screen).\n"
                        + "Needs vignette.json in config/restyler (made by vignette_v3.py)."))).build());
        y += BH + GAP + 6;

        addDrawableChild(ButtonWidget.builder(Text.literal("Done"), b -> close()).dimensions(x, y, BW, BH).build());
    }

    private Text lookText() {
        return Text.literal("Look: " + r.getLook().title);
    }

    private Text vigText() {
        return Text.literal("Vignette: " + (!r.hasVignette() ? "no file" : r.isDevignette() ? "ON" : "OFF"));
    }

    private static Text onOff(String name, boolean on) {
        return Text.literal(name + ": " + (on ? "ON" : "OFF"));
    }

    private Text dehazeText() {
        return Text.literal("Dehaze: " + (r.getDehaze() == 0 ? "off" : "levels"));
    }

    private Text speedText() {
        int n = r.getRunEvery();
        return Text.literal(n == 1 ? "Every frame" : "Every " + n + (n == 2 ? "nd" : "rd") + " frame");
    }

    private static String shortModel(String m) {
        String s = m.replace(".onnx", "").replace("restyler_", "");
        return s.length() > 12 ? s.substring(0, 12) + ".." : s;
    }

    @Override
    public void renderBackground(DrawContext context, int mouseX, int mouseY, float delta) {
        // no blur: the restyled world stays visible behind the menu
        int x = (width - BW) / 2;
        context.fill(x - 8, 8, x + BW + 8, 34 + 8 * (BH + GAP) + 20, 0x88000000);
    }

    @Override
    public void render(DrawContext context, int mouseX, int mouseY, float delta) {
        super.render(context, mouseX, mouseY, delta);
        context.drawCenteredTextWithShadow(textRenderer, Text.literal("Restyler"), width / 2, 12, 0xFFFFFFFF);
        context.drawCenteredTextWithShadow(textRenderer, Text.literal(r.statusLine()), width / 2, 22, 0xFFAAAAAA);
    }

    @Override
    public boolean shouldPause() {
        return false;
    }

    private static final class Slider extends SliderWidget {
        private final String label;
        private final DoubleConsumer onChange;

        Slider(int x, int y, int w, int h, String label, double value, DoubleConsumer onChange) {
            super(x, y, w, h, Text.empty(), value);
            this.label = label;
            this.onChange = onChange;
            updateMessage();
        }

        @Override
        protected void updateMessage() {
            setMessage(Text.literal(label + ": " + Math.round(value * 100) + "%"));
        }

        @Override
        protected void applyValue() {
            onChange.accept(value);
        }
    }
}
