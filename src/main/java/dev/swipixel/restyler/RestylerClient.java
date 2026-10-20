package dev.swipixel.restyler;

import com.mojang.brigadier.arguments.StringArgumentType;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.command.v2.ClientCommandManager;
import net.fabricmc.fabric.api.client.command.v2.ClientCommandRegistrationCallback;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;
import net.fabricmc.fabric.api.client.networking.v1.ClientPlayConnectionEvents;
import net.fabricmc.fabric.api.client.rendering.v1.WorldRenderEvents;
import net.minecraft.util.math.Vec3d;
import net.minecraft.client.option.KeyBinding;
import net.minecraft.client.util.InputUtil;
import org.lwjgl.glfw.GLFW;

/**
 * Keys (changeable in Options > Controls): K on/off, U look (Auto / Normal / WetCraft), [ ] strength, O menu.
 * /restyler                      status
 * /restyler look auto|normal|wet
 * /restyler model [file.onnx]    switch network (no file = newest restyler_v3_epXX.onnx)
 */
public class RestylerClient implements ClientModInitializer {

    @Override
    public void onInitializeClient() {
        Restyler r = Restyler.INSTANCE;
        r.init();

        UpdateCheck.start();
        ClientPlayConnectionEvents.JOIN.register((handler, sender, client) ->
                client.execute(() -> UpdateCheck.showIfAny(client)));

        // end of world rendering (before the hand): camera + depth for the stabilizer
        WorldRenderEvents.END.register(ctx -> {
            Vec3d p = ctx.camera().getPos();
            r.onWorldEnd(ctx.projectionMatrix(), ctx.positionMatrix(), p.x, p.y, p.z);
        });

        KeyBinding toggle = key("toggle", GLFW.GLFW_KEY_K);
        KeyBinding lookKey = key("look", GLFW.GLFW_KEY_U);
        KeyBinding less = key("less", GLFW.GLFW_KEY_LEFT_BRACKET);
        KeyBinding more = key("more", GLFW.GLFW_KEY_RIGHT_BRACKET);
        KeyBinding menu = key("menu", GLFW.GLFW_KEY_O);

        ClientCommandRegistrationCallback.EVENT.register((dispatcher, registryAccess) -> dispatcher.register(
                ClientCommandManager.literal("restyler")
                        .executes(ctx -> {
                            Restyler.say("Restyler " + (r.isEnabled() ? "ON" : "OFF") + "  |  " + r.loadedModel()
                                    + "  |  " + r.statusLine(), false);
                            return 1;
                        })
                        .then(ClientCommandManager.literal("look")
                                .then(ClientCommandManager.literal("auto").executes(ctx -> look(r, Restyler.Look.AUTO)))
                                .then(ClientCommandManager.literal("normal").executes(ctx -> look(r, Restyler.Look.NORMAL)))
                                .then(ClientCommandManager.literal("wet").executes(ctx -> look(r, Restyler.Look.WET))))
                        .then(ClientCommandManager.literal("model")
                                .executes(ctx -> {
                                    r.useModel("");
                                    return 1;
                                })
                                .then(ClientCommandManager.argument("file", StringArgumentType.greedyString())
                                        .executes(ctx -> {
                                            r.useModel(StringArgumentType.getString(ctx, "file"));
                                            return 1;
                                        })))));

        ClientTickEvents.END_CLIENT_TICK.register(client -> {
            r.tick(client);
            while (menu.wasPressed()) {
                client.setScreen(new RestylerScreen());
            }
            while (toggle.wasPressed()) {
                r.toggle();
            }
            while (lookKey.wasPressed()) {
                r.cycleLook();
            }
            while (less.wasPressed()) {
                r.changeStrength(-0.1f);
            }
            while (more.wasPressed()) {
                r.changeStrength(0.1f);
            }
        });
    }

    private static KeyBinding key(String name, int code) {
        return KeyBindingHelper.registerKeyBinding(
                new KeyBinding("key.restyler." + name, InputUtil.Type.KEYSYM, code, "category.restyler"));
    }

    private static int look(Restyler r, Restyler.Look l) {
        r.setLook(l);
        Restyler.say("Look: " + r.lookText(), true);
        return 1;
    }
}
