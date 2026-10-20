package dev.swipixel.restylermixin;

import dev.swipixel.restyler.Restyler;
import net.minecraft.client.gui.DrawContext;
import net.minecraft.client.gui.hud.InGameHud;
import net.minecraft.client.render.RenderTickCounter;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/** At the start of the HUD the screen holds the finished world without the HUD: restyle it there. */
@Mixin(InGameHud.class)
public class InGameHudMixin {

    @Inject(method = "render", at = @At("HEAD"))
    private void restyler$beforeHud(DrawContext context, RenderTickCounter tickCounter, CallbackInfo ci) {
        Restyler.INSTANCE.onBeforeHud(context);
    }
}
