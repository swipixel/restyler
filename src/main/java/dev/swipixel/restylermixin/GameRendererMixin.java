package dev.swipixel.restylermixin;

import dev.swipixel.restyler.Restyler;
import net.minecraft.client.render.GameRenderer;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * End of renderWorld: the world is drawn, the depth buffer was cleared and holds only the hand / held item.
 * The GUI clears it again before the HUD, so the hand mask has to be taken here.
 */
@Mixin(GameRenderer.class)
public class GameRendererMixin {

    @Inject(method = "renderWorld", at = @At("RETURN"))
    private void restyler$afterHand(CallbackInfo ci) {
        Restyler.INSTANCE.afterHand();
    }
}
