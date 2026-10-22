# Restyler V3 (3.0.0) — release build (self-installing)

Real-time AI restyle for Minecraft (Fabric 1.21.1 + Fabric API). Players only drop `restyler-3.0.0.jar` into `mods`.
On first start the mod downloads:
1. `restyler_v3_ep60.onnx`, `restyler_v3_ep60_partial.onnx` and `styles.json` (~11 MB) into `config/restyler/`
2. NVIDIA's GPU runtime (~1.6 GB, once) if CUDA isn't installed. Other GPUs fall back to the CPU (slow).

## What's in V3
- **Network:** ep60 student (1.4M-parameter FiLM U-Net, distilled from FLUX.2 klein paintings), 512x288.
- **Performance mode** (default ON): the network runs in the background, its look is moved with the camera every
  frame using the depth buffer (60 FPS).
- **Partial update:** only what changed is repainted (new strips when turning, a moving mob, a broken block).
- **Sharpen:** spare time repaints the screen at 2.5x resolution in the background; when it's done the whole screen
  focuses at once over 1 s (no tile-by-tile building). Sharp parts move with the camera and survive repaints; they
  follow light changes (torches, time of day). Off in WetCraft.
- **Debug view** (menu or `/restyler debug`): green sharp, blue waiting for focus, yellow 1x, red not cached.
- Light lock, Keep dark, Dehaze, Vignette removal, Look: Auto (weather) / Normal / WetCraft.

## Defaults
Performance ON · Partial ON · Sharpen ON · Detail 0 % · Light lock ON · Look: Auto (weather).
Keys: O menu, U look, K on/off, [ ] strength.

## Publish
Model host: https://huggingface.co/SwiPixel/RestylerV3 (`BASE_URL` and SHA-256s are set in `ModelFetcher.java`).
1. Double-click `publish.bat`: uploads `hf\` (both models, styles.json, model card) with your logged-in `hf` CLI,
   then commits this folder to git.
2. `gradlew build` (Minecraft closed) → `build\libs\restyler-3.0.0.jar`. `copyit.bat` puts it into your instance.
3. Test like a new player: move `config\restyler` away, start the game, watch the action bar ("downloading ...").
4. GitHub: `gh repo create SwiPixel/restyler --public --source . --push`.

Only one Restyler jar in `mods` at a time (this one replaces the dev build; both are mod id `restyler`).
Existing configs keep their own settings; the new defaults apply to new players (or delete `config/restyler/restyler.properties`).
