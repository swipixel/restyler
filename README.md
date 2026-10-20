# Restyler 0.3 — release build (self-installing)

Players only drop `restyler-0.3.0.jar` into `mods` (Fabric 1.21.1 + Fabric API). On first start the mod downloads:
1. the model `restyler_v3_ep60.onnx` + `styles.json` (a few MB) into `config/restyler/`
2. NVIDIA's GPU runtime (~1.6 GB, once) if CUDA isn't installed. Other GPUs fall back to the CPU (slow).

## Publish (once)
Model host: https://huggingface.co/SwiPixel/RestylerV3 (`BASE_URL` and SHA-256 are already set in `ModelFetcher.java`).
1. Double-click `publish.bat`. It uploads `hf\` (model, styles.json, model card) with your logged-in `hf` CLI, then makes this folder a git repo with one commit.
2. `gradlew build` (Minecraft closed) → `build\libs\restyler-0.3.0.jar`.
3. Test like a new player: move `config\restyler` away, start the game, watch the action bar ("downloading ...").
4. GitHub: `gh repo create SwiPixel/restyler --public --source . --push`.

## Defaults
Model ep60 · Performance mode OFF (full quality) · Light lock ON · Look: Auto (weather).
Keys: O menu, U look, K on/off, [ ] strength. Performance mode (menu) = 60 FPS, lower quality.

Only one Restyler jar in `mods` at a time (this one replaces the dev build; both are mod id `restyler`).
