---
tags:
  - minecraft
  - onnx
  - image-to-image
  - real-time
library_name: onnx
---

# Restyler V3 (mod 3.0.0)

Real-time AI restyler for Minecraft (Fabric 1.21.1). It repaints every frame with realistic materials on the player's own GPU.

These are the files the **Restyler** mod downloads by itself on first start. Players don't need to fetch them by hand.

| File | What | SHA-256 |
|---|---|---|
| `restyler_v3_ep60.onnx` | student network: 1.4M-parameter FiLM U-Net, fp32 | `996f916a007c7be6c974234810e23e87cdbae2917ef80f03504cd3f6129ce692` |
| `restyler_v3_ep60_partial.onnx` | the same network with partial-update inputs (dynamic size, `norm_stats` in/out, `use_stats`): repaints only what changed and sharpens at 2.5x | `63b1bb6ca37bfc238d246a027f93f44f263433f94db36e24934285b0fdc625ef` |
| `styles.json` | 48 style vectors (labels, mu, vectors) for the network's style input | `2552531dba5216fdaf356987d4491709bd0690da6017c8223b4c2bed7bf6fe57` |

## How it was made
1. Vanilla, HUD-free Minecraft frames were collected.
2. FLUX.2 klein repainted them (the "teacher").
3. A small U-Net (the "student") was distilled from those paintings so it runs live: about 30–45 ms/frame on a GTX 1650 (CUDA, ONNX Runtime). Epoch 60 is the release checkpoint.

## In game
- Keys: `O` opens the menu, `U` changes the look (Auto/weather, Normal, WetCraft), `K` turns it on or off, `[` `]` change the strength.
- **Performance mode** (default ON): 60 FPS. The network runs in the background and its look is moved with the camera using the depth buffer.
- **Partial update**: only what changed on screen is repainted.
- **Sharpen**: spare time repaints the screen at 2.5x resolution; when done, the whole screen focuses at once over 1 s.
- Light lock (default ON): the game decides where it is light; the model only changes materials.

NVIDIA GPU recommended. The mod downloads the CUDA runtime (~1.6 GB, once) if it's missing. Other GPUs fall back to the CPU (slow).
