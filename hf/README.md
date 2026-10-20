---
tags:
  - minecraft
  - onnx
  - image-to-image
  - real-time
library_name: onnx
---

# Restyler V3

Real-time AI restyler for Minecraft (Fabric 1.21.1). It repaints every frame with realistic materials on the player's own GPU.

These are the files the **Restyler** mod downloads by itself on first start. Players don't need to fetch them by hand.

| File | What | SHA-256 |
|---|---|---|
| `restyler_v3_ep60.onnx` | student network: 1.4M-parameter FiLM U-Net, fp32 | `996f916a007c7be6c974234810e23e87cdbae2917ef80f03504cd3f6129ce692` |
| `styles.json` | 48 style vectors (labels, mu, vectors) for the network's style input | `2552531dba5216fdaf356987d4491709bd0690da6017c8223b4c2bed7bf6fe57` |

## How it was made
1. Vanilla, HUD-free Minecraft frames were collected.
2. FLUX.2 klein repainted them (the "teacher").
3. A small U-Net (the "student") was distilled from those paintings so it runs live: about 30–45 ms/frame on a GTX 1650 (CUDA, ONNX Runtime). Epoch 60 is the release checkpoint.

## In game
- Keys: `O` opens the menu, `U` changes the look (Auto/weather, Normal, WetCraft), `K` turns it on or off, `[` `]` change the strength.
- **Performance mode** (menu): 60 FPS, but much lower quality (rain reflections and fine detail suffer). Default is OFF.
- Light lock (default ON): the game decides where it is light; the model only changes materials.

NVIDIA GPU recommended. The mod downloads the CUDA runtime (~1.6 GB, once) if it's missing. Other GPUs fall back to the CPU (slow).
