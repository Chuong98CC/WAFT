# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WAFT (Warping-Alone Field Transforms) is an optical flow estimation method that replaces RAFT-style cost volumes with high-resolution warping. It ranks 1st on Spring, Sintel, and KITTI benchmarks. Two algorithm variants exist: **waft-a1** and **waft-a2**, with waft-a2 being the newer, more flexible version supporting multiple frozen feature encoders (Twins, DepthAnythingV2, DINOv3).

## Environment

- Python 3.12, PyTorch 2.7.0, CUDA 12.8
- Requires `xformers` (installed separately per official instructions)
- Conda environment: `conda create --name waft python=3.12 && conda activate waft && pip install -r requirements.txt`

## Commands

### Training (multi-GPU DDP)
```bash
# Stage 1: FlyingChairs only
python train.py --cfg config/a2/twins/chairs.json
# Stage 2: Fine-tune on Chairs+Things
python train.py --cfg config/a2/twins/chairs-things.json --restore_ckpt checkpoints/chairs/waft-a2/twins/42/50000.pth
```
Config files are organized by algorithm version and backbone: `config/{a1,a2}/{twins,dav2,dinov3}/`. All hyperparameters (lr, batch_size, iters, image_size, etc.) live in the JSON config, not on the command line. Override the seed with `--seed N`.

### Evaluation
```bash
python evaluate.py --cfg config/a2/twins/chairs-things.json --ckpt ckpts/twins/zero-shot.pth --dataset sintel
```
Datasets: `sintel`, `kitti`, `spring`. Optionally pass `--scale` for multi-scale inference (e.g., `--scale -1` for half-resolution input, upsampled output).

### Submission (generate leaderboard outputs)
```bash
python submission.py --cfg config/a2/twins/tar-c-t-kitti.json --ckpt ckpts/twins/kitti.pth --dataset kitti
```

### Demo (visualize on sample images)
```bash
python demo.py --cfg config/a2/twins/chairs.json --ckpt ckpts/twins/chairs.pth --dataset sintel
```

## Architecture

### Model variants

Both models follow the same iterative-refinement pattern and share these components:
- **Frozen feature encoder** extracts dense features from image pairs (not trained)
- **fnet** (ResNet18 variant) extracts image-level features that are concatenated with encoder features
- **Iterative transformer** (VisionTransformer, patch_size=8) refines a hidden state `net` at 1/2 resolution
- **Flow head** predicts (Δflow, weight, log-bias) at each iteration
- **Convex upsampling** learns 3×3 mask weights to upsample 2× flow to full resolution

Key differences:
| | waft-a1 (`model/waft_a1.py:ViTWarpV8`) | waft-a2 (`model/waft_a2.py:WAFTv2`) |
|---|---|---|
| Encoder | DepthAnythingV2 only (frozen) | Twins / DAv2 / DINOv3 (frozen, configurable) |
| fnet | timm ResNet18 layers as ConvBlocks | Custom `ResNet18Deconv` with deconv upsampling |
| Pad factor | Fixed 112 | Backbone-dependent (32/112/16) |
| forward output | Always computes NF loss terms | Only computes NF loss when `flow_gt` is provided |

### Key files

| File | Purpose |
|---|---|
| `train.py` | Training loop with DDP, OneCycleLR, wandb logging, periodic checkpointing |
| `evaluate.py` | Validation on Sintel/KITTI/Spring with EPE, F1, px1 metrics |
| `submission.py` | Generate benchmark submission files (.flo, .flo5, .png) |
| `demo.py` | Visualize predictions, errors, and uncertainty heatmaps |
| `model/__init__.py` | Factory: `fetch_model(args)` dispatches to waft-a1 or waft-a2 |
| `model/backbone/head.py` | `DPTHead` — DPT-style fusion blocks shared by all backbones |
| `model/backbone/vit.py` | `VisionTransformer` — iterative transformer with PatchEmbed + DPTHead |
| `dataloader/template.py` | `FlowDataset` base class with augmentation, retry-on-error fetch |
| `dataloader/loader.py` | `fetch_dataloader()` — dataset assembly with dataset-specific augment params |
| `dataloader/augmentor.py` | `FlowAugmentor` — spatial/color augmentation |
| `criterion/loss.py` | `sequence_loss` — exponentially weighted sum of negative free energy terms |
| `config/parser.py` | JSON config → argparse.Namespace, with CLI overrides |
| `inference_tools.py` | `InferenceWrapper` — padding, tiling, optional multi-scale inference |

### Data flow (training)

1. `train.py` parses JSON config, creates model via `fetch_model()`, wraps in DDP
2. `fetch_dataloader()` assembles training datasets (with dataset-specific augment params and repeat factors for smaller datasets)
3. Each iteration: `image1, image2, flow_gt, valid` → model → output dict with `flow` (list of per-iteration predictions), `info` (uncertainty params), `nf` (negative free energy per iteration)
4. `sequence_loss()` computes exponentially-weighted NF loss (higher weight on later iterations)

### Inference flow

`InferenceWrapper.calc_flow()` handles:
1. Optional input scaling (`2**scale` factor)
2. Optional padding to `train_size` (multiples of 64)
3. Optional tiling with Gaussian-weighted blending
4. Final interpolation back to original resolution

### Loss formulation

The loss uses a negative free energy (NF) approach — each pixel predicts parameters of a Laplace distribution (μ=flow, b=log-bias) plus a mixture weight. The NF loss combines data fidelity and uncertainty in a principled way. `var_min` and `var_max` clamp the log-bias terms.

## Checkpoints

Checkpoints are saved to `checkpoints/{name}/{algorithm}/{feature_encoder}/{seed}/` every 10k steps and a final one. Model weights are on Google Drive (link in README). For downstream use, the recommended checkpoint is the a1 adaptation model.

## Key dependencies

- `timm` — pretrained ViT/Twins backbones and ResNet layers
- `wandb` — experiment tracking (project name from config `name` field)
- `xformers` — memory-efficient attention (must install separately)
- Third-party: `thirdparty/DepthAnythingV2/` contains DAv2 model code; `model/backbone/dinov3.py` loads DINOv3 via `torch.hub` from `thirdparty/dinov3/`
