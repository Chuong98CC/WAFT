# ONNX Export for WAFTv2 (DAv2 Backbone) — Design Spec

**Date:** 2026-08-06
**Status:** Approved

## Overview

Export the WAFTv2 optical flow model (waft-a2, DepthAnythingV2 backbone) to ONNX format
for deployment with ONNX Runtime and TensorRT. The exported model takes a pair of RGB
images and produces a single optical flow field.

## CLI Interface

```
python export_onnx.py \
  --cfg config/a2/dav2/chairs-things.json \
  --ckpt ckpts/dav2/zero-shot.pth \
  --height 448 --width 672 \
  --iters 5 \
  --output waftv2_dav2.onnx \
  --opset 16
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--cfg` | str | required | JSON model config |
| `--ckpt` | str | required | Trained checkpoint `.pth` |
| `--height` | int | `448` | Input image height (must be multiple of 112) |
| `--width` | int | `672` | Input image width (must be multiple of 112) |
| `--iters` | int | `5` | Iterative refinement steps to unroll into the graph |
| `--output` | str | `waftv2_dav2.onnx` | Output .onnx file path |
| `--opset` | int | `16` | ONNX opset version |

## Module Layout

```
WAFT/
├── export_onnx.py           ← NEW: CLI + export orchestration
└── utils/
    └── onnx_utils.py        ← NEW: FloatFunctional patching + ONNXExportWrapper
```

## utils/onnx_utils.py

### `Add` module

A trivial replacement for `nn.quantized.FloatFunctional`:

```python
class Add(nn.Module):
    def forward(self, a, b):
        return a + b
```

### `patch_float_functional(module)`

Recursively walks the module tree. For every `nn.quantized.FloatFunctional` attribute,
replaces it with an `Add()` instance. Call sites that do `self.skip_add.add(out, x)`
become `self.skip_add(out, x)` — functionally identical for FP32 inference.

Files touched by this patch:
- `model/backbone/head.py` (ResidualConvUnit, FeatureFusionBlock)
- `model/backbone/twins.py` (not used for dav2, but patched if present)
- `thirdparty/DepthAnythingV2/depth_anything_v2/util/blocks.py` (DAv2 internal blocks)

### `ONNXExportWrapper(nn.Module)`

Wraps the patched model to produce a clean, ONNX-friendly forward graph:

```python
class ONNXExportWrapper(nn.Module):
    def __init__(self, model, iters=5):
        super().__init__()
        self.model = model
        self.iters = iters

    def forward(self, image1, image2):
        # image1, image2: [1, 3, H, W] in range [0, 255], RGB
        output = self.model(image1, image2, iters=self.iters)
        return output['flow'][-1]  # final flow: [1, 2, H, W]
```

Key design decisions:
- Only the wrapper is exported; the original model stays unmodified on disk.
- `iters` is baked in — the for-loop unrolls during tracing, producing a single
  static graph.
- Only the final flow tensor is returned — no dict, no info predictions, no
  intermediate iterations.
- The `flow_gt` conditional path in `WAFTv2.forward` is never taken during
  inference, so the NF loss computation is not traced.

## export_onnx.py

### Export flow

1. **Parse config and create model** via `parse_args()` + `fetch_model(args)`
2. **Load checkpoint** via `load_ckpt()`, set `model.eval()`
3. **Patch** all `FloatFunctional` → `Add` recursively
4. **Wrap** in `ONNXExportWrapper(model, iters)`
5. **Create dummy inputs** `(torch.randn(1, 3, H, W), torch.randn(1, 3, H, W))`
6. **Validate resolution** — both H and W must be divisible by the DAv2 pad factor (112).
   Error out with a clear message if not.
7. **Export** with `torch.onnx.export()`:
   - `input_names=['image1', 'image2']`
   - `output_names=['flow']`
   - `opset_version` from `--opset`
   - No dynamic axes (static shape export)
8. **Verify**:
   - `onnx.checker.check_model()` for structural validity
   - ONNX Runtime inference on the same random input, compared against PyTorch output
   - Assert max absolute delta < 1e-4
9. **Print summary**: model path, input shape, iteration count, file size

### Input/output spec

| | Shape | Range | Description |
|---|---|---|---|
| `image1` | `[1, 3, H, W]` | `[0, 255]` | First frame, RGB |
| `image2` | `[1, 3, H, W]` | `[0, 255]` | Second frame, RGB |
| `flow` | `[1, 2, H, W]` | float | Optical flow (dx, dy) |

Image normalization (`/255` then `Normalize(mean, std)`) is traced as part of the
graph inside `WAFTv2.normalize_image()`, so the ONNX model expects raw 0-255 RGB input —
matching the PyTorch model's interface exactly.

### Edge cases

| Case | Behavior |
|---|---|
| Resolution not divisible by 112 | Print error with the required constraint, exit(1) |
| Checkpoint file not found | Error at load_ckpt stage |
| ONNX ops not supported in given opset | `torch.onnx.export` will raise with details |
| GPU vs CPU | Export always on CPU (avoids CUDA-in-ONNX complications) |
| Verification mismatch | Print the max delta and a warning; exported file is still written |

### Dependencies

- `onnx` (for `onnx.checker.check_model`)
- `onnxruntime` (for verification inference)
- Existing: `torch`, `timm`, `cv2` (only in DAv2 — see note below)

Note: `model/backbone/head.py` imports `cv2` at module level but never uses it.
This import is harmless for ONNX export (it's in the patched model, not in the
exported graph) but worth noting. We will not fix it as part of this work since
the design scope is ONNX export only.
