# WAFTBase + WAFTOnnx — Design Spec

**Date:** 2026-08-06
**Status:** Approved

## Overview

Backend-agnostic pre/post-processing class `WAFTBase` paired with an ONNX Runtime
concrete class `WAFTOnnx` that inherits both `WAFTBase` and the existing `ONNXModel`.
The split keeps image processing independent of the inference backend so a future
`WAFTTrt(TRTModel, WAFTBase)` is straightforward.

## Module Layout

```
WAFT/
├── model/
│   ├── base/
│   │   ├── base_onnx.py       ← existing (unchanged)
│   │   └── base_trt.py        ← existing (unchanged)
│   ├── waft_base.py           ← NEW: WAFTBase mixin
│   └── waft_onnx.py           ← NEW: WAFTOnnx(WAFTBase, ONNXModel)
└── weights/
    └── waftv2_dav2_i5_448x672.onnx
```

## Inheritance & MRO

```
WAFTOnnx(WAFTBase, ONNXModel)
    │
    ├── WAFTBase   — preprocess() / postprocess() / __call__()
    └── ONNXModel  — session, target_h, target_w, run()
```

`ONNXModel.__init__` sets `target_h`/`target_w` (from the ONNX graph metadata);
`WAFTBase.__init__` only stores the `bgr_input` flag.  Both `__init__`s are called
explicitly by `WAFTOnnx.__init__`.

## model/waft_base.py

### `WAFTBase`

Backend-agnostic mixin.  Designed to pair with any class that exposes `target_h`,
`target_w`, and `run(feed: dict) -> dict`.

**Constructor**

```python
def __init__(self, bgr_input: bool = True):
    self._bgr_input = bgr_input
```

### `preprocess(self, img1, img2) -> dict`

Each argument may be a file path (`str`) or a numpy array (`uint8`, H×W×3).
Steps, per image:

1. **Load** — if `str`, `cv2.imread(path)` → BGR uint8.
2. **BGR→RGB** — if `self._bgr_input`, `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`.
3. **Letterbox** (follows `TRTModel.resize_img`):
   - `scale = min(target_h / h, target_w / w)`, floored to 2 decimals.
   - Resize via `cv2.resize` to `(new_w, new_h)`.
   - Center-pad with pure black `(0,0,0)` via `cv2.copyMakeBorder`.
   - Record `orig_h, orig_w, scale_factor, tile_h, tile_w, pad_top, pad_left`
     into `self._meta`.
4. **To tensor** — HWC→CHW, add batch dim, cast to `float32` (values stay [0, 255]).

Returns `{"image1": array(1,3,H,W), "image2": array(1,3,H,W)}`, ready for `self.run()`.

### `postprocess(self, raw_output: dict) -> np.ndarray`

1. Extract `flow = raw_output["flow"]` → `[1, 2, H, W]`.
2. Crop padding: `flow[:, :, pad_top:pad_top+tile_h, pad_left:pad_left+tile_w]`.
3. Remove batch dim → `[2, tile_h, tile_w]`.
4. Resize to `(orig_h, orig_w)` via `cv2.resize(..., INTER_LINEAR)`.
5. Scale flow x-component by `orig_w / tile_w`, y-component by `orig_h / tile_h`.
6. Return `[orig_h, orig_w, 2]` float32.

### `__call__(self, img1, img2) -> np.ndarray`

```python
feed = self.preprocess(img1, img2)
raw = self.run(feed)
return self.postprocess(raw)
```

## model/waft_onnx.py

### `WAFTOnnx(WAFTBase, ONNXModel)`

```python
class WAFTOnnx(WAFTBase, ONNXModel):
    def __init__(self, onnx_path: str, device: str = "cuda",
                 bgr_input: bool = True):
        ONNXModel.__init__(self, onnx_path, device)
        WAFTBase.__init__(self, bgr_input=bgr_input)
```

No additional logic — `WAFTBase.__call__` drives the pipeline; `ONNXModel.run`
performs inference and handles dtype casting (uint8 → float32 per the ONNX input
metadata).

## Data Flow

```
img1 (BGR uint8, 480×640)
img2 (BGR uint8, 480×640)
        │
  preprocess()
        │  BGR→RGB, letterbox → (448,672), CHW, batch, float32
        ▼
  ONNXModel.run()
        │  feed {"image1": (1,3,448,672), "image2": (1,3,448,672)}
        │  output {"flow": (1,2,448,672)}
        ▼
  postprocess()
        │  crop pad, resize → (480,640), scale flow values
        ▼
  flow (480, 640, 2) float32
```

## Edge Cases

| Case | Behavior |
|---|---|
| Image path doesn't exist | `cv2.imread` returns `None` → raise `FileNotFoundError` with path |
| Image already larger than target | Scale < 1.0, resize down, no padding needed on that axis |
| Image exactly matches target | Scale = 1.0, no resize or pad needed |
| ONNX model has different input names | `preprocess` hardcodes `image1`/`image2` (matches export) |
| Single image passed as both args | Works — identical preprocess applied independently |

## Testing

Sample frames `data/astribot_stereo_lrb/extract_frames/stereo_left/frame_000150.jpg`
and `frame_000154.jpg` (480×640 BGR, stride=4) serve as a smoke test:

```python
model = WAFTOnnx("weights/waftv2_dav2_i5_448x672.onnx")
flow = model(".../frame_000150.jpg", ".../frame_000154.jpg")
assert flow.shape == (480, 640, 2)
```
