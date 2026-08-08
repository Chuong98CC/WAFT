# Video Inference Tool — Design Spec

**Date:** 2026-08-06
**Status:** Approved

## Overview

A CLI tool `infer_video.py` that runs WAFT optical flow inference on a user-provided
video file, producing flow visualizations, raw flow data, and/or overlay videos.
Video I/O helpers are extracted into `utils/video_utils.py` for reuse.

## CLI Interface

```
python infer_video.py \
  --input path/to/video.mp4 \
  --cfg config/a2/twins/chairs-things.json \
  --ckpt checkpoints/waft-a2/model.pth \
  --output-dir ./output \
  --output-mode all \
  --start 0 \
  --stride 1 \
  --max-frames 500 \
  --scale 0.0
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--input` | str | required | Path to input video file |
| `--cfg` | str | required | JSON model config (same format as train/evaluate) |
| `--ckpt` | str | required | Model checkpoint `.pth` file |
| `--output-dir` | str | `./output` | Root directory for outputs |
| `--output-mode` | str choices | `all` | `flow` (visualizations), `raw` (.flo files), `overlay` (flow blended onto frames), `all` (all three) |
| `--start` | int | `0` | Skip first N frames before processing |
| `--stride` | int | `1` | Gap between each frame pair (1 = consecutive) |
| `--max-frames` | int | `None` | Max number of frame PAIRS to process; if None, process until EOF |
| `--scale` | float | `0.0` | Multi-scale inference factor (passed to InferenceWrapper) |

### Module Layout

```
WAFT/
├── utils/
│   └── video_utils.py   ← NEW: flow video encoding + video info
└── infer_video.py        ← NEW: CLI + frame iteration + orchestration
```

## utils/video_utils.py

Three functions, no model dependency:

- **`get_video_info(video_path)`** → `(total_frames, fps, width, height)`
  Probes the video via `cv2.VideoCapture`. Returns 0 for total_frames if the codec
  cannot report it (guarded — callers must handle 0 gracefully).

- **`encode_flow_video(flow_frames, output_path, fps)`**
  Takes a list of flow visualization images (BGR numpy arrays, as produced by
  `flow_viz.flow_to_image`), encodes to `.mp4` using `cv2.VideoWriter` with mp4v codec.
  Prints the output path on success.

- **`encode_overlay_video(original_frames, flow_frames, output_path, fps, alpha=0.5)`**
  Same as above but each frame is `cv2.addWeighted(original_frame, 1-alpha, flow_vis, alpha, 0)`.
  Both input lists must be the same length.

## infer_video.py

### Frame iteration (forward-only, no seeking)

```python
cap = cv2.VideoCapture(video_path)
# Warm-up: discard first 'start' frames
for _ in range(args.start):
    cap.read()

# Read first anchor frame
ret, frame_a = cap.read()
assert ret, "Not enough frames after start offset"

pbar = tqdm(total=max_frames)

for pair_idx in range(max_frames):
    # Skip gap frames (stride - 1)
    for _ in range(args.stride - 1):
        cap.read()

    ret, frame_b = cap.read()
    if not ret:
        break  # EOF

    # Preprocess: BGR→RGB, ndarray→tensor, [H,W,C]→[1,C,H,W]
    img1 = torch.from_numpy(frame_a[:,:,::-1].copy()).permute(2,0,1).float()[None].cuda()
    img2 = torch.from_numpy(frame_b[:,:,::-1].copy()).permute(2,0,1).float()[None].cuda()

    # Run inference
    output = model.calc_flow(img1, img2)
    flow = output['flow'][-1][0].permute(1,2,0).cpu().numpy()  # [H,W,2]

    # Write per-frame outputs
    write_frame_outputs(pair_idx, frame_a, flow, output_mode, output_dirs)

    # Collect for video encoding
    if output_mode in ('flow', 'all'):
        flow_vis_frames.append(flow_to_image(flow, convert_to_bgr=True))
    if output_mode in ('overlay', 'all'):
        overlay_original_frames.append(frame_a)
        overlay_flow_frames.append(flow_to_image(flow, convert_to_bgr=True))

    frame_a = frame_b  # slide forward
    pbar.update(1)

pbar.close()
cap.release()

# Encode videos
encode_flow_video(flow_vis_frames, ...)
encode_overlay_video(overlay_original_frames, overlay_flow_frames, ...)
```

For `start=10, stride=3, max_frames=100` the pairs are:
(10,13), (13,16), (16,19), ..., (10+99×3, 10+100×3).

### Output structure

```
output_dir/
└── <video_filename_without_ext>/
    ├── flow/      # frame_0000.png, frame_0001.png, ...
    │   └── flow.mp4
    ├── raw/       # frame_0000.flo, frame_0001.flo, ...
    └── overlay/   # frame_0000.png, frame_0001.png, ...
        └── overlay.mp4
```

- PNGs and .flo files are written incrementally during the loop.
- .mp4 videos are encoded once at the end (collecting only visualized frames in memory).
- Original frames for overlay video are the anchor frame of each pair (`frame_a`).

### Model loading

Uses the existing codepath from evaluate.py/submission.py:
```python
from config.parser import parse_args
from model import fetch_model
from utils.utils import load_ckpt
from inference_tools import InferenceWrapper

args = parse_args(parser)  # CLI overrides JSON config entries
model = fetch_model(args).cuda().eval()
load_ckpt(model, args.ckpt)
wrapped_model = InferenceWrapper(
    model, scale=args.scale, train_size=args.image_size,
    pad_to_train_size=False, tiling=False
)
```

Flow output is taken from `output['flow'][-1]` (final iteration).

### Edge cases

| Case | Behavior |
|---|---|
| Video has fewer frames than `start` | Print error, exit(1) |
| Video exhausted before `max_frames` | Process what's available, log actual count |
| Frame dimensions not divisible by model pad factor | `InferenceWrapper` handles padding, so no special case |
| `cv2.VideoCapture.get(CAP_PROP_FRAME_COUNT)` returns 0 or negative | Handle gracefully: tqdm total only set when known, progress bar shows only count |

### Progress display

- `tqdm` with `total=max_frames` (or total estimated from `(frame_count - start) // stride` if max_frames not set).
- After completion, print summary: "Processed N pairs in X seconds. Output saved to: ..."
