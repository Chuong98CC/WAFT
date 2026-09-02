#!/usr/bin/env python
"""Run WAFT optical flow inference on a video file using an ONNX or TensorRT model.

Usage (ONNX):
    python infer_video_onnx.py \\
        --input path/to/video.mp4 \\
        --onnx weights/waftv2_dav2_i5_448x672.onnx \\
        --output-dir ./output \\
        --output-mode all \\
        --start 0 \\
        --stride 4 \\
        --max-frames 150

Usage (TensorRT):
    python infer_video_onnx.py \\
        --input path/to/video.mp4 \\
        --trt weights/waftv2/waftv2_dinov3_i5_640x480.engine \\
        --output-dir ./output \\
        --output-mode all \\
        --start 0 \\
        --stride 4 \\
        --max-frames 150
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np
from tqdm import tqdm

from model.waft_onnx import WAFTOnnx
from utils.flow_viz import flow_to_image
from utils.frame_utils import writeFlow
from utils.video_utils import encode_flow_video, get_video_info


# ---------------------------------------------------------------------------
# Output helpers (unchanged from infer_video.py)
# ---------------------------------------------------------------------------

def setup_output_dirs(output_root: str, video_name: str, output_mode: str) -> dict:
    """Create output subdirectories and return their paths."""
    base = os.path.join(output_root, video_name)
    dirs: dict[str, str] = {}
    if output_mode in ("flow", "all"):
        dirs["flow"] = os.path.join(base, "flow")
    if output_mode in ("raw", "all"):
        dirs["raw"] = os.path.join(base, "raw")
    if output_mode in ("overlay", "all"):
        dirs["overlay"] = os.path.join(base, "overlay")
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def write_frame_outputs(
    pair_idx: int,
    frame_a: np.ndarray,
    flow: np.ndarray,
    output_mode: str,
    output_dirs: dict,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Write per-frame outputs and return visualizations for video encoding.

    Only ``raw`` mode writes per-frame ``.flo`` files.
    """
    flow_vis = flow_to_image(flow, convert_to_bgr=True)

    if output_mode in ("raw", "all") and "raw" in output_dirs:
        name = f"frame_{pair_idx:06d}"
        writeFlow(os.path.join(output_dirs["raw"], f"{name}.flo"), flow)

    overlay_frame = None
    if output_mode in ("overlay", "all"):
        overlay_frame = cv2.addWeighted(frame_a, 0.5, flow_vis, 0.5, 0)

    return flow_vis, overlay_frame


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def infer_video(args: argparse.Namespace) -> None:
    """Main inference loop over video frame pairs."""

    # ── Probe video ──────────────────────────────────────────────────────
    video_name = os.path.splitext(os.path.basename(args.input))[0]
    total_frames, fps, width, height = get_video_info(args.input)
    print(f"Video: {args.input}")
    print(f"  Resolution: {width}x{height}, FPS: {fps:.2f}")
    if total_frames > 0:
        print(f"  Total frames: {total_frames}")
    else:
        print(f"  Total frames: unknown (codec limitation)")

    # ── Load model (ONNX or TensorRT) ────────────────────────────────────
    if args.trt:
        from model.waft_trt import WAFTTrl

        print(f"\nLoading TensorRT engine from: {args.trt}")
        model = WAFTTrl(args.trt, bgr_input=not args.no_bgr_input)
        backend = "TRT"
    else:
        print(f"\nLoading ONNX model from: {args.onnx}")
        model = WAFTOnnx(args.onnx, device=args.device, bgr_input=not args.no_bgr_input)
        backend = "ONNX"

    print(f"  Backend:      {backend}")
    print(f"  Target resolution: {model.target_h}x{model.target_w}")

    # ── Open video & iterate ─────────────────────────────────────────────
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video file: {args.input}")
        sys.exit(1)

    # Warm-up: discard first 'start' frames
    for skip_idx in range(args.start):
        ret, _ = cap.read()
        if not ret:
            print(
                f"ERROR: Video ended before skipping {args.start} frames "
                f"(only {skip_idx} readable)."
            )
            cap.release()
            sys.exit(1)

    # Read first anchor frame
    ret, frame_a = cap.read()
    if not ret:
        print(
            f"ERROR: Cannot read frame {args.start} from the video. "
            "The codec may be unsupported by your OpenCV/ffmpeg build. "
            "Try re-encoding to H.264:\n"
            "  ffmpeg -i input.mp4 -c:v libx264 -preset fast -crf 23 output.mp4"
        )
        cap.release()
        sys.exit(1)

    # Determine iteration limit
    max_pairs = (
        args.max_frames
        if args.max_frames is not None
        else (
            max(0, (total_frames - args.start)) // args.stride
            if total_frames > 0
            else None
        )
    )

    # Set up progress bar
    output_dirs = setup_output_dirs(args.output_dir, video_name, args.output_mode)
    pbar = tqdm(total=max_pairs, desc="Processing", unit="pair")

    # State for video encoding (collected lazily)
    flow_vis_frames: list[np.ndarray] = []
    overlay_frames: list[np.ndarray] = []

    pair_idx = 0
    start_time = time.time()

    while max_pairs is None or pair_idx < max_pairs:
        # Skip gap frames (stride - 1)
        for _ in range(args.stride - 1):
            cap.read()

        ret, frame_b = cap.read()
        if not ret:
            break  # EOF

        # Run inference — model.__call__ handles pre/post processing
        flow = model(frame_a, frame_b)  # → [H_orig, W_orig, 2] float32

        # Write per-frame outputs and collect for video encoding
        flow_vis, overlay_frame = write_frame_outputs(
            pair_idx, frame_a, flow, args.output_mode, output_dirs
        )

        if args.output_mode in ("flow", "all"):
            flow_vis_frames.append(flow_vis)
        if args.output_mode in ("overlay", "all") and overlay_frame is not None:
            overlay_frames.append(overlay_frame)

        frame_a = frame_b  # slide forward
        pair_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    if pair_idx == 0:
        print("No frame pairs were processed. Check --start and --stride settings.")
        sys.exit(0)

    elapsed = time.time() - start_time
    print(
        f"\nProcessed {pair_idx} pairs in {elapsed:.1f}s "
        f"({pair_idx / elapsed:.1f} pairs/s)"
    )

    # ── Encode videos ────────────────────────────────────────────────────
    output_fps = fps / args.stride if fps > 0 else 30.0

    if flow_vis_frames:
        encode_flow_video(
            flow_vis_frames,
            os.path.join(output_dirs.get("flow", ""), "flow.mp4"),
            output_fps,
        )

    if overlay_frames:
        encode_flow_video(
            overlay_frames,
            os.path.join(output_dirs.get("overlay", ""), "overlay.mp4"),
            output_fps,
        )

    print(f"\nOutput saved to: {os.path.join(args.output_dir, video_name)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run WAFT optical flow inference on a video file (ONNX or TensorRT backend).",
    )
    parser.add_argument(
        "--input", required=True, type=str, help="Path to input video file."
    )

    # Mutually exclusive backend selection
    backend = parser.add_mutually_exclusive_group(required=True)
    backend.add_argument(
        "--onnx", type=str, default=None, help="Path to ONNX model (.onnx)."
    )
    backend.add_argument(
        "--trt", type=str, default=None, help="Path to TensorRT engine (.engine / .trt / .plan)."
    )

    parser.add_argument(
        "--output-dir",
        default="./output",
        type=str,
        help="Root output directory (default: ./output).",
    )
    parser.add_argument(
        "--output-mode",
        default="all",
        type=str,
        choices=["flow", "raw", "overlay", "all"],
        help="What to produce: flow visualizations, raw .flo data, "
        "overlay video, or all three (default: all).",
    )
    parser.add_argument(
        "--start",
        default=0,
        type=int,
        help="Skip first N frames (default: 0).",
    )
    parser.add_argument(
        "--stride",
        default=1,
        type=int,
        help="Gap between paired frames, >=1 (default: 1).",
    )
    parser.add_argument(
        "--max-frames",
        default=None,
        type=int,
        help="Max number of frame PAIRS to process (default: until EOF).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        choices=["cuda", "cpu"],
        help="ONNX Runtime device (default: cuda). Ignored for TensorRT.",
    )
    parser.add_argument(
        "--no-bgr-input",
        action="store_true",
        help="Images are already RGB (skip internal BGR→RGB conversion).",
    )

    args = parser.parse_args()
    infer_video(args)


if __name__ == "__main__":
    main()
