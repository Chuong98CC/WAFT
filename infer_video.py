#!/usr/bin/env python
"""Run WAFT optical flow inference on a video file.

Usage:
    python infer_video.py \
        --input path/to/video.mp4 \
        --cfg config/a2/twins/chairs-things.json \
        --ckpt checkpoints/waft-a2/model.pth \
        --output-dir ./output \
        --output-mode all \
        --start 0 \
        --stride 1 \
        --max-frames 500
"""

import argparse
import os
import sys
import time

import cv2
import torch
from tqdm import tqdm

from config.parser import parse_args
from model import fetch_model
from utils.flow_viz import flow_to_image
from utils.frame_utils import writeFlow
from utils.utils import load_ckpt
from utils.video_utils import encode_flow_video, get_video_info
from inference_tools import InferenceWrapper


def setup_output_dirs(output_root, video_name, output_mode):
    """Create output subdirectories and return their paths.

    Args:
        output_root (str): Root output directory.
        video_name (str): Name of the video file (without extension).
        output_mode (str): One of 'flow', 'raw', 'overlay', 'all'.

    Returns:
        dict: Mapping of mode keys to directory paths.
    """
    base = os.path.join(output_root, video_name)
    dirs = {}
    if output_mode in ("flow", "all"):
        dirs["flow"] = os.path.join(base, "flow")
    if output_mode in ("raw", "all"):
        dirs["raw"] = os.path.join(base, "raw")
    if output_mode in ("overlay", "all"):
        dirs["overlay"] = os.path.join(base, "overlay")

    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    return dirs


def write_frame_outputs(pair_idx, frame_a, flow, output_mode, output_dirs):
    """Write per-frame outputs and return flow visualization for video encoding.

    Only 'raw' mode writes per-frame files (.flo). The 'flow' and 'overlay'
    modes accumulate frames in-memory and encode a single .mp4 at the end.

    Args:
        pair_idx (int): Index of the current frame pair.
        frame_a (np.ndarray): Anchor frame (BGR, uint8).
        flow (np.ndarray): Optical flow array of shape [H, W, 2].
        output_mode (str): One of 'flow', 'raw', 'overlay', 'all'.
        output_dirs (dict): Mode → directory path mapping.

    Returns:
        tuple: (flow_vis, overlay_frame) where:
            - flow_vis: flow visualization image (BGR, uint8)
            - overlay_frame: blended overlay image (BGR, uint8), or None if
              overlay mode is not requested
    """
    flow_vis = flow_to_image(flow, convert_to_bgr=True, rad_max=16)

    if output_mode in ("raw", "all") and "raw" in output_dirs:
        name = f"frame_{pair_idx:06d}"
        writeFlow(os.path.join(output_dirs["raw"], f"{name}.flo"), flow)

    overlay_frame = None
    if output_mode in ("overlay", "all"):
        overlay_frame = cv2.addWeighted(frame_a, 0.5, flow_vis, 0.5, 0)

    return flow_vis, overlay_frame


@torch.no_grad()
def infer_video(args):
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

    # ── Load model ───────────────────────────────────────────────────────
    print(f"\nLoading model from: {args.ckpt}")
    args.gpus = [0]
    model = fetch_model(args).cuda()
    load_ckpt(model, args.ckpt)
    model = model.eval()
    wrapped_model = InferenceWrapper(
        model,
        scale=args.scale if hasattr(args, "scale") else 0.0,
        train_size=args.image_size,
        pad_to_train_size=False,
        tiling=False,
    )
    print("Model loaded.")

    # ── Open video & iterate ─────────────────────────────────────────────
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video file: {args.input}")
        sys.exit(1)

    # Warm-up: discard first 'start' frames
    for skip_idx in range(args.start):
        ret, _ = cap.read()
        if not ret:
            print(f"ERROR: Video ended before skipping {args.start} frames (only {skip_idx} readable).")
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
    max_pairs = args.max_frames if args.max_frames is not None else (
        max(0, (total_frames - args.start)) // args.stride if total_frames > 0 else None
    )

    # Set up progress bar
    output_dirs = setup_output_dirs(args.output_dir, video_name, args.output_mode)
    pbar = tqdm(total=max_pairs, desc="Processing", unit="pair")

    # State for video encoding (collected lazily)
    flow_vis_frames = []
    overlay_frames = []

    pair_idx = 0
    start_time = time.time()

    while max_pairs is None or pair_idx < max_pairs:
        # Skip gap frames (stride - 1)
        for _ in range(args.stride - 1):
            cap.read()

        ret, frame_b = cap.read()
        if not ret:
            break  # EOF

        # BGR → RGB, numpy → tensor
        img1 = torch.from_numpy(frame_a[:, :, ::-1].copy()).permute(2, 0, 1).float()[None].cuda()
        img2 = torch.from_numpy(frame_b[:, :, ::-1].copy()).permute(2, 0, 1).float()[None].cuda()

        # Run inference
        output = wrapped_model.calc_flow(img1, img2)
        flow = output["flow"][-1][0].permute(1, 2, 0).cpu().numpy()  # [H, W, 2]

        # Write per-frame outputs and collect flow/overlay frames for video encoding
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
    print(f"\nProcessed {pair_idx} pairs in {elapsed:.1f}s ({pair_idx / elapsed:.1f} pairs/s)")

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


def main():
    parser = argparse.ArgumentParser(
        description="Run WAFT optical flow inference on a video file.",
    )
    parser.add_argument("--input", required=True, type=str,
                        help="Path to input video file.")
    parser.add_argument("--cfg", required=True, type=str,
                        help="Model JSON config file.")
    parser.add_argument("--ckpt", required=True, type=str,
                        help="Model checkpoint .pth file.")
    parser.add_argument("--output-dir", default="./output", type=str,
                        help="Root output directory (default: ./output).")
    parser.add_argument("--output-mode", default="all", type=str,
                        choices=["flow", "raw", "overlay", "all"],
                        help="What to produce: flow visualizations, raw .flo data, "
                             "overlay video, or all three (default: all).")
    parser.add_argument("--start", default=0, type=int,
                        help="Skip first N frames (default: 0).")
    parser.add_argument("--stride", default=1, type=int,
                        help="Gap between paired frames, >=1 (default: 1). "
                             "stride=1 means consecutive pairs.")
    parser.add_argument("--max-frames", default=None, type=int,
                        help="Max number of frame PAIRS to process "
                             "(default: process until EOF).")
    parser.add_argument("--scale", default=0.0, type=float,
                        help="Multi-scale inference factor (default: 0.0).")

    args = parse_args(parser)
    infer_video(args)


if __name__ == "__main__":
    main()
