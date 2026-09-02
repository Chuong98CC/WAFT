#!/usr/bin/env python
"""Run WAFT optical flow inference on an image pair using a PT2 artifact.

Usage:
    python infer_pt2/infer_image_pt2.py \
        --image1 path/to/frame_0001.png \
        --image2 path/to/frame_0002.png \
        --output flow_01.flo

The artifact expects images of exactly the exported resolution: they are
letterboxed internally (aspect-preserving resize + centre-pad, like the
ONNX/TensorRT backends) and the flow is mapped back to the original size.

Minimal by design: no visualization.  To visualize, import helpers from the
main repo, e.g. ``utils.flow_viz.flow_to_image`` or
``utils.frame_utils.writeFlow`` on the returned array.
"""

import argparse
import os
import sys
import time

# Make the WAFT repo root importable and REMOVE this script's directory from
# sys.path: <repo>/utils is a namespace package (no __init__.py) and would be
# shadowed by the regular package at <repo>/infer_pt2/utils, breaking
# `from utils.utils import ...` inside the repo's own modules.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_SCRIPT_DIR))
if _SCRIPT_DIR in sys.path:
    sys.path.remove(_SCRIPT_DIR)

import numpy as np

from infer_pt2.waftv2_pt2 import WAFTv2_PT2
from utils.frame_utils import writeFlow


def main():
    parser = argparse.ArgumentParser(
        description="Run WAFT optical flow inference on an image pair (PT2 backend).",
    )
    parser.add_argument(
        "--image1", required=True, type=str, help="Path to the first image."
    )
    parser.add_argument(
        "--image2", required=True, type=str, help="Path to the second image."
    )
    parser.add_argument(
        "--pt2",
        default="weights/waftv2/waftv2_bf16.pt2",
        type=str,
        help="Path to the exported .pt2 artifact "
             "(default: weights/waftv2/waftv2_bf16.pt2).",
    )
    parser.add_argument(
        "--output", default=None, type=str,
        help="Optional path to write the flow as a .flo file.",
    )
    parser.add_argument(
        "--device", default=None, type=str, choices=["cuda", "cpu"],
        help="Device to run on (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--no-bgr-input",
        action="store_true",
        help="Images are already RGB (skip internal BGR→RGB conversion).",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")

    # ── Load model ───────────────────────────────────────────────────────
    model = WAFTv2_PT2(
        args.pt2, device=device, bgr_input=not args.no_bgr_input
    )

    # ── Run inference ────────────────────────────────────────────────────
    t0 = time.time()
    flow = model(args.image1, args.image2)  # [H_orig, W_orig, 2] float32
    elapsed = time.time() - t0

    h, w = flow.shape[:2]
    mag = np.sqrt((flow ** 2).sum(axis=-1))
    print(f"\nInput images: {args.image1}, {args.image2}")
    print(f"Flow: {w}x{h}, mean |flow| = {mag.mean():.3f} px, "
          f"max |flow| = {mag.max():.3f} px")
    print(f"Ran in {elapsed * 1000:.1f} ms ({elapsed:.1f} s)")

    # ── Optional .flo dump (data, not visualization) ─────────────────────
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        writeFlow(args.output, flow)
        print(f"Flow saved to: {args.output}")


if __name__ == "__main__":
    main()
