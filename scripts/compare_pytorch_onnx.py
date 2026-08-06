#!/usr/bin/env python
"""Compare PyTorch vs ONNX WAFT optical flow outputs on the same image pair.

Loads both backends, feeds them identically-preprocessed images, and reports
per-pixel error statistics plus side-by-side visualizations.

Usage:
    python scripts/compare_pytorch_onnx.py
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import torch

# Ensure the repo root is on sys.path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from config.parser import parse_args as parse_model_args
from model import fetch_model
from model.waft_base import WAFTBase
from model.waft_onnx import WAFTOnnx
from utils.flow_viz import flow_to_image
from utils.utils import load_ckpt


# ── Paths ──────────────────────────────────────────────────────────────────
ONNX_PATH = os.path.join(_REPO_ROOT, "weights", "waftv2_dav2_i5_448x672.onnx")
PT_CKPT = os.path.join(_REPO_ROOT, "waftv2-ckpts", "dav2", "zero-shot.pth")
PT_CFG = os.path.join(_REPO_ROOT, "config", "a2", "dav2", "chairs-things.json")
IMG1_PATH = os.path.join(
    _REPO_ROOT,
    "data", "astribot_stereo_lrb", "extract_frames", "stereo_left",
    "frame_000150.jpg",
)
IMG2_PATH = os.path.join(
    _REPO_ROOT,
    "data", "astribot_stereo_lrb", "extract_frames", "stereo_left",
    "frame_000154.jpg",
)
OUT_DIR = os.path.join(_REPO_ROOT, "output", "compare")


def _print_header(title: str) -> None:
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")


# ── Helpers ────────────────────────────────────────────────────────────────

def _stats(arr: np.ndarray, label: str = "") -> None:
    """Print min, max, mean, std for an array."""
    tag = f"  {label}: " if label else ""
    print(
        f"{tag}min={arr.min():.4f}  max={arr.max():.4f}  "
        f"mean={arr.mean():.4f}  std={arr.std():.4f}"
    )


def _error_metrics(pt: np.ndarray, onnx: np.ndarray) -> dict:
    """Return a dict of error metrics between two flow arrays."""
    delta = np.abs(pt - onnx)
    epe = np.sqrt(np.sum(delta ** 2, axis=-1))  # endpoint error per pixel
    return {
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "epe_mean": float(epe.mean()),
        "epe_std": float(epe.std()),
        "epe_px1": float((epe > 1.0).mean() * 100),  # % pixels with EPE > 1
        "epe_px3": float((epe > 3.0).mean() * 100),
    }


# ── PyTorch inference helper ───────────────────────────────────────────────

@torch.no_grad()
def _run_pytorch(model, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Run the PyTorch WAFT model on two preprocessed float32 images.

    Args:
        model: WAFTv2 model in eval mode.
        img1, img2: ``(1, 3, H, W)`` float32 numpy arrays in [0, 255] RGB.

    Returns:
        Flow ``(H, W, 2)`` float32 numpy array (final iteration, upsampled).
    """
    t1 = torch.from_numpy(img1).cuda()
    t2 = torch.from_numpy(img2).cuda()
    output = model(t1, t2, iters=5)
    flow = output["flow"][-1]          # [1, 2, H, W]
    return flow[0].permute(1, 2, 0).cpu().numpy()  # → [H, W, 2]


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── 1. Load ONNX model ─────────────────────────────────────────────────
    _print_header("1. ONNX model")
    onnx_model = WAFTOnnx(ONNX_PATH, device="cuda")

    # ── 2. Load PyTorch model ──────────────────────────────────────────────
    _print_header("2. PyTorch model")

    # Re-use the config parser from export_onnx.py
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default=PT_CFG)
    args = parse_model_args(parser)
    args.gpus = [0]

    model = fetch_model(args).cuda()
    load_ckpt(model, PT_CKPT)
    model = model.eval()
    print(f"  Loaded checkpoint: {PT_CKPT}")
    print(f"  Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # ── 3. Preprocess images (shared) ──────────────────────────────────────
    _print_header("3. Preprocessing")

    # Use WAFTBase to preprocess identically for both backends
    base = WAFTBase(bgr_input=True)
    # Borrow target geometry from the ONNX model
    base.target_h = onnx_model.target_h
    base.target_w = onnx_model.target_w
    feed = base.preprocess(IMG1_PATH, IMG2_PATH)
    meta = base._meta
    print(f"  Original:  {meta['orig_h']} × {meta['orig_w']}")
    print(f"  Model:     {base.target_h} × {base.target_w}")
    print(f"  Scale:     {meta['scale_factor']:.4f}")
    print(f"  Tile:      {meta['tile_h']} × {meta['tile_w']}")
    print(f"  Pad (t,l): ({meta['pad_top']}, {meta['pad_left']})")

    # ── 4. Run both backends ───────────────────────────────────────────────
    _print_header("4. Inference")

    # ONNX
    raw_onnx = onnx_model.run(feed)  # raw dict at model resolution
    flow_onnx_raw = raw_onnx["flow"][0].transpose(1, 2, 0)  # → [H, W, 2]
    print(f"  ONNX  raw flow shape: {flow_onnx_raw.shape}")

    # PyTorch
    flow_pt_raw = _run_pytorch(model, feed["image1"], feed["image2"])
    print(f"  PyTorch raw flow shape: {flow_pt_raw.shape}")

    # ── 5. Compare raw outputs (model resolution) ──────────────────────────
    _print_header("5. Raw output comparison (model resolution)")

    _stats(flow_pt_raw, "PyTorch")
    _stats(flow_onnx_raw, "ONNX")

    metrics_raw = _error_metrics(flow_pt_raw, flow_onnx_raw)
    print(f"\n  Max  absolute delta:  {metrics_raw['max_abs']:.4f}")
    print(f"  Mean absolute delta:  {metrics_raw['mean_abs']:.4f}")
    print(f"  EPE mean:             {metrics_raw['epe_mean']:.4f}")
    print(f"  EPE std:              {metrics_raw['epe_std']:.4f}")
    print(f"  EPE > 1 px:           {metrics_raw['epe_px1']:.1f}%")
    print(f"  EPE > 3 px:           {metrics_raw['epe_px3']:.1f}%")

    # ── 6. Compare postprocessed outputs (original resolution) ─────────────
    _print_header("6. Postprocessed comparison (original resolution)")

    flow_pt_orig = base.postprocess({"flow": flow_pt_raw[np.newaxis].transpose(0, 3, 1, 2)})
    flow_onnx_orig = base.postprocess({"flow": flow_onnx_raw[np.newaxis].transpose(0, 3, 1, 2)})

    _stats(flow_pt_orig, "PyTorch")
    _stats(flow_onnx_orig, "ONNX")

    metrics_orig = _error_metrics(flow_pt_orig, flow_onnx_orig)
    print(f"\n  Max  absolute delta:  {metrics_orig['max_abs']:.4f}")
    print(f"  Mean absolute delta:  {metrics_orig['mean_abs']:.4f}")
    print(f"  EPE mean:             {metrics_orig['epe_mean']:.4f}")
    print(f"  EPE > 1 px:           {metrics_orig['epe_px1']:.1f}%")

    # ── 7. Save visualizations ─────────────────────────────────────────────
    _print_header("7. Visualizations")

    # Load original images for overlay
    img1_orig = cv2.imread(IMG1_PATH)
    img1_rgb = cv2.cvtColor(img1_orig, cv2.COLOR_BGR2RGB)

    # Flow visualizations
    flow_pt_vis = flow_to_image(flow_pt_orig, convert_to_bgr=True)
    flow_onnx_vis = flow_to_image(flow_onnx_orig, convert_to_bgr=True)

    # Error map
    epe_map = np.sqrt(np.sum((flow_pt_orig - flow_onnx_orig) ** 2, axis=-1))
    epe_map = np.clip(epe_map / epe_map.max(), 0, 1) if epe_map.max() > 0 else epe_map
    epe_color = cv2.applyColorMap((epe_map * 255).astype(np.uint8), cv2.COLORMAP_HOT)

    cv2.imwrite(os.path.join(OUT_DIR, "flow_pytorch.png"), flow_pt_vis)
    cv2.imwrite(os.path.join(OUT_DIR, "flow_onnx.png"), flow_onnx_vis)
    cv2.imwrite(os.path.join(OUT_DIR, "flow_diff_epe.png"), epe_color)
    cv2.imwrite(os.path.join(OUT_DIR, "frame_a_rgb.png"),
                cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2BGR))

    # Overlay: PyTorch flow on frame
    overlay_pt = cv2.addWeighted(img1_orig, 0.4, flow_pt_vis, 0.6, 0)
    overlay_onnx = cv2.addWeighted(img1_orig, 0.4, flow_onnx_vis, 0.6, 0)
    cv2.imwrite(os.path.join(OUT_DIR, "overlay_pytorch.png"), overlay_pt)
    cv2.imwrite(os.path.join(OUT_DIR, "overlay_onnx.png"), overlay_onnx)

    print(f"  Saved to: {OUT_DIR}/")
    print(f"    flow_pytorch.png   — PyTorch flow visualization")
    print(f"    flow_onnx.png      — ONNX flow visualization")
    print(f"    flow_diff_epe.png  — EPE error heatmap")
    print(f"    overlay_pytorch.png — PyTorch flow overlaid on frame")
    print(f"    overlay_onnx.png   — ONNX flow overlaid on frame")

    # ── 8. Summary ─────────────────────────────────────────────────────────
    _print_header("Summary")
    print(
        f"  Raw model output:  EPE = {metrics_raw['epe_mean']:.4f} ± "
        f"{metrics_raw['epe_std']:.4f} px"
    )
    print(
        f"  Final (orig res):  EPE = {metrics_orig['epe_mean']:.4f} ± "
        f"{metrics_orig['epe_std']:.4f} px"
    )
    if metrics_raw["epe_mean"] < 0.1:
        print("  ✓ PyTorch and ONNX outputs are numerically equivalent.")
    else:
        print(
            f"  ⚠ EPE > 0.1 px — check for preprocessing differences "
            f"({metrics_raw['epe_px1']:.1f}% of pixels have EPE > 1 px)"
        )


if __name__ == "__main__":
    main()
