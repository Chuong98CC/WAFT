#!/usr/bin/env python
"""Export a WAFTv2 model to a torch.export ``.pt2`` artifact in bfloat16.

Same config / checkpoint / resolution as ``scripts/export_onnx.sh``:
    python infer_pt2/export_pt2.py \\
        --cfg config/a2/dinov3/chairs-things.json \\
        --ckpt waftv2-ckpts/dinov3/zero-shot.pth \\
        --height 480 --width 640 \\
        --iters 5 \\
        --output weights/waftv2/waftv2_bf16.pt2

The exported artifact (ExportedProgram) expects two RGB images in [0, 255]
range as bfloat16 tensors at the exported resolution and returns a single
optical flow tensor [1, 2, H, W].  Run inference on it with
``infer_pt2/infer_image_pt2.py`` (:class:`WAFTv2_PT2`).
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

import torch

from config.parser import parse_args
from infer_pt2.utils.export_utils import PT2ExportWrapper, patch_normalize_image
from model import fetch_model
from utils.onnx_utils import patch_float_functional
from utils.utils import load_ckpt

# Model-internal pad factor per feature encoder (WAFTv2.factor).
PAD_FACTORS = {"twins": 32, "dav2": 112, "dinov3": 16}


def main():
    parser = argparse.ArgumentParser(
        description="Export WAFTv2 to a torch.export (.pt2) artifact (bf16).",
    )
    parser.add_argument(
        "--cfg", required=True, type=str, help="Model JSON config file.",
    )
    parser.add_argument(
        "--ckpt", required=True, type=str, help="Model checkpoint .pth file.",
    )
    parser.add_argument(
        "--height", default=448, type=int,
        help="Input image height (default: 448).",
    )
    parser.add_argument(
        "--width", default=672, type=int,
        help="Input image width (default: 672).",
    )
    parser.add_argument(
        "--iters", default=5, type=int,
        help="Iterative refinement steps to unroll (default: 5).",
    )
    parser.add_argument(
        "--output", default="weights/waftv2/waftv2_bf16.pt2", type=str,
        help="Output .pt2 artifact path (default: weights/waftv2/waftv2_bf16.pt2).",
    )
    parser.add_argument(
        "--pad-factor", default=None, type=int,
        help="Divisibility check factor (default: derived from the cfg's "
             "feature_encoder: twins=32, dav2=112, dinov3=16).",
    )

    args = parse_args(parser)

    # ── Validate resolution ────────────────────────────────────────────────
    # NOTE: parse_args() only copies non-None CLI values onto the config
    # namespace, so the pad-factor flag may be absent from `args`.
    pad_factor = (
        getattr(args, "pad_factor", None)
        or PAD_FACTORS.get(args.feature_encoder, 16)
    )
    if args.height % pad_factor != 0 or args.width % pad_factor != 0:
        print(
            f"ERROR: Input resolution ({args.height}x{args.width}) must be "
            f"divisible by {pad_factor} (pad factor of the "
            f"'{args.feature_encoder}' encoder)."
        )
        print(
            f"Suggest: height={((args.height // pad_factor) * pad_factor)}, "
            f"width={((args.width // pad_factor) * pad_factor)}"
        )
        sys.exit(1)

    print(f"Configuration: {args.cfg}")
    print(f"Checkpoint:    {args.ckpt}")
    print(f"Resolution:    {args.height} x {args.width}")
    print(f"Iterations:    {args.iters}")
    print(f"Precision:     bfloat16")
    print(f"Output:        {args.output}")

    # ── Build model ─────────────────────────────────────────────────────────
    print("\nLoading model ...")
    args.gpus = [0]  # required by fetch_model
    model = fetch_model(args).cpu()
    load_ckpt(model, args.ckpt)
    model = model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # ── Patch & wrap for torch.export ──────────────────────────────────────
    # FloatFunctional is opaque to torch.export; replace it with plain `+`.
    patch_float_functional(model)
    # torchvision Normalize has data-dependent guards; replace with bf16-safe
    # arithmetic so the exported graph is uniformly bfloat16.
    patch_normalize_image(model, dtype=torch.bfloat16)

    export_model = PT2ExportWrapper(model, iters=args.iters)
    export_model = export_model.bfloat16().eval()

    # ── Dummy inputs (bf16, values in [0, 255]) ────────────────────────────
    H, W = args.height, args.width

    # The model allocates flow grids with explicit `device=` ops (torch.zeros,
    # coords_grid), so the export-time device is baked into the graph.  Export
    # on the device the artifact will run on; bf16 needs CUDA anyway.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print(
            "WARNING: CUDA not available — the artifact will be CPU-baked "
            "and can only run on CPU."
        )
    export_model = export_model.to(device)
    image1 = (torch.rand(1, 3, H, W, device=device) * 255.0).bfloat16()
    image2 = (torch.rand(1, 3, H, W, device=device) * 255.0).bfloat16()

    # ── Export ─────────────────────────────────────────────────────────────
    # No dynamic_shapes → all input dims are static; the artifact then only
    # runs at the exported (H, W) resolution.
    print(f"\nExporting to PT2 on {device} (this may take a few minutes) ...")
    t0 = time.time()
    exported_program = torch.export.export(export_model, (image1, image2))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.export.save(exported_program, args.output)
    elapsed = time.time() - t0

    file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Exported: {args.output} ({file_size_mb:.1f} MB) in {elapsed:.1f}s")

    # Sanity check: report the input names / sizes baked into the artifact.
    for node in exported_program.graph.nodes:
        if node.op == "placeholder" and node.name in exported_program.graph_signature.user_inputs:
            val = node.meta["val"]
            print(f"  input '{node.name}': shape {tuple(val.shape)}, dtype {val.dtype}")

    # ── Verify numerical consistency (PyTorch bf16 vs artifact) ────────────
    # The exported artifact runs on the export-time device, so both sides are
    # evaluated on `device` (no cross-device move of the graph module).
    print(f"\nRunning numerical verification (bf16 PyTorch vs PT2 artifact) on {device} ...")
    try:
        with torch.no_grad():
            pt_out = export_model(image1, image2)

        reloaded = torch.export.load(args.output)  # params restored on `device`
        gm = reloaded.module()  # inference-only graph; no train/eval modes
        pt2_out = gm(image1, image2)

        delta = (pt2_out.float() - pt_out.float()).abs()
        max_delta = delta.max().item()
        mean_delta = delta.mean().item()
        mean_abs_flow = pt_out.float().abs().mean().item()
        rel_error = mean_delta / (mean_abs_flow + 1e-8)

        print(f"Mean absolute flow: {mean_abs_flow:.4f}")
        print(f"Max absolute delta:  {max_delta:.2e}")
        print(f"Mean absolute delta: {mean_delta:.2e}")
        print(f"Relative error:      {rel_error:.4f}")

        if rel_error < 0.01:
            print(
                "Verification PASSED — relative error < 1% "
                "(expected for bf16 inference)."
            )
        else:
            print(
                f"WARNING: Relative error ({rel_error:.4f}) exceeds "
                "tolerance (0.01). The artifact may have numerical issues."
            )
    except Exception as e:  # keep the artifact even if verification fails
        import traceback
        print(f"\nNumerical verification FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()

    print(f"\nDone. Inputs: (1, 3, {H}, {W}) bf16 → Output: (1, 2, {H}, {W})")


if __name__ == "__main__":
    main()
