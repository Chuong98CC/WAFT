#!/usr/bin/env python
"""Export a WAFTv2 (DAv2 backbone) model to ONNX format.

Usage:
    python export_onnx.py \\
        --cfg config/a2/dav2/chairs-things.json \\
        --ckpt ckpts/dav2/zero-shot.pth \\
        --height 448 --width 672 \\
        --iters 5 \\
        --output waftv2_dav2.onnx

The exported model expects two RGB images in [0, 255] range and returns
a single optical flow tensor [1, 2, H, W].
"""

import argparse
import os
import sys

import numpy as np
import torch

from config.parser import parse_args
from model import fetch_model
from utils.onnx_utils import ONNXExportWrapper
from utils.utils import load_ckpt


def main():
    parser = argparse.ArgumentParser(
        description="Export WAFTv2 (DAv2 backbone) to ONNX.",
    )
    parser.add_argument(
        "--cfg", required=True, type=str, help="Model JSON config file.",
    )
    parser.add_argument(
        "--ckpt", required=True, type=str, help="Model checkpoint .pth file.",
    )
    parser.add_argument(
        "--height", default=448, type=int,
        help="Input image height (default: 448, must be multiple of 112).",
    )
    parser.add_argument(
        "--width", default=672, type=int,
        help="Input image width (default: 672, must be multiple of 112).",
    )
    parser.add_argument(
        "--iters", default=5, type=int,
        help="Iterative refinement steps to unroll (default: 5).",
    )
    parser.add_argument(
        "--output", default="waftv2_dav2.onnx", type=str,
        help="Output .onnx file path (default: waftv2_dav2.onnx).",
    )
    parser.add_argument(
        "--opset", default=16, type=int,
        help="ONNX opset version (default: 16).",
    )

    args = parse_args(parser)

    # ── Validate resolution ────────────────────────────────────────────────
    pad_factor = 112  # DAv2 backbone
    if args.height % pad_factor != 0 or args.width % pad_factor != 0:
        print(
            f"ERROR: Input resolution ({args.height}x{args.width}) must be "
            f"divisible by {pad_factor} (DAv2 pad factor)."
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
    print(f"ONNX opset:    {args.opset}")
    print(f"Output:        {args.output}")

    # ── Build model ─────────────────────────────────────────────────────────
    print("\nLoading model ...")
    args.gpus = [0]  # required by fetch_model; export always on CPU
    model = fetch_model(args).cpu()
    load_ckpt(model, args.ckpt)
    model = model.eval()

    # ── Wrap for export ────────────────────────────────────────────────────
    export_model = ONNXExportWrapper(model, iters=args.iters)
    export_model = export_model.cpu().eval()

    # ── Dummy inputs ───────────────────────────────────────────────────────
    H, W = args.height, args.width
    image1 = torch.randn(1, 3, H, W).cpu()
    image2 = torch.randn(1, 3, H, W).cpu()

    # ── Export ─────────────────────────────────────────────────────────────
    print("\nExporting to ONNX ...")
    torch.onnx.export(
        export_model,
        (image1, image2),
        args.output,
        input_names=["image1", "image2"],
        output_names=["flow"],
        opset_version=args.opset,
        dynamic_axes=None,  # static shape export
        dynamo=False,  # use TorchScript tracer (handles FloatFunctional)
    )

    file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Exported: {args.output} ({file_size_mb:.1f} MB)")

    # ── Verify ONNX validity ───────────────────────────────────────────────
    try:
        import onnx
        print("\nValidating ONNX model ...")
        onnx_model = onnx.load(args.output)
        onnx.checker.check_model(onnx_model)
        print("ONNX model structure is valid.")
    except ImportError:
        print("\nSkipping ONNX model check (onnx not installed).")
        onnx_model = None

    # ── Verify numerical consistency ───────────────────────────────────────
    try:
        import onnxruntime

        print("Running numerical verification (PyTorch vs ONNX Runtime) ...")

        with torch.no_grad():
            pt_out = export_model(image1, image2).numpy()

        ort_session = onnxruntime.InferenceSession(
            args.output,
            providers=["CPUExecutionProvider"],
        )
        ort_out = ort_session.run(
            None, {"image1": image1.numpy(), "image2": image2.numpy()}
        )[0]

        delta = np.abs(pt_out - ort_out)
        max_delta = np.max(delta)
        mean_delta = np.mean(delta)
        mean_abs_flow = np.mean(np.abs(pt_out))
        rel_error = mean_delta / (mean_abs_flow + 1e-8)

        print(f"Mean absolute flow: {mean_abs_flow:.4f}")
        print(f"Max absolute delta:  {max_delta:.2e}")
        print(f"Mean absolute delta: {mean_delta:.2e}")
        print(f"Relative error:      {rel_error:.4f}")

        if rel_error < 0.01:
            print(
                "Verification PASSED — relative error < 1% "
                "(expected for FP32 cross-backend inference)."
            )
        else:
            print(
                f"WARNING: Relative error ({rel_error:.4f}) exceeds "
                "tolerance (0.01). The ONNX model may have numerical issues."
            )
    except ImportError:
        print(
            "\nSkipping numerical verification (onnxruntime not installed)."
        )

    print(f"\nDone. Inputs: (1, 3, {H}, {W}) → Output: (1, 2, {H}, {W})")


if __name__ == "__main__":
    main()
