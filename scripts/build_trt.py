#!/usr/bin/env python
"""Build a TensorRT engine from a WAFT ONNX model.

Usage:
    python scripts/build_trt.py \\
        --onnx weights/waftv2/waftv2_dinov3_i5_640x480.onnx \\
        --output weights/waftv2/waftv2_dinov3_i5_640x480.engine

    # With FP16 (faster, may have minor precision loss):
    python scripts/build_trt.py \\
        --onnx weights/waftv2/waftv2_dinov3_i5_640x480.onnx \\
        --output weights/waftv2/waftv2_dinov3_i5_640x480_fp16.engine \\
        --fp16
"""

from __future__ import annotations

import argparse
import sys
import time

import tensorrt as trt


def build_engine(
    onnx_path: str,
    engine_path: str,
    fp16: bool = False,
    workspace_gb: int = 8,
    timing_iterations: int = 50,
) -> None:
    """Build a TensorRT engine from an ONNX model.

    Parameters
    ----------
    onnx_path : str
        Path to the ONNX model.
    engine_path : str
        Output path for the serialized engine.
    fp16 : bool
        Enable FP16 inference (network I/O stays FP32; compute is FP16).
    workspace_gb : int
        Maximum workspace size in GB for tactic selection.
    timing_iterations : int
        Number of iterations the builder runs each kernel timing trace for.
        Increase for more deterministic performance on complex models.
    """
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX
    print(f"Parsing ONNX model: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("ERROR: Failed to parse the ONNX model.")
            for i in range(parser.num_errors):
                print(f"  [{parser.get_error(i).code()}] {parser.get_error(i).desc()}")
            sys.exit(1)
    print(f"  Parsed {network.num_layers} layers successfully.")

    # Inspect inputs / outputs
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"  Input  [{i}]: {inp.name}  shape={inp.shape}  dtype={inp.dtype}")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"  Output [{i}]: {out.name}  shape={out.shape}  dtype={out.dtype}")

    # Configure builder
    config = builder.create_builder_config()
    config.max_memory_usage_v2 = workspace_gb * 1024**3
    config.builder_optimization_level = 5  # maximum optimization
    try:
        config.set_timing_profile_iterations(timing_iterations)
    except AttributeError:
        pass  # older TRT doesn't expose this

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("  FP16 mode: enabled")

    # Build (this is the slow step — can take several minutes for large models)
    print(f"\nBuilding engine (this may take a while for a {network.num_layers}-layer model) ...")
    t0 = time.time()

    # Serialized plan — try build_serialized_network first (TRT 10+)
    try:
        plan = builder.build_serialized_network(network, config)
    except TypeError:
        # Older API fallback
        engine = builder.build_engine(network, config)
        if engine is None:
            print("ERROR: Engine build failed.")
            sys.exit(1)
        plan = engine.serialize()

    if plan is None:
        print("ERROR: Engine build returned None.")
        sys.exit(1)

    elapsed = time.time() - t0
    size_mb = len(plan) / (1024**2)
    print(f"  Built in {elapsed:.1f}s — engine size: {size_mb:.1f} MB")

    # Write engine
    with open(engine_path, "wb") as f:
        f.write(plan)
    print(f"Engine saved to: {engine_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a TensorRT engine from a WAFT ONNX model."
    )
    parser.add_argument(
        "--onnx", required=True, type=str, help="Path to ONNX model (.onnx)."
    )
    parser.add_argument(
        "--output", required=True, type=str, help="Path for output engine (.engine)."
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable FP16 precision (IO stays FP32, compute in FP16).",
    )
    parser.add_argument(
        "--workspace-gb",
        default=8,
        type=int,
        help="Max workspace memory in GB (default: 8).",
    )
    parser.add_argument(
        "--timing-iterations",
        default=50,
        type=int,
        help="Builder timing iterations (default: 50).",
    )

    args = parser.parse_args()
    build_engine(
        onnx_path=args.onnx,
        engine_path=args.output,
        fp16=args.fp16,
        workspace_gb=args.workspace_gb,
        timing_iterations=args.timing_iterations,
    )


if __name__ == "__main__":
    main()
