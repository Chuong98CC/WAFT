"""Flow post-processing for PT2-based WAFT inference.

Mirrors ``WAFTBase.postprocess`` (ONNX / TensorRT backends): crop the
letterbox padding and rescale the flow back to the original resolution.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch


def postprocess_flow(flow: torch.Tensor, meta: dict) -> np.ndarray:
    """Crop padding and rescale a raw model output to the original size.

    Args:
        flow: Model output ``[1, 2, H, W]`` (bf16, any device) at the
            letterboxed ``(target_h, target_w)`` resolution.
        meta: Letterbox metadata from
            :func:`infer_pt2.utils.img_utils.letterbox` (of image 1).

    Returns:
        np.ndarray: Flow of shape ``(orig_h, orig_w, 2)``, float32 (values
        are in pixels at the original resolution).
    """
    tile_h = meta["tile_h"]
    tile_w = meta["tile_w"]
    pad_top = meta["pad_top"]
    pad_left = meta["pad_left"]

    # Crop the letterbox padding (region that carries no image content).
    flow = flow[:, :, pad_top:pad_top + tile_h, pad_left:pad_left + tile_w]

    # Drop batch dim → [tile_h, tile_w, 2] float32 on CPU.
    flow = flow[0].float().cpu().numpy().transpose(1, 2, 0)

    # Resize to the original resolution.
    flow = cv2.resize(
        flow, (meta["orig_w"], meta["orig_h"]), interpolation=cv2.INTER_LINEAR
    )

    # Rescale flow values (flow is in pixels).
    flow[:, :, 0] *= meta["orig_w"] / tile_w
    flow[:, :, 1] *= meta["orig_h"] / tile_h

    return flow
