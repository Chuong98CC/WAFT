"""torch.export (``.pt2``) helpers: normalize patch + unrolled wrapper.

Mirrors ``utils/onnx_utils.py`` (ONNXExportWrapper / patches), tuned for
bfloat16 export: the image-normalization constants are created in the
model dtype so the exported graph stays uniformly bf16.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# ImageNet stats used by WAFTv2.normalize_image (matches repo defaults).
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]


def patch_normalize_image(model: nn.Module, dtype: torch.dtype = torch.bfloat16):
    """Replace the model's ``normalize_image`` with a pt2-safe version.

    ``torchvision.transforms.Normalize`` contains a data-dependent guard
    (``if (std == 0).any()``) that fails under ``torch.export``.  The new
    method computes ``(img/255 - mean) / std`` with pure arithmetic in
    ``dtype``.

    The mean/std constants are registered as model buffers (not baked as
    raw tensors in the traced graph), so that ``module.to(device)`` moves
    them together with the exported graph — allowing a CPU export to run
    on CUDA at inference time.

    Args:
        model (nn.Module): A WAFTv2 model instance.  Modified in-place.
        dtype: dtype for the normalization constants.
    """
    # [1, 3, 1, 1] for broadcasting over [B, 3, H, W]
    model.register_buffer(
        "norm_mean", torch.tensor(IMAGE_MEAN, dtype=dtype).view(1, 3, 1, 1)
    )
    model.register_buffer(
        "norm_std", torch.tensor(IMAGE_STD, dtype=dtype).view(1, 3, 1, 1)
    )

    def _forward(self, img):
        """img: [B, 3, H, W] in range [0, 255], RGB.  Returns normalized tensor."""
        return ((img / 255.0) - self.norm_mean) / self.norm_std

    model.normalize_image = _forward.__get__(model, type(model))


class PT2ExportWrapper(nn.Module):
    """Wraps a WAFT model to expose a clean ``torch.export``-friendly forward.

    - Unrolls the iterative refinement loop (``iters`` is baked in).
    - Returns only the final flow tensor ``[B, 2, H, W]`` (no dicts, no
      info predictions, no intermediate iterations).
    - The ``flow_gt`` conditional path in the underlying model is never
      taken during inference, so the NF-loss computation is not traced.

    Args:
        model (nn.Module): A patched WAFTv2 model in eval mode.
        iters (int): Number of refinement iterations to unroll (default 5).
    """

    def __init__(self, model: nn.Module, iters: int = 5) -> None:
        super().__init__()
        self.model = model
        self.iters = iters

    def forward(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        """Estimate optical flow.

        Args:
            image1 (torch.Tensor): First frame, ``[B, 3, H, W]``, RGB, [0, 255].
            image2 (torch.Tensor): Second frame, ``[B, 3, H, W]``, RGB, [0, 255].

        Returns:
            torch.Tensor: Optical flow ``[B, 2, H, W]`` (final iteration).
        """
        output = self.model(image1, image2, iters=self.iters)
        return output["flow"][-1]
