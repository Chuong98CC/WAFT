"""ONNX export helpers for WAFT models.

Provides FloatFunctional + Normalize patching and a thin wrapper that
unrolls the iterative refinement loop into a single static forward pass
suitable for torch.onnx.export().
"""

import torch
import torch.nn as nn


class Add(nn.Module):
    """Element-wise addition — drop-in for FloatFunctional.

    FloatFunctional exposes ``.add(a, b)`` but torch.fx tracing can
    route that through ``forward`` via ``__getattribute__``.
    """

    def forward(self, a, b):
        return a + b

    def __getattribute__(self, name):
        if name == "add":
            return self.forward
        return super().__getattribute__(name)


def patch_float_functional(module):
    """Recursively replace every nn.quantized.FloatFunctional with Add.

    FloatFunctional is not recognized by ONNX, but it is functionally
    identical to plain ``+`` for FP32 inference.  This walks the full
    module tree (including nested sub-modules) so that third-party
    DAv2 blocks, the Twins encoder, and head.py are all covered.

    Args:
        module (nn.Module): Root module to patch (modified in-place).
    """
    for name, child in module.named_children():
        if isinstance(child, nn.quantized.FloatFunctional):
            setattr(module, name, Add())
        else:
            patch_float_functional(child)


def _make_normalize_forward(mean, std):
    """Return a ``forward(self, img)`` that normalizes without torchvision.

    torchvision.transforms.Normalize contains a data-dependent guard
    ``if (std == 0).any()`` that fails under ``torch.export`` (used by
    PyTorch ≥ 2.6 for ONNX tracing).  This closure computes the same
    affine transform with pure arithmetic ops.
    """
    # [1, 3, 1, 1] for broadcasting over [B, 3, H, W]
    mean_t = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)

    def _forward(self, img):
        """img: [B, 3, H, W] in range [0, 255], RGB.  Returns normalized tensor."""
        x = img / 255.0
        x = (x - mean_t.to(x.device)) / std_t.to(x.device)
        return x.contiguous()

    return _forward


def patch_normalize_image(model):
    """Replace the model's ``normalize_image`` with an ONNX-safe version.

    The new method computes ``(img/255 - mean) / std`` using explicit
    arithmetic instead of ``torchvision.transforms.Normalize``, avoiding
    a data-dependent branch that ``torch.export`` cannot trace.

    Args:
        model (nn.Module): A WAFTv2 or ViTWarpV8 model instance.
            Modified in-place.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    model.normalize_image = _make_normalize_forward(mean, std).__get__(
        model, type(model)
    )


class ONNXExportWrapper(nn.Module):
    """Wraps a WAFT model to expose a clean ONNX-friendly forward graph.

    - Unrolls the iterative refinement loop (``iters`` is baked in).
    - Returns only the final flow tensor ``[B, 2, H, W]`` (no dicts, no
      info predictions, no intermediate iterations).
    - The ``flow_gt`` conditional path in the underlying model is never
      taken during inference, so the NF-loss computation is not traced.

    Args:
        model (nn.Module): A patched WAFTv2 (or ViTWarpV8) model in eval mode.
        iters (int): Number of refinement iterations to unroll (default 5).
    """

    def __init__(self, model, iters=5):
        super().__init__()
        self.model = model
        self.iters = iters

    def forward(self, image1, image2):
        """Estimate optical flow.

        Args:
            image1 (torch.Tensor): First frame, ``[B, 3, H, W]``, RGB, [0, 255].
            image2 (torch.Tensor): Second frame, ``[B, 3, H, W]``, RGB, [0, 255].

        Returns:
            torch.Tensor: Optical flow ``[B, 2, H, W]`` (final iteration).
        """
        output = self.model(image1, image2, iters=self.iters)
        return output["flow"][-1]
