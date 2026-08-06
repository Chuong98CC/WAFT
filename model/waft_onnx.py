"""WAFT optical flow inference via ONNX Runtime.

Concrete class inheriting :class:`WAFTBase` (pre/post-processing) and
:class:`ONNXModel` (session management).
"""

from __future__ import annotations

from model.base.base_onnx import ONNXModel
from model.waft_base import WAFTBase


class WAFTOnnx(WAFTBase, ONNXModel):
    """WAFT optical flow inference backed by ONNX Runtime.

    Parameters
    ----------
    onnx_path : str
        Path to the ``.onnx`` model file.
    device : str
        ``"cuda"`` (CUDA EP with CPU fallback) or ``"cpu"``.
    bgr_input : bool
        If ``True`` (default), input images are BGR and will be converted
        to RGB internally.
    """

    def __init__(
        self, onnx_path: str, device: str = "cuda", bgr_input: bool = True
    ) -> None:
        ONNXModel.__init__(self, onnx_path, device)
        WAFTBase.__init__(self, bgr_input=bgr_input)
