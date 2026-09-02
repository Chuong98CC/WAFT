"""WAFT optical flow inference via TensorRT.

Concrete class inheriting :class:`WAFTBase` (pre/post-processing) and
:class:`TRTModel` (engine management + execution).

The TensorRT import is deferred so the module is importable even when
``tensorrt`` is not installed (e.g. in export-only environments).
"""

from __future__ import annotations

from model.waft_base import WAFTBase


def WAFTTrl(engine_path: str, bgr_input: bool = True):
    """Create a WAFT optical flow inference model backed by TensorRT.

    The class is constructed dynamically so ``tensorrt`` is only imported
    when a TRT engine is actually loaded.

    Parameters
    ----------
    engine_path : str
        Path to the ``.engine`` (or ``.trt`` / ``.plan``) TensorRT engine file.
    bgr_input : bool
        If ``True`` (default), input images are BGR and will be converted
        to RGB internally.

    Returns
    -------
    _WAFTTrl
        An instance that provides :meth:`preprocess`, :meth:`run`,
        :meth:`postprocess`, and :meth:`__call__` (full pipeline).
    """
    from model.base.base_trt import TRTModel

    class _WAFTTrl(WAFTBase, TRTModel):
        """WAFT + TensorRT inference (lazily constructed)."""

        def __init__(self, engine_path: str, bgr_input: bool = True) -> None:
            TRTModel.__init__(self, engine_path)
            WAFTBase.__init__(self, bgr_input=bgr_input)

        # --------------------------------------------------------------
        # Bridge: WAFTBase calls self.run(feed), TRTModel exposes self._run
        # --------------------------------------------------------------

        def run(self, feed: dict) -> dict:
            """Run inference.

            Parameters
            ----------
            feed : dict
                ``{"image1": np.ndarray, "image2": np.ndarray}`` —  (1,3,H,W)
                float32 arrays in [0, 255], produced by
                :meth:`WAFTBase.preprocess`.

            Returns
            -------
            dict
                ``{"flow": np.ndarray}`` — (1,2,H,W) float32 optical flow.
            """
            return self._run(feed, np_output=True)

    return _WAFTTrl(engine_path, bgr_input=bgr_input)
