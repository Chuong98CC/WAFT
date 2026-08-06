"""Backend-agnostic pre/post-processing for WAFT optical flow inference.

``WAFTBase`` is a mixin designed to pair with any inference backend that
exposes ``target_h`` / ``target_w`` (model input geometry) and
``run(feed: dict) -> dict`` (e.g. ``ONNXModel``, ``TRTModel``).  It handles
image loading, BGR→RGB, letterbox resize + pad, and post-inference flow
cropping / scaling back to the original resolution.
"""

from __future__ import annotations

import os

import cv2
import numpy as np


class WAFTBase:
    """Backend-agnostic preprocessing and postprocessing for WAFT flow models.

    Parameters
    ----------
    bgr_input : bool
        If ``True`` (default), input images are assumed to be BGR (the
        OpenCV convention) and are converted to RGB before inference.
    """

    def __init__(self, bgr_input: bool = True) -> None:
        self._bgr_input = bgr_input
        self._meta: dict | None = None  # set by preprocess

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _load_image(img) -> np.ndarray:
        """Return a uint8 H×W×3 image from a path or an existing array."""
        if isinstance(img, str):
            if not os.path.isfile(img):
                raise FileNotFoundError(f"Image not found: {img}")
            arr = cv2.imread(img)
            if arr is None:
                raise FileNotFoundError(
                    f"Could not read image (check format / permissions): {img}"
                )
            return arr
        if isinstance(img, np.ndarray):
            return img
        raise TypeError(
            f"Expected str (path) or np.ndarray, got {type(img).__name__}"
        )

    def _letterbox(self, img: np.ndarray) -> tuple[np.ndarray, dict]:
        """Aspect-preserving resize + centre-pad to ``(target_h, target_w)``.

        Mirrors the logic in ``TRTModel.resize_img`` so the same metadata
        dict can be consumed by postprocessing regardless of backend.
        """
        orig_h, orig_w = img.shape[:2]

        # Scale so the largest dimension fits ; floor to 2 decimals.
        raw_scale = min(self.target_w / orig_w, self.target_h / orig_h)
        scale_factor = np.floor(raw_scale * 100.0) / 100.0
        if scale_factor <= 0:
            scale_factor = raw_scale

        new_w = int(orig_w * scale_factor)
        new_h = int(orig_h * scale_factor)
        img_resized = cv2.resize(img, (new_w, new_h))

        pad_w = self.target_w - new_w
        pad_h = self.target_h - new_h
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        img_padded = cv2.copyMakeBorder(
            img_resized,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

        meta = {
            "orig_h": orig_h,
            "orig_w": orig_w,
            "scale_factor": float(scale_factor),
            "tile_h": new_h,
            "tile_w": new_w,
            "pad_top": int(pad_top),
            "pad_left": int(pad_left),
        }
        return img_padded, meta

    def preprocess(self, img1, img2) -> dict[str, np.ndarray]:
        """Preprocess a pair of images into a feed dict for the backend.

        Parameters
        ----------
        img1 : str or np.ndarray
            Path to first image, or uint8 H×W×3 array.
        img2 : str or np.ndarray
            Path to second image, or uint8 H×W×3 array.

        Returns
        -------
        dict
            ``{"image1": (1,3,H,W) float32, "image2": (1,3,H,W) float32}``
            with values in [0, 255] (the ONNX model normalizes internally).
            Also stores letterbox metadata in ``self._meta`` for
            :meth:`postprocess`.
        """
        img1 = self._load_image(img1)
        img2 = self._load_image(img2)

        # BGR → RGB
        if self._bgr_input:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # Letterbox (metadata from image1 drives both for consistency)
        img1_padded, meta = self._letterbox(img1)
        img2_padded, _ = self._letterbox(img2)
        self._meta = meta

        # HWC uint8 → (1, 3, H, W) float32
        def _to_input(arr: np.ndarray) -> np.ndarray:
            return np.ascontiguousarray(
                arr.transpose(2, 0, 1)
            )[np.newaxis].astype(np.float32)

        return {"image1": _to_input(img1_padded), "image2": _to_input(img2_padded)}

    # ------------------------------------------------------------------
    # Postprocessing
    # ------------------------------------------------------------------

    def postprocess(self, raw_output: dict) -> np.ndarray:
        """Crop padding and rescale flow back to the original resolution.

        Parameters
        ----------
        raw_output : dict
            Backend output dict, must contain ``"flow"`` → ``[1, 2, H, W]``.

        Returns
        -------
        np.ndarray
            Flow array of shape ``(orig_h, orig_w, 2)``, float32.
        """
        if self._meta is None:
            raise RuntimeError(
                "No preprocessing metadata; call preprocess() before postprocess()."
            )

        meta = self._meta
        flow = raw_output["flow"]  # [1, 2, H, W]

        # Crop padding
        pt = meta["pad_top"]
        pl = meta["pad_left"]
        flow = flow[:, :, pt:pt + meta["tile_h"], pl:pl + meta["tile_w"]]

        # Remove batch dim → [2, tile_h, tile_w]
        flow = flow[0]

        # Resize to original resolution
        flow = flow.transpose(1, 2, 0)  # → [tile_h, tile_w, 2]
        flow = cv2.resize(
            flow, (meta["orig_w"], meta["orig_h"]), interpolation=cv2.INTER_LINEAR
        )

        # Scale flow values (flow is in pixels)
        flow[:, :, 0] *= meta["orig_w"] / meta["tile_w"]
        flow[:, :, 1] *= meta["orig_h"] / meta["tile_h"]

        return flow

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __call__(self, img1, img2) -> np.ndarray:
        """Run the full pipeline: preprocess → infer → postprocess.

        Parameters
        ----------
        img1, img2 : str or np.ndarray
            Image paths or uint8 H×W×3 arrays.

        Returns
        -------
        np.ndarray
            Optical flow ``(orig_h, orig_w, 2)``, float32.
        """
        feed = self.preprocess(img1, img2)
        raw = self.run(feed)
        return self.postprocess(raw)
