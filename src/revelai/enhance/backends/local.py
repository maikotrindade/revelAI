"""The local backend: everything runs on the user's machine (specification 10.3).

This is the default, and it is what makes the privacy promise keepable: no API
key, nothing leaves the computer, and the tool is fully usable with no account
anywhere.

Denoising and dust removal here are classical algorithms, not models. They are
deterministic, they invent nothing, and they run instantly, which is exactly why
the pipeline tries them before reaching for anything heavier.

Face restoration, colourisation and super-resolution need a model. Those are
loaded from the model cache if the user has fetched one, and reported as
unavailable otherwise. They are never a silent no-op.
"""

from __future__ import annotations

import cv2
import numpy as np

from revelai.enhance.backends.base import EnhancerBackend, Operation

__all__ = ["LocalBackend"]


class LocalBackend(EnhancerBackend):
    """Classical restoration, plus any models present in the local cache."""

    name = "local"
    sends_images_away = False

    def __init__(self, models=None) -> None:
        from revelai.enhance.backends.models import ModelCache

        self.models = models if models is not None else ModelCache()

    def capabilities(self) -> set[str]:
        available = {"denoise", "dust"}
        available |= self.models.available_operations()
        return available

    def unavailable_reason(self, operation: str) -> str:
        """Say what is missing and what the user can actually do about it."""
        if operation not in ("faces", "colorize", "upscale"):
            return super().unavailable_reason(operation)

        registered = self.models.specs(operation)
        variable = f"REVELAI_MODEL_{operation.upper()}"
        if registered:
            return (
                f"the local backend needs a model for {operation} and none is cached. "
                f"Download {registered[0].name} from {registered[0].url} into "
                f"{self.models.directory}, point at your own with {variable}, "
                f"or use --backend with a hosted provider."
            )
        return (
            f"RevelAI has no local model registered for {operation}. Point at your own "
            f"ONNX model with {variable}, or use --backend with a hosted provider."
        )

    # -- classical ---------------------------------------------------------

    def denoise(self, image: np.ndarray) -> tuple[np.ndarray, Operation]:
        """Non-local means: average pixels whose *neighbourhoods* look alike.

        On a scan of a print this removes grain and sensor noise while leaving
        edges alone, because an edge pixel has no lookalikes on the other side
        of the edge to be averaged with.
        """
        working, restore = _as_bgr8(image)
        result = cv2.fastNlMeansDenoisingColored(working, None, 4, 4, 7, 21)
        return restore(result), Operation(
            "denoise", "opencv.fastNlMeansDenoisingColored", cv2.__version__
        )

    def dust(self, image: np.ndarray) -> tuple[np.ndarray, Operation]:
        """Find dust and scratches, then inpaint only those pixels.

        Dust on a print is small, high contrast and *lighter or darker than
        everything around it*. A median filter destroys it, so the difference
        between the image and its own median isolates it. Only the isolated
        pixels are repaired, which is why this cannot smear the photograph the
        way a blanket median would.
        """
        working, restore = _as_bgr8(image)
        grey = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
        median = cv2.medianBlur(grey, 5)
        difference = cv2.absdiff(grey, median)

        # A speck stands out from the local variation. Scale the threshold to
        # the image's own noise so a grainy scan is not entirely "dust".
        noise = float(np.median(difference)) + 1.0
        mask = (difference > max(12.0, noise * 6.0)).astype(np.uint8)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        )
        # Anything large is part of the picture, not a defect on top of it.
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        limit = max(24, int(0.0002 * grey.size))
        keep = np.zeros(count, dtype=bool)
        for index in range(1, count):
            if stats[index, cv2.CC_STAT_AREA] <= limit:
                keep[index] = True
        speck_mask = keep[labels].astype(np.uint8) * 255
        if not speck_mask.any():
            return restore(working), Operation(
                "dust", "median-difference + inpaint", cv2.__version__
            )
        speck_mask = cv2.dilate(speck_mask, np.ones((3, 3), np.uint8))
        repaired = cv2.inpaint(working, speck_mask, 3, cv2.INPAINT_TELEA)
        return restore(repaired), Operation("dust", "median-difference + inpaint", cv2.__version__)

    # -- model backed ------------------------------------------------------

    def upscale(self, image: np.ndarray, factor: int) -> tuple[np.ndarray, Operation]:
        model = self.models.load("upscale", factor=factor)
        if model is None:
            self._refuse("upscale")
        return model.apply(image, factor=factor), model.operation("upscale")

    def faces(self, image: np.ndarray) -> tuple[np.ndarray, Operation]:
        model = self.models.load("faces")
        if model is None:
            self._refuse("faces")
        return model.apply(image), model.operation("faces", generative=True)

    def colorize(self, image: np.ndarray) -> tuple[np.ndarray, Operation]:
        model = self.models.load("colorize")
        if model is None:
            self._refuse("colorize")
        return model.apply(image), model.operation("colorize", generative=True)


def _as_bgr8(image: np.ndarray):
    """OpenCV's restoration functions are 8-bit only; put the depth back after.

    A 16-bit input is scaled down for the operation and scaled back up
    afterwards. That loses precision the operation could not have used anyway,
    and it keeps the pipeline's promise that a 16-bit photograph stays a 16-bit
    photograph.
    """
    array = np.asarray(image)
    if array.dtype == np.uint16:
        eight = (array / 257.0).round().astype(np.uint8)

        def restore(result: np.ndarray) -> np.ndarray:
            return (result.astype(np.float32) * 257.0).round().clip(0, 65535).astype(np.uint16)

        return eight, restore

    return array, lambda result: result
