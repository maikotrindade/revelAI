"""Cropping and deskewing (specification 8.4).

One resampling per photograph, and no more. The crop and the deskew happen in a
single ``warpPerspective`` on the original full-resolution pixels, with
``INTER_LANCZOS4``. Detection may run on a downscaled copy of the page; cropping
never does.

Nothing in this module adjusts colour, brightness, contrast, saturation,
sharpness or noise. ``split`` crops and deskews. That is the whole contract.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - exercised implicitly on every platform
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError("OpenCV is required: pip install opencv-python") from exc

from revelai.split.refine import RotRect

__all__ = ["crop_quad", "crop_rect", "inset_corners"]


def inset_corners(corners: np.ndarray, inset: float) -> np.ndarray:
    """Pull a quadrilateral ``inset`` pixels inward, towards its own centre.

    The trim exists because a crop taken exactly on the detected border tends to
    keep a hairline of album paper along one edge, which is more objectionable
    in a family album than losing three pixels of sky.
    """
    pts = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    if inset == 0:
        return pts.copy()
    centre = pts.mean(axis=0)
    out = np.empty_like(pts)
    for i, point in enumerate(pts):
        direction = centre - point
        length = float(np.hypot(*direction))
        out[i] = point + direction / length * inset if length > 1e-9 else point
    return out


def crop_quad(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Crop and deskew a quadrilateral out of ``image`` in one resampling.

    The output size is the real pixel size of the crop, rounded: the longer of
    each opposite pair of sides. Nothing is scaled up or down.
    """
    pts = np.asarray(corners, dtype=np.float64).reshape(4, 2)

    top = float(np.hypot(*(pts[1] - pts[0])))
    bottom = float(np.hypot(*(pts[2] - pts[3])))
    left = float(np.hypot(*(pts[3] - pts[0])))
    right = float(np.hypot(*(pts[2] - pts[1])))

    width = max(1, int(round(max(top, bottom))))
    height = max(1, int(round(max(left, right))))

    destination = np.array(
        [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(pts.astype(np.float32), destination)
    return cv2.warpPerspective(
        image,
        transform,
        (width, height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE,
    )


def crop_rect(image: np.ndarray, rect: RotRect, inset: float = 0.0) -> np.ndarray:
    """Crop a rotated rectangle, trimmed inward by ``inset`` pixels."""
    return crop_quad(image, inset_corners(rect.corners(), inset))
