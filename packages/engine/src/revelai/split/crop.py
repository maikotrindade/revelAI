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
    """Trim ``inset`` pixels off every edge of a quadrilateral.

    The trim exists because a crop taken exactly on the detected border tends to
    keep a hairline of album paper along one edge, which is more objectionable
    in a family album than losing three pixels of sky.

    Each *edge* is moved inward along its own normal and the corners are then
    re-intersected. Sliding the corners towards the centre instead would be
    simpler and is wrong: a corner moves along the diagonal, so an edge of a
    600 by 400 crop would lose ``inset * cos(theta)`` rather than ``inset``, and
    ``--inset 5`` would trim a different amount horizontally and vertically.
    """
    pts = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    if inset == 0:
        return pts.copy()

    centre = pts.mean(axis=0)
    lines: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(4):
        start, end = pts[i], pts[(i + 1) % 4]
        direction = end - start
        length = float(np.hypot(*direction))
        if length < 1e-9:
            return pts.copy()
        direction = direction / length
        normal = np.array([-direction[1], direction[0]])
        if np.dot(centre - start, normal) < 0:
            normal = -normal
        lines.append((start + normal * inset, direction))

    out = np.empty_like(pts)
    for i in range(4):
        # Corner i is where the edge before it meets the edge after it.
        point_a, dir_a = lines[(i - 1) % 4]
        point_b, dir_b = lines[i]
        denominator = dir_a[0] * dir_b[1] - dir_a[1] * dir_b[0]
        if abs(denominator) < 1e-9:  # parallel edges, degenerate quad
            return pts.copy()
        delta = point_b - point_a
        t = (delta[0] * dir_b[1] - delta[1] * dir_b[0]) / denominator
        out[i] = point_a + dir_a * t
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
