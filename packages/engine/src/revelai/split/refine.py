"""Rotated-rectangle refinement by line integral (specification 8.3).

This is the module that actually solves the problem. A candidate coming out of
``detect.py`` is roughly in the right place; this turns it into a crop that is
correct to well under a pixel.

The method fits **one rotated rectangle**, not four independent lines:

* sweep the angle over a small window around the candidate;
* at each angle, slide each of the four edges along its own normal;
* score an edge by the trimmed mean of the absolute directional derivative,
  integrated along the whole edge;
* keep the angle whose four best edges sum highest.

Two properties follow from that, and both matter. Integrating along the entire
edge lets a long straight line - the real border of the print - outscore any
short high-contrast detail inside the picture, however strong that detail is.
Forcing a rotated rectangle prevents the degenerate result where one edge slides
onto a neighbouring photograph and the crop comes out trapezoidal. Fitting the
four lines independently was tried and does not work; it produces skewed crops.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

__all__ = [
    "RotRect",
    "EdgeFit",
    "RefineResult",
    "PreparedImage",
    "prepare_image",
    "refine_rect",
    "score_lines",
]

# Defaults from the specification. They are conservative on purpose: a small
# search radius is what stops an edge jumping onto a touching neighbour.
DEFAULT_SEARCH_RADIUS = 20
DEFAULT_ANGLE_SPAN = 4.0
DEFAULT_ANGLE_STEP = 0.12
DEFAULT_BLUR_SIGMA = 1.2
TRIM_LOW = 0.15
TRIM_HIGH = 0.05
END_MARGIN = 0.10
MAX_SAMPLES = 256


@dataclass(frozen=True)
class RotRect:
    """A rotated rectangle: centre, size, and the angle of its width axis."""

    cx: float
    cy: float
    w: float
    h: float
    angle: float

    def __post_init__(self) -> None:
        # A rotated rectangle has four equivalent descriptions: rotating by 90
        # degrees and swapping the sides gives the same rectangle back. Which
        # one is stored is not cosmetic, because the crop is warped starting
        # from this rectangle's own first corner, so the description decides
        # whether a landscape print comes out landscape or on its side.
        #
        # Folding the angle into [-45, 45) picks the description whose axes are
        # closest to the page's own. Album pages are photographed roughly
        # upright and prints are mounted roughly square to the page, so that is
        # the one that keeps each photograph the way round it sits on the page.
        # Which way up the *scene* is remains a question only --auto-orient can
        # answer; this is about not adding a quarter turn of our own.
        width, height, angle = float(self.w), float(self.h), float(self.angle)
        turns = math.floor((angle + 45.0) / 90.0)
        if turns:
            angle -= 90.0 * turns
            if turns % 2:
                width, height = height, width
        object.__setattr__(self, "cx", float(self.cx))
        object.__setattr__(self, "cy", float(self.cy))
        object.__setattr__(self, "w", width)
        object.__setattr__(self, "h", height)
        object.__setattr__(self, "angle", angle)

    @property
    def axes(self) -> tuple[np.ndarray, np.ndarray]:
        """Unit vectors along the width and the height of the rectangle."""
        rad = math.radians(self.angle)
        cos, sin = math.cos(rad), math.sin(rad)
        return np.array([cos, sin]), np.array([-sin, cos])

    @property
    def centre(self) -> np.ndarray:
        return np.array([self.cx, self.cy])

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def aspect(self) -> float:
        """Long side over short side, always >= 1."""
        short, long_ = sorted((abs(self.w), abs(self.h)))
        return float(long_ / short) if short > 0 else math.inf

    def corners(self) -> np.ndarray:
        """The four corners, clockwise, starting from the rectangle's own top-left."""
        u, v = self.axes
        hw, hh = self.w / 2.0, self.h / 2.0
        c = self.centre
        return np.array(
            [
                c - u * hw - v * hh,
                c + u * hw - v * hh,
                c + u * hw + v * hh,
                c - u * hw + v * hh,
            ]
        )

    def shrunk(self, inset: float) -> RotRect:
        """The same rectangle pulled ``inset`` pixels inward on every edge."""
        return RotRect(self.cx, self.cy, self.w - 2 * inset, self.h - 2 * inset, self.angle)

    def scaled(self, factor: float) -> RotRect:
        """The same rectangle in a coordinate system scaled by ``factor``."""
        return RotRect(
            self.cx * factor, self.cy * factor, self.w * factor, self.h * factor, self.angle
        )

    @classmethod
    def from_corners(cls, points: np.ndarray) -> RotRect:
        """Build from four consecutive corners of a rectangle."""
        pts = np.asarray(points, dtype=np.float64).reshape(4, 2)
        centre = pts.mean(axis=0)
        top = pts[1] - pts[0]
        bottom = pts[2] - pts[3]
        left = pts[3] - pts[0]
        width_axis = (top + bottom) / 2.0
        w = (np.linalg.norm(top) + np.linalg.norm(bottom)) / 2.0
        h = (np.linalg.norm(left) + np.linalg.norm(pts[2] - pts[1])) / 2.0
        angle = math.degrees(math.atan2(width_axis[1], width_axis[0]))
        return cls(centre[0], centre[1], w, h, angle)

    @classmethod
    def from_cv(cls, rect) -> RotRect:
        """Build from an OpenCV ``minAreaRect`` tuple."""
        (cx, cy), (w, h), angle = rect
        return cls(cx, cy, w, h, angle)

    def to_cv(self) -> tuple[tuple[float, float], tuple[float, float], float]:
        return (self.cx, self.cy), (self.w, self.h), self.angle


@dataclass
class EdgeFit:
    """What the search found for one edge of the rectangle."""

    #: Signed displacement along the outward normal, in pixels.
    offset: float
    #: Trimmed mean of the absolute directional derivative along the edge.
    score: float
    #: Width of the gradient peak in pixels. A print border is abrupt and gives
    #: a narrow peak; a phone shadow falls off gently and gives a broad one.
    peak_width: float


@dataclass
class RefineResult:
    rect: RotRect
    score: float
    edges: list[EdgeFit]

    @property
    def weakest_edge(self) -> float:
        return min(e.score for e in self.edges) if self.edges else 0.0

    @property
    def widest_peak(self) -> float:
        return max(e.peak_width for e in self.edges) if self.edges else math.inf


@dataclass
class PreparedImage:
    """Blurred image gradients, computed once and reused by every candidate."""

    gx: np.ndarray
    gy: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.gx.shape[:2]


def prepare_image(image: np.ndarray, blur_sigma: float = DEFAULT_BLUR_SIGMA) -> PreparedImage:
    """Grey, blur, and differentiate once for the whole page.

    The blur is what makes the directional derivative meaningful at sub-pixel
    offsets; without it the score curve is jagged and the peak cannot be
    interpolated.
    """
    array = np.asarray(image)
    if array.ndim == 3:
        channels = array.shape[2]
        if channels == 4:
            array = array[..., :3]
        grey = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY) if array.shape[2] == 3 else array[..., 0]
    else:
        grey = array

    grey = grey.astype(np.float32)
    if array.dtype == np.uint16:
        grey /= 257.0  # keep scores on the same scale as 8-bit input

    if blur_sigma > 0:
        grey = cv2.GaussianBlur(grey, (0, 0), blur_sigma)
    gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return PreparedImage(gx=gx, gy=gy)


def _trimmed_mean(values: np.ndarray, low: float = TRIM_LOW, high: float = TRIM_HIGH) -> np.ndarray:
    """Mean of each row after dropping the lowest and highest tails.

    Dropping the bottom 15 per cent tolerates a stretch of edge that is faint,
    occluded or torn. Dropping the top 5 per cent stops a handful of very bright
    pixels - a speck of dust, a highlight - from carrying the whole edge.
    """
    n = values.shape[1]
    lo = int(n * low)
    hi = n - int(n * high)
    if hi - lo < 1:
        lo, hi = 0, n
    # Only the membership of the middle band matters, not its order, so a
    # partition does the job of a sort at a fraction of the cost. The search
    # evaluates a few million samples per photograph and this is most of it.
    ordered = np.partition(values, (lo, hi - 1), axis=1)
    return ordered[:, lo:hi].mean(axis=1)


def _subpixel_peak(scores: np.ndarray, index: int) -> float:
    """Parabolic interpolation of the peak position, in grid units."""
    if index <= 0 or index >= len(scores) - 1:
        return 0.0
    a, b, c = float(scores[index - 1]), float(scores[index]), float(scores[index + 1])
    denom = a - 2.0 * b + c
    if abs(denom) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))


def _peak_width(scores: np.ndarray, index: int) -> float:
    """Width in pixels of the region where the score stays above half the peak."""
    peak = float(scores[index])
    if peak <= 0:
        return float(len(scores))
    half = peak * 0.5
    left = index
    while left > 0 and scores[left - 1] >= half:
        left -= 1
    right = index
    while right < len(scores) - 1 and scores[right + 1] >= half:
        right += 1
    return float(right - left + 1)


def _score_edge(
    prepared: PreparedImage,
    base: np.ndarray,
    tangent: np.ndarray,
    normal: np.ndarray,
    length: float,
    offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Score every candidate offset for one edge.

    Returns ``(scores, samples_per_offset)``. Sampling ignores 10 per cent at
    each end of the edge, because the corners are where a neighbouring print or
    the page background is most likely to intrude.
    """
    usable = length * (1.0 - 2.0 * END_MARGIN)
    if usable <= 1.0:
        return np.zeros(len(offsets)), np.zeros(0)

    n_samples = int(np.clip(round(usable), 16, MAX_SAMPLES))
    t = np.linspace(-usable / 2.0, usable / 2.0, n_samples)

    # (n_offsets, n_samples) sampling grid: base + tangent*t + normal*offset
    px = base[0] + tangent[0] * t[None, :] + normal[0] * offsets[:, None]
    py = base[1] + tangent[1] * t[None, :] + normal[1] * offsets[:, None]

    map_x = np.ascontiguousarray(px, dtype=np.float32)
    map_y = np.ascontiguousarray(py, dtype=np.float32)
    gx = cv2.remap(prepared.gx, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    gy = cv2.remap(prepared.gy, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    directional = np.abs(gx * normal[0] + gy * normal[1])
    return _trimmed_mean(directional), t


def _batch_edge_scores(
    prepared: PreparedImage,
    bases: np.ndarray,
    tangents: np.ndarray,
    normals: np.ndarray,
    length: float,
    offsets: np.ndarray,
) -> np.ndarray:
    """Score two opposite edges, at every angle and every offset, in one pass.

    ``bases`` and ``normals`` are ``(2, A, 2)`` - two edges, A angles, x and y.
    ``tangents`` is ``(A, 2)``, shared by both edges of the pair. The return is
    ``(2, A, O)``.

    The whole search is assembled into a single sampling grid and handed to one
    ``remap`` call per gradient component. Scoring each angle separately is the
    obvious way to write this and is roughly twenty times slower, because the
    work per call is tiny and the per-call overhead is not.
    """
    n_edges, n_angles, _ = bases.shape
    n_offsets = len(offsets)

    usable = length * (1.0 - 2.0 * END_MARGIN)
    if usable <= 1.0:
        return np.zeros((n_edges, n_angles, n_offsets))
    n_samples = int(np.clip(round(usable), 16, MAX_SAMPLES))
    t = np.linspace(-usable / 2.0, usable / 2.0, n_samples)

    def grid(axis: int) -> np.ndarray:
        return (
            bases[..., axis][:, :, None, None]
            + tangents[..., axis][None, :, None, None] * t[None, None, None, :]
            + normals[..., axis][:, :, None, None] * offsets[None, None, :, None]
        )

    rows = n_edges * n_angles * n_offsets
    map_x = np.ascontiguousarray(grid(0).reshape(rows, n_samples), dtype=np.float32)
    map_y = np.ascontiguousarray(grid(1).reshape(rows, n_samples), dtype=np.float32)

    # remap converts coordinates to fixed point internally, so keep every
    # dimension well inside SHRT_MAX by chunking a very large search.
    chunk = 16384
    sampled_x = np.empty((rows, n_samples), np.float32)
    sampled_y = np.empty((rows, n_samples), np.float32)
    for begin in range(0, rows, chunk):
        stop = min(begin + chunk, rows)
        sampled_x[begin:stop] = cv2.remap(
            prepared.gx,
            map_x[begin:stop],
            map_y[begin:stop],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        sampled_y[begin:stop] = cv2.remap(
            prepared.gy,
            map_x[begin:stop],
            map_y[begin:stop],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    gx = sampled_x.reshape(n_edges, n_angles, n_offsets, n_samples)
    gy = sampled_y.reshape(n_edges, n_angles, n_offsets, n_samples)
    directional = np.abs(
        gx * normals[..., 0][:, :, None, None] + gy * normals[..., 1][:, :, None, None]
    )

    flat = directional.reshape(-1, n_samples)
    return _trimmed_mean(flat).reshape(n_edges, n_angles, n_offsets)


def refine_rect(
    image: np.ndarray | PreparedImage,
    rect: RotRect,
    *,
    search_radius: float = DEFAULT_SEARCH_RADIUS,
    angle_span: float = DEFAULT_ANGLE_SPAN,
    angle_step: float = DEFAULT_ANGLE_STEP,
    blur_sigma: float = DEFAULT_BLUR_SIGMA,
) -> RefineResult:
    """Refine ``rect`` onto the strongest rotated rectangle nearby.

    ``image`` may be a page, or a :class:`PreparedImage` when the gradients have
    already been computed for the whole page - which is what the split pipeline
    does, since every candidate on a page shares them.
    """
    prepared = image if isinstance(image, PreparedImage) else prepare_image(image, blur_sigma)

    radius = int(round(max(1.0, search_radius)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)

    span = max(0.0, float(angle_span))
    step = max(1e-3, float(angle_step))
    n_angles = int(round(span / step))
    deltas = np.linspace(-span, span, 2 * n_angles + 1) if n_angles > 0 else np.array([0.0])

    angles = rect.angle + deltas
    rad = np.radians(angles)
    u = np.stack([np.cos(rad), np.sin(rad)], axis=-1)  # (A, 2) width axis
    v = np.stack([-np.sin(rad), np.cos(rad)], axis=-1)  # (A, 2) height axis
    centre = rect.centre
    hw, hh = rect.w / 2.0, rect.h / 2.0

    # Top and bottom: normals along -v and +v, tangent u, length w.
    horizontal = _batch_edge_scores(
        prepared,
        bases=np.stack([centre - v * hh, centre + v * hh]),
        tangents=u,
        normals=np.stack([-v, v]),
        length=rect.w,
        offsets=offsets,
    )
    # Left and right: normals along -u and +u, tangent v, length h.
    vertical = _batch_edge_scores(
        prepared,
        bases=np.stack([centre - u * hw, centre + u * hw]),
        tangents=v,
        normals=np.stack([-u, u]),
        length=rect.h,
        offsets=offsets,
    )

    per_edge = np.concatenate([horizontal, vertical], axis=0)  # (4, A, O)
    best_per_edge = per_edge.max(axis=2)  # (4, A)
    best_angle = int(np.argmax(best_per_edge.sum(axis=0)))

    fits: list[EdgeFit] = []
    for edge in range(4):
        scores = per_edge[edge, best_angle]
        index = int(np.argmax(scores))
        fits.append(
            EdgeFit(
                offset=float(offsets[index]) + _subpixel_peak(scores, index),
                score=float(scores[index]),
                peak_width=_peak_width(scores, index),
            )
        )

    total = float(best_per_edge[:, best_angle].sum())
    angle = float(angles[best_angle])
    top, bottom, left, right = (f.offset for f in fits)

    axis_u = u[best_angle]
    axis_v = v[best_angle]
    new_centre = centre + axis_u * ((right - left) / 2.0) + axis_v * ((bottom - top) / 2.0)
    width = rect.w + left + right
    height = rect.h + top + bottom

    if width <= 1.0 or height <= 1.0:
        # The search collapsed the rectangle; keep the candidate rather than
        # returning something that cannot be cropped.
        return RefineResult(rect=rect, score=total, edges=fits)

    return RefineResult(
        rect=RotRect(new_centre[0], new_centre[1], width, height, angle),
        score=total,
        edges=fits,
    )


def score_lines(
    prepared: PreparedImage,
    base: np.ndarray,
    tangent: np.ndarray,
    normal: np.ndarray,
    length: float,
    offsets: np.ndarray,
) -> np.ndarray:
    """Line-integral score for a family of parallel lines.

    The same measurement the edge search uses, exposed so detection can ask a
    different question with it: not "where is this edge?" but "is there a
    photograph border running through the middle of this candidate?".
    """
    scores, _ = _score_edge(prepared, base, tangent, normal, length, np.asarray(offsets, float))
    return scores
