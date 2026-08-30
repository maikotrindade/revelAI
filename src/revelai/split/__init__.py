"""The split stage: photographs of album pages in, individual photographs out.

The pipeline is detection (8.1, 8.2), refinement (8.3), validation (8.5) and
cropping (8.4), in that order. Everything geometric happens before a single
output pixel is produced, so that the crop itself is one clean resampling of the
original page.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from revelai.naming import order_on_page
from revelai.split.crop import crop_quad, inset_corners
from revelai.split.detect import DetectParams, PageRegion, find_candidates
from revelai.split.refine import (  # noqa: F401
    DEFAULT_ANGLE_SPAN,
    DEFAULT_ANGLE_STEP,
    DEFAULT_SEARCH_RADIUS,
    PreparedImage,
    RefineResult,
    RotRect,
    prepare_image,
    refine_rect,
)

__all__ = [
    "SplitOptions",
    "Detection",
    "PageResult",
    "split_page",
    "crop_detection",
]


@dataclass
class SplitOptions:
    """Everything the split stage can be told to do differently."""

    min_area: float = 0.005
    max_area: float = 0.9
    inset: int = 3
    search_radius: int = DEFAULT_SEARCH_RADIUS
    angle_span: float = DEFAULT_ANGLE_SPAN
    angle_step: float = DEFAULT_ANGLE_STEP
    #: Detections overlapping by more than this are reported as an overlap.
    overlap_warn: float = 0.05
    #: A box this far from the page median area is reported as unusual.
    area_deviation: float = 0.5
    #: A gradient peak wider than this is a shadow, not the border of a print.
    soft_edge_width: float = 9.0
    #: How many times refinement may be re-applied to its own result.
    refine_passes: int = 3

    def detect_params(self) -> DetectParams:
        return DetectParams(min_area=self.min_area, max_area=self.max_area)


@dataclass
class Detection:
    """One photograph found on one page."""

    #: The refined rectangle, before the safety inset.
    rect: RotRect
    #: The quadrilateral actually cropped, after the inset.
    crop_corners: np.ndarray
    #: Strength of the weakest of the four edges.
    edge_score: float
    #: Widest gradient peak among the four edges, in pixels.
    peak_width: float
    #: Per-detection validation notes, e.g. "overlaps another photograph".
    flags: list[str] = field(default_factory=list)


@dataclass
class PageResult:
    """Everything the split stage learned about one page."""

    detections: list[Detection]
    page: PageRegion | None = None
    scale: float = 1.0
    #: Page-level validation messages (8.5). Never interrupts the run.
    notes: list[str] = field(default_factory=list)
    needs_review: bool = False

    @property
    def flagged(self) -> int:
        return sum(1 for d in self.detections if d.flags)


def _refine_until_still(
    prepared: PreparedImage, candidate: RotRect, options: SplitOptions
) -> RefineResult:
    """Apply the refinement repeatedly until the rectangle stops moving.

    One pass can move each edge by at most the search radius, and the radius is
    deliberately small so that an edge cannot leap onto a touching neighbour.
    That leaves a gap: a candidate that starts more than a radius away from the
    truth - the visible part of a partly covered print, say - can never reach
    it in a single pass.

    Re-applying the pass to its own output closes the gap without widening the
    radius. Each step is still a short, local move, so the guarantee that
    matters is kept: the intermediate positions inside a neighbouring print
    score worse than the real border, so the search climbs to the border and
    stays there rather than crossing into the neighbour.
    """
    result = refine_rect(
        prepared,
        candidate,
        search_radius=options.search_radius,
        angle_span=options.angle_span,
        angle_step=options.angle_step,
    )
    for _ in range(max(0, options.refine_passes - 1)):
        previous = result.rect
        result = refine_rect(
            prepared,
            previous,
            search_radius=options.search_radius,
            angle_span=options.angle_span,
            angle_step=options.angle_step,
        )
        if float(np.abs(result.rect.corners() - previous.corners()).max()) < 0.25:
            break
    return result


def _overlap_fraction(a: RotRect, b: RotRect) -> float:
    import cv2

    area, _ = cv2.intersectConvexConvex(
        a.corners().astype(np.float32), b.corners().astype(np.float32)
    )
    if area <= 0:
        return 0.0
    return float(area / min(a.area, b.area))


def _validate(
    detections: list[Detection], page: PageRegion | None, scale: float, options: SplitOptions
) -> tuple[list[str], bool]:
    """Per-page validation (8.5). Logs, never interrupts.

    Everything reported here ends up in the run summary and marks the page as
    needing review. None of it stops a crop being written: a flagged crop is
    still usually the right crop, and the user is the one who should decide.
    """
    notes: list[str] = []
    needs_review = False

    if not detections:
        return ["no photographs detected"], True

    for i, a in enumerate(detections):
        for b in detections[i + 1 :]:
            overlap = _overlap_fraction(a.rect, b.rect)
            if overlap > options.overlap_warn:
                message = f"two photographs overlap by {overlap:.0%}"
                notes.append(message)
                a.flags.append(message)
                b.flags.append(message)
                needs_review = True

    areas = sorted(d.rect.area for d in detections)
    median = areas[len(areas) // 2]
    for detection in detections:
        if median > 0 and abs(detection.rect.area - median) / median > options.area_deviation:
            message = "size is far from the other photographs on this page"
            detection.flags.append(message)
            notes.append(message)
            needs_review = True

    if page is not None and scale > 0:
        page_rect = page.rect.scaled(1.0 / scale)
        margin = 0.02 * max(page_rect.w, page_rect.h)
        page_corners = page_rect.corners()
        low = page_corners.min(axis=0) + margin
        high = page_corners.max(axis=0) - margin
        for detection in detections:
            corners = detection.rect.corners()
            if (corners < low).any() or (corners > high).any():
                message = "touches the edge of the page; it may be cut off"
                detection.flags.append(message)
                notes.append(message)
                needs_review = True

    for detection in detections:
        if detection.peak_width > options.soft_edge_width:
            # A print border is abrupt. A shadow cast by the phone falls off
            # gently and produces a broad gradient peak, which is how the two
            # are told apart.
            message = "an edge is soft; it may be a shadow rather than a border"
            detection.flags.append(message)
            notes.append(message)
            needs_review = True

    return notes, needs_review


def split_page(
    image: np.ndarray,
    options: SplitOptions | None = None,
    *,
    prepared: PreparedImage | None = None,
) -> PageResult:
    """Find every photograph on one page and work out exactly where to cut it."""
    options = options or SplitOptions()

    candidates, page, scale = find_candidates(image, options.detect_params())
    if not candidates:
        return PageResult(
            detections=[],
            page=page,
            scale=scale,
            notes=["no photographs detected"],
            needs_review=True,
        )

    # Refinement runs at full resolution: the geometry that reaches the warp has
    # to be the geometry of the original pixels, not of a downscaled copy.
    prepared = prepared or prepare_image(image)

    refined: list[Detection] = []
    for candidate in candidates:
        result = _refine_until_still(prepared, candidate, options)
        refined.append(
            Detection(
                rect=result.rect,
                crop_corners=inset_corners(result.rect.corners(), options.inset),
                edge_score=result.weakest_edge,
                peak_width=result.widest_peak,
            )
        )

    order = order_on_page([d.rect for d in refined])
    detections = [refined[i] for i in order]

    notes, needs_review = _validate(detections, page, scale, options)
    return PageResult(
        detections=detections, page=page, scale=scale, notes=notes, needs_review=needs_review
    )


def crop_detection(image: np.ndarray, detection: Detection) -> np.ndarray:
    """Produce the final pixels for one detection, in one resampling."""
    return crop_quad(image, detection.crop_corners)
