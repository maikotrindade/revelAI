"""The split stage: photographs of album pages in, individual photographs out.

The pipeline is detection (8.1, 8.2), refinement (8.3), validation (8.5) and
cropping (8.4), in that order. Everything geometric happens before a single
output pixel is produced, so that the crop itself is one clean resampling of the
original page.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

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
    "PageReport",
    "SplitReport",
    "split_page",
    "crop_detection",
    "run_split",
    "write_debug_map",
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
    # The first pass is what finds the angle. Later passes are only closing the
    # remaining distance, so they sweep a narrow window around the angle already
    # found instead of the full span, which is most of the cost of a pass.
    polish_span = max(options.angle_step * 4.0, 0.6)
    for _ in range(max(0, options.refine_passes - 1)):
        previous = result.rect
        result = refine_rect(
            prepared,
            previous,
            search_radius=options.search_radius,
            angle_span=min(polish_span, options.angle_span),
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


# --------------------------------------------------------------------------
# Running the stage over a folder of pages
# --------------------------------------------------------------------------


@dataclass
class PageReport:
    """What happened to one page."""

    source: Path
    detections: int = 0
    written: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    needs_review: bool = False
    error: str | None = None
    #: Crops the verifier flagged, as (filename, reasons).
    flagged_crops: list[tuple[str, list[str]]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class SplitReport:
    """What happened to the whole run."""

    pages: list[PageReport] = field(default_factory=list)
    output: Path | None = None
    dry_run: bool = False
    start_index: int = 1

    @property
    def photographs(self) -> int:
        return sum(p.detections for p in self.pages)

    @property
    def failed(self) -> list[PageReport]:
        return [p for p in self.pages if not p.ok]

    @property
    def needing_review(self) -> list[PageReport]:
        return [p for p in self.pages if p.ok and (p.needs_review or p.flagged_crops)]

    @property
    def flagged_crops(self) -> list[tuple[str, list[str]]]:
        return [item for page in self.pages for item in page.flagged_crops]


def write_debug_map(path: Path, image: np.ndarray, result: PageResult) -> Path:
    """Draw what the detector saw, for a human to look at.

    Debug output never goes near the output folder: that folder holds nothing
    but ``photo_XXXXXXXX.png`` files.
    """
    import cv2

    from revelai.io import write_png

    canvas = image.copy()
    if canvas.dtype == np.uint16:
        canvas = (canvas / 257.0).astype(np.uint8)
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    if result.page is not None and result.scale:
        page_corners = result.page.rect.scaled(1.0 / result.scale).corners()
        cv2.polylines(canvas, [page_corners.astype(np.int32)], True, (0, 140, 255), 6)

    thickness = max(2, int(round(max(canvas.shape[:2]) / 500)))
    for number, detection in enumerate(result.detections, start=1):
        colour = (0, 0, 255) if detection.flags else (0, 220, 0)
        cv2.polylines(canvas, [detection.rect.corners().astype(np.int32)], True, colour, thickness)
        cv2.polylines(
            canvas, [detection.crop_corners.astype(np.int32)], True, (255, 180, 0), thickness
        )
        anchor = detection.rect.corners()[0].astype(int)
        cv2.putText(
            canvas,
            str(number),
            tuple(anchor + np.array([8, 40])),
            cv2.FONT_HERSHEY_SIMPLEX,
            thickness * 0.6,
            colour,
            thickness,
        )
    return write_png(path, canvas, overwrite=True)


def run_split(
    inputs: list[Path],
    output: Path,
    options: SplitOptions | None = None,
    *,
    start_index: int | None = None,
    dry_run: bool = False,
    debug_dir: Path | None = None,
    write_metadata: bool = True,
    jobs: int = 1,
    on_page=None,
    review=None,
    inspect=None,
) -> SplitReport:
    """Split every page in ``inputs`` into individually numbered photographs.

    Numbering is global and continuous across the whole run, and the run visits
    pages in natural filename order, so two runs over the same input produce
    exactly the same names.
    """
    from concurrent.futures import ThreadPoolExecutor

    from revelai import __version__
    from revelai.io import ImageReadError, read_image, write_png
    from revelai.naming import photo_filename, resolve_start_index

    options = options or SplitOptions()
    output = Path(output)
    index = resolve_start_index(output, start_index)
    report = SplitReport(output=output, dry_run=dry_run, start_index=index)

    if debug_dir is not None:
        debug_dir = Path(debug_dir)
        if debug_dir.resolve() == output.resolve():
            from revelai.io import OutputCollisionError

            raise OutputCollisionError(
                "--debug-dir must not point at the output folder; "
                "the output folder may contain nothing but photo_XXXXXXXX.png files"
            )

    def analyse(source: Path):
        """Read and detect. Pure, so it is safe to run several at a time."""
        try:
            loaded = read_image(source)
        except ImageReadError as exc:
            return source, None, str(exc)
        try:
            return source, (loaded, split_page(loaded.pixels, options)), None
        except Exception as exc:  # noqa: BLE001 - a bad page must not stop the run
            return source, None, f"could not be processed: {exc}"

    # Detection runs in parallel, but results are consumed in page order, so
    # the numbering does not depend on which page finished first.
    if jobs > 1 and review is None:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            analysed = list(pool.map(analyse, inputs))
    else:
        analysed = [analyse(source) for source in inputs]

    for source, payload, error in analysed:
        page_report = PageReport(source=source)
        if error is not None:
            page_report.error = error
            report.pages.append(page_report)
            if on_page:
                on_page(page_report)
            continue

        loaded, result = payload

        if review is not None:
            result = review(loaded, result)

        page_report.detections = len(result.detections)
        page_report.notes = list(result.notes)
        page_report.needs_review = result.needs_review

        if debug_dir is not None:
            write_debug_map(debug_dir / f"{source.stem}_detected.png", loaded.pixels, result)

        for detection in result.detections:
            name = photo_filename(index)
            page_report.written.append(name)

            # The crop is produced whenever anything needs to look at it, which
            # includes a dry run with --verify: checking a batch before
            # committing to it is exactly when verification is most useful.
            pixels = None
            extra: dict[str, str] = {}
            if not dry_run or inspect is not None:
                pixels = crop_detection(loaded.pixels, detection)

            if inspect is not None:
                pixels, extra, problems = inspect(pixels, detection, source, name)
                if problems:
                    detection.flags.extend(problems)
                    page_report.flagged_crops.append((name, list(problems)))

            if not dry_run:
                text = None
                if write_metadata:
                    corners = ";".join(f"{x:.3f},{y:.3f}" for x, y in detection.crop_corners)
                    text = {
                        "revelai:version": __version__,
                        "revelai:stage": "split",
                        "revelai:source": source.name,
                        "revelai:corners": corners,
                        "revelai:angle": f"{detection.rect.angle:.4f}",
                    }
                    if detection.flags:
                        text["revelai:flags"] = "; ".join(detection.flags)
                    text.update(extra)
                write_png(output / name, pixels, icc_profile=loaded.icc_profile, text=text)
            index += 1

        report.pages.append(page_report)
        if on_page:
            on_page(page_report)

    return report
