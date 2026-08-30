"""Page isolation and initial candidates (specification 8.1 and 8.2).

The job here is recall, not precision. Anything this module proposes is handed
to ``refine.py``, which fixes the geometry; what matters is that every real
photograph produces a candidate roughly in the right place, and that neither the
table nor the album page itself ever becomes one.

Two things make this harder than the flatbed-scanner case every prior tool
assumes. The background is a table, so it is the *darkest* thing in the frame
rather than the brightest, and a single global threshold segments the whole page
as one object. And the light across a phone photograph is uneven, so the album
paper is not one colour: on the fixtures used by the tests it runs from about
190 to 250. Comparing pixels against a single estimated paper colour therefore
finds the lit half of the page, not the photographs.

Both are handled by estimating the illumination first and comparing everything
against it, so "differs from the paper" means "differs from the paper *here*".

No single strategy is reliable, so two run side by side and their results merge:

* **difference from paper**, on the illumination-normalised image, finds prints
  whose tone differs from the album page;
* **enclosed edges** finds prints whose only distinguishing feature is a
  geometric border, which is the case for a print faded to nearly the tone of
  the paper it is mounted on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from revelai.split.refine import PreparedImage, RotRect, prepare_image, score_lines

__all__ = ["DetectParams", "PageRegion", "Candidate", "isolate_page", "find_candidates"]


@dataclass
class DetectParams:
    """Tunables for detection. Defaults come from the specification."""

    #: Photograph area as a fraction of the page area.
    min_area: float = 0.005
    max_area: float = 0.9
    #: Aspect ratios outside 1:4 to 4:1 are not photographs.
    max_aspect: float = 4.0
    #: Detection runs on a downscaled page for speed. Cropping never does.
    work_size: int = 1400
    #: Two candidates overlapping by more than this are the same object.
    suppress_iou: float = 0.30
    #: Relative deviation from the local paper tone that counts as "not paper".
    paper_deviation: float = 0.045
    #: A contour must fill this much of its own bounding rectangle to be a print.
    #:
    #: A well-found photograph measures 0.98 or better. The value is set well
    #: below that but well above the 0.73 to 0.77 that a *fragment* of a print
    #: scores, because a fragment that survives can be mistaken for one half of
    #: a pair and take a correct crop down with it.
    min_rectangularity: float = 0.82
    #: A candidate overlapping the page itself by more than this is the page.
    max_page_iou: float = 0.70
    #: How much of a candidate must lie inside another to count as contained.
    containment: float = 0.85
    #: A part of a merged candidate must be at least this fraction of it. Two
    #: touching photographs are about half the blob each; a scatter of small
    #: fragments inside one photograph is not evidence of a merge.
    min_part_fraction: float = 0.25
    #: The parts must between them account for this much of the merged blob.
    merge_coverage: float = 0.65
    #: How strong a line running through a candidate must be, relative to that
    #: candidate's own strongest border, before it is treated as the seam
    #: between two photographs mounted edge to edge.
    #:
    #: Measured over the fixtures, a single photograph's strongest internal line
    #: reaches 0.287 of its own border and a genuine seam starts at 0.320. That
    #: is a margin of about ten per cent, which is not enough to decide on
    #: alone: a horizon, a roofline or the edge of a table all produce a long
    #: straight line inside a perfectly ordinary photograph. So this never
    #: splits anything by itself. It corroborates a split that the region
    #: strategies already proposed, and otherwise it only raises a flag for
    #: review, which is what --review and --verify are for.
    seam_ratio: float = 0.30


@dataclass
class Candidate:
    """A proposed photograph, and what detection is unsure about."""

    rect: RotRect
    #: How rectangular the contour it came from was.
    rectangularity: float = 1.0
    #: Strength of the strongest line running through it, against its own
    #: borders. A high value on a candidate that was *not* split means it may be
    #: two photographs mounted edge to edge.
    seam: float = 0.0


@dataclass
class PageRegion:
    """The album page inside the photograph of it."""

    #: Filled mask of the page, at working scale.
    mask: np.ndarray
    #: The page outline as a rotated rectangle, at working scale.
    rect: RotRect
    #: Page area in working-scale pixels.
    area: float
    #: Per-pixel estimate of the album paper under the local light, BGR float32.
    illumination: np.ndarray = field(default_factory=lambda: np.zeros((1, 1, 3), np.float32))


# --------------------------------------------------------------------------
# Preparation
# --------------------------------------------------------------------------


def _to_working_scale(image: np.ndarray, work_size: int) -> tuple[np.ndarray, float]:
    """Downscale for detection. Returns the working image and its scale factor."""
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= work_size:
        return image, 1.0
    scale = work_size / float(longest)
    resized = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _as_bgr8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype == np.uint16:
        array = (array / 257.0).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    if array.shape[2] == 4:
        return array[..., :3].copy()
    return array


def _local_texture(grey: np.ndarray, size: int = 9) -> np.ndarray:
    """Local standard deviation. Album paper is flat; photographic content is not."""
    values = grey.astype(np.float32)
    mean = cv2.boxFilter(values, -1, (size, size))
    mean_sq = cv2.boxFilter(values * values, -1, (size, size))
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))


def _illumination_field(bgr: np.ndarray, page_mask: np.ndarray) -> np.ndarray:
    """Estimate the album paper as it appears under the light, per pixel.

    This runs in two stages, and the second one is what makes the threshold in
    :func:`_candidates_by_paper_difference` mean anything.

    A morphological closing removes everything darker than its structuring
    element, so a closing wider than the largest print leaves the paper behind.
    It is computed on a heavily downscaled copy, which is what makes a
    structuring element that large affordable. But a closing is an *upper
    envelope*: it tracks the brightest grain of the paper, not its typical
    level, and on the test fixtures the paper sits a good 6 per cent below its
    own envelope - further from it than the threshold that is supposed to
    separate paper from prints. Used directly, it marks half the page as a
    photograph.

    So the envelope is only used to nominate paper. Pixels that are close to it
    and locally flat are taken as album paper, and the field is refit to those
    pixels alone by normalised convolution, which interpolates the paper tone
    smoothly across the areas where paper is hidden under a print. The result
    tracks the typical paper level rather than its envelope, and paper then
    deviates from it by its own grain, a couple of per cent.

    Everything outside the page is replaced by a bright in-page value before the
    closing, otherwise the dark table bleeds into the estimate along the page
    edge and the outermost prints stop being distinguishable from it.
    """
    height, width = bgr.shape[:2]
    inside = page_mask > 0

    filled = bgr.astype(np.float32).copy()
    if inside.any():
        bright = np.percentile(bgr[inside].reshape(-1, 3).astype(np.float32), 85, axis=0)
        filled[~inside] = bright

    # Stage one: the upper envelope.
    small_long = 160
    scale = small_long / float(max(height, width))
    small = cv2.resize(
        filled,
        (max(8, round(width * scale)), max(8, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    size = max(9, int(round(0.36 * max(small.shape[:2]))) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    closed = cv2.morphologyEx(small, cv2.MORPH_CLOSE, kernel)
    closed = cv2.GaussianBlur(closed, (0, 0), max(2.0, size / 6.0))
    envelope = np.maximum(
        cv2.resize(closed, (width, height), interpolation=cv2.INTER_LINEAR).astype(np.float32),
        1.0,
    )

    # Stage two: nominate paper, then refit the field to paper alone.
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    envelope_grey = envelope.mean(axis=2)
    relative = grey / np.maximum(envelope_grey, 1.0)
    texture = _local_texture(grey)
    flat = texture <= max(3.0, float(np.percentile(texture[inside], 45))) if inside.any() else True
    seed = (relative > 0.90) & flat & inside
    if cv2.countNonZero(seed.astype(np.uint8)) < max(64, int(0.02 * max(1, inside.sum()))):
        return envelope

    return _fit_smooth_surface(bgr, seed, fallback=envelope)


def _fit_smooth_surface(bgr: np.ndarray, seed: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Fit a quadratic surface, per channel, to the nominated paper pixels.

    Blurring the seeded pixels would be the obvious way to spread the paper tone
    across the page, but a Gaussian has no support beyond the page boundary and
    therefore biases towards the middle. On a page lit from one side that is not
    a small effect: the dim edge of the paper ends up several per cent below its
    own estimate and is marked as a photograph, which welds the real prints and
    the page margin into one blob that no area or shape filter can rescue.

    A low-order polynomial extrapolates instead of averaging, so the estimate
    stays correct out to the very edge of the page. Quadratic is the right
    order: it covers a light gradient across the page and simple vignetting,
    while being far too stiff to bend around a photograph and explain it away.

    Two safeguards make the extrapolation trustworthy. The fit is repeated with
    outliers dropped, because a large flat bright area inside a photograph looks
    exactly like paper to the test that nominates seeds, and a patch of one
    tilts the whole surface. And the result is held to the range of the paper it
    was fitted to, because a quadratic asked to extrapolate is under no
    obligation to stay sensible.
    """
    height, width = bgr.shape[:2]
    ys, xs = np.nonzero(seed)
    if len(xs) < 64:
        return fallback

    # A few thousand samples describe a quadratic perfectly well.
    if len(xs) > 20000:
        step = len(xs) // 20000 + 1
        xs, ys = xs[::step], ys[::step]

    nx = xs.astype(np.float64) / max(width - 1, 1)
    ny = ys.astype(np.float64) / max(height - 1, 1)
    basis = np.stack([np.ones_like(nx), nx, ny, nx * nx, nx * ny, ny * ny], axis=1)
    samples = bgr[ys, xs].astype(np.float64)

    keep = np.ones(len(xs), dtype=bool)
    coefficients = None
    for _ in range(3):
        try:
            coefficients, *_ = np.linalg.lstsq(basis[keep], samples[keep], rcond=None)
        except np.linalg.LinAlgError:  # pragma: no cover - degenerate seeds
            return fallback
        error = np.abs(basis @ coefficients - samples).max(axis=1)
        spread = float(np.median(error[keep]))
        if spread <= 1e-6:
            break
        tightened = keep & (error <= max(4.0, 3.0 * spread))
        if tightened.sum() < 64 or tightened.sum() == keep.sum():
            break
        keep = tightened

    if coefficients is None:  # pragma: no cover - defensive
        return fallback

    residual = float(np.abs(basis[keep] @ coefficients - samples[keep]).mean())
    if not np.isfinite(residual) or residual > 12.0:
        # The seeds are not describing one smooth surface; do not trust the fit.
        return fallback

    # Evaluate by broadcasting a row and a column rather than materialising a
    # (height * width, 6) basis matrix, which for a full page is a hundred
    # million floats to build and multiply for a six term polynomial.
    gx = (np.arange(width, dtype=np.float32) / max(width - 1, 1))[None, :]
    gy = (np.arange(height, dtype=np.float32) / max(height - 1, 1))[:, None]
    terms = (np.float32(1.0), gx, gy, gx * gx, gx * gy, gy * gy)
    field_ = np.zeros((height, width, 3), np.float32)
    for term, channel_coefficients in zip(terms, coefficients, strict=True):
        field_ += np.asarray(term, np.float32)[..., None] * channel_coefficients.astype(np.float32)

    low = samples[keep].min(axis=0).astype(np.float32) * 0.85
    high = samples[keep].max(axis=0).astype(np.float32) * 1.15
    field_ = np.clip(field_, low, high)
    return np.maximum(field_, 1.0)


# --------------------------------------------------------------------------
# 8.1 Page isolation
# --------------------------------------------------------------------------


def isolate_page(image: np.ndarray, params: DetectParams | None = None) -> PageRegion:
    """Find the album page and confine everything else to it.

    The page normally sits on a dark surface. Segmenting the dominant bright
    region first is what stops the table, its edge, or the album cover from ever
    becoming a candidate.
    """
    params = params or DetectParams()
    bgr = _as_bgr8(image)
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grey, (0, 0), 2.0)

    _, bright = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = grey.shape[:2]
    mask = np.zeros((height, width), np.uint8)

    if contours:
        # The page is a rectangle, so its convex hull is the page. Filling the
        # hull rather than the contour puts the prints back inside the page:
        # they are darker than the paper, so they are holes in the bright mask.
        hull = cv2.convexHull(max(contours, key=cv2.contourArea))
        cv2.fillConvexPoly(mask, hull, 255)
        rect = RotRect.from_cv(cv2.minAreaRect(hull))
    else:
        # No bright region at all: work on the whole frame rather than refuse.
        mask[:] = 255
        rect = RotRect(width / 2.0, height / 2.0, float(width), float(height), 0.0)

    return PageRegion(
        mask=mask,
        rect=rect,
        area=float(cv2.countNonZero(mask)),
        illumination=_illumination_field(bgr, mask),
    )


# --------------------------------------------------------------------------
# 8.2 Initial candidates
# --------------------------------------------------------------------------


def _rects_from_mask(mask: np.ndarray, min_rectangularity: float) -> list[tuple[RotRect, float]]:
    """Rotated rectangles for every contour in ``mask``, at every nesting level.

    ``RETR_LIST`` rather than ``RETR_EXTERNAL`` is deliberate. A decorated frame
    printed on the album page is a closed ring enclosing the prints, and with
    ``RETR_EXTERNAL`` every photograph inside it is discarded as a child
    contour. That single detail is the difference between finding five
    photographs on a decorated page and finding one.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    rects: list[tuple[RotRect, float]] = []
    for contour in contours:
        if len(contour) < 4:
            continue
        rect = RotRect.from_cv(cv2.minAreaRect(contour))
        if rect.w < 4 or rect.h < 4:
            continue
        rectangularity = float(cv2.contourArea(contour) / max(rect.area, 1e-6))
        if rectangularity < min_rectangularity:
            continue
        rects.append((rect, rectangularity))
    return rects


def _candidates_by_paper_difference(
    bgr: np.ndarray, page: PageRegion, params: DetectParams
) -> list[tuple[RotRect, float]]:
    """Anything on the page that is not the colour of the page beneath it."""
    smoothed = cv2.GaussianBlur(bgr, (0, 0), 2.0).astype(np.float32)
    relative = smoothed / page.illumination
    deviation = np.abs(relative - 1.0).max(axis=2)

    mask = (deviation > params.paper_deviation).astype(np.uint8) * 255
    mask = cv2.bitwise_and(mask, page.mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Pull the prints apart before tracing contours. Two photographs mounted
    # edge to edge are separated only by a shadow line, and without this they
    # come back as a single blob twice the right size.
    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    return _rects_from_mask(mask, params.min_rectangularity)


def _candidates_by_enclosed_edges(
    bgr: np.ndarray, page: PageRegion, params: DetectParams
) -> list[tuple[RotRect, float]]:
    """Regions enclosed by a step in brightness, whatever their colour.

    This is the strategy that survives a print faded to nearly the tone of the
    paper, where there is almost no colour contrast and only the geometric
    border gives the print away.
    """
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grey, (0, 0), 1.5)

    inside = page.mask > 0
    if not inside.any():
        return []

    gx = cv2.Scharr(blurred, cv2.CV_32F, 1, 0) / 32.0
    gy = cv2.Scharr(blurred, cv2.CV_32F, 0, 1) / 32.0
    magnitude = cv2.magnitude(gx, gy)
    # Hysteresis: a weak but continuous border survives, isolated grain does not.
    high = max(2.5, float(np.percentile(magnitude[inside], 97)) * 0.35)
    low = high * 0.4
    edges = cv2.Canny(
        (gx * 32.0).astype(np.int16),
        (gy * 32.0).astype(np.int16),
        low * 32.0,
        high * 32.0,
        L2gradient=True,
    )
    edges = cv2.bitwise_and(edges, page.mask)
    edges = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    )
    return _rects_from_mask(edges, params.min_rectangularity)


# --------------------------------------------------------------------------
# Merging
# --------------------------------------------------------------------------


def _rect_iou(a: RotRect, b: RotRect) -> float:
    area, _ = cv2.intersectConvexConvex(
        a.corners().astype(np.float32), b.corners().astype(np.float32)
    )
    if area <= 0:
        return 0.0
    union = a.area + b.area - area
    return float(area / union) if union > 0 else 0.0


def _suppress_overlaps(
    scored: list[tuple[RotRect, float]], threshold: float
) -> list[tuple[RotRect, float]]:
    """Keep the most rectangular of any group of candidates for one object."""
    ordered = sorted(
        scored,
        key=lambda item: (
            -round(item[1], 4),
            -round(item[0].area, 3),
            item[0].cy,
            item[0].cx,
        ),
    )
    kept: list[tuple[RotRect, float]] = []
    for rect, score in ordered:
        if all(_rect_iou(rect, other) < threshold for other, _ in kept):
            kept.append((rect, score))
    return kept


def _containment(inner: RotRect, outer: RotRect) -> float:
    """Fraction of ``inner`` that lies inside ``outer``."""
    area, _ = cv2.intersectConvexConvex(
        inner.corners().astype(np.float32), outer.corners().astype(np.float32)
    )
    return float(area / inner.area) if inner.area > 0 else 0.0


def _internal_border_ratio(rect: RotRect, prepared: PreparedImage) -> float:
    """Strength of the strongest line through a candidate, against its own borders.

    Two photographs mounted edge to edge are separated only by a shadow line or
    a sliver of paper, and no region-based method can see that. A line integral
    can: the seam runs the full width of the blob and scores like a border,
    because it is two borders.

    The measurement is relative to the candidate's own strongest edge, because
    an absolute threshold would depend on the exposure of the photograph and on
    how dark the album paper is.
    """
    u, v = rect.axes
    centre = rect.centre
    internal = 0.0
    outer = 0.0
    for normal, tangent, extent, span in ((u, v, rect.w, rect.h), (v, u, rect.h, rect.w)):
        if extent < 40 or span < 20:
            continue
        outer = max(
            outer,
            float(
                score_lines(
                    prepared,
                    centre,
                    tangent,
                    normal,
                    span,
                    np.array([-extent / 2.0, extent / 2.0]),
                ).max()
            ),
        )
        # Only the middle of the candidate: a seam near the rim would leave a
        # sliver rather than a photograph.
        reach = 0.35 * extent
        offsets = np.arange(-reach, reach + 1e-6, 1.0)
        if len(offsets) >= 5:
            internal = max(
                internal,
                float(score_lines(prepared, centre, tangent, normal, span, offsets).max()),
            )
    return internal / outer if outer > 1e-6 else 0.0


def _drop_merged(
    scored: list[tuple[RotRect, float]],
    params: DetectParams,
    prepared: PreparedImage | None = None,
) -> list[tuple[RotRect, float]]:
    """Discard a candidate that is explained by two or more disjoint children.

    Two photographs mounted edge to edge, with only a shadow line between them,
    are found both as a pair and individually. The pair is a *better* rectangle
    than either print - it is larger and its outline is cleaner - so scoring by
    rectangularity alone keeps the merge and suppresses the two real crops.

    A candidate that is tiled by disjoint smaller candidates is a merge, and the
    parts are what the user wants.
    """
    survivors: list[tuple[RotRect, float]] = []
    for index, (parent, score) in enumerate(scored):
        children: list[RotRect] = []
        for other_index, (child, _) in enumerate(scored):
            if other_index == index or child.area >= parent.area * 0.85:
                continue
            if child.area < params.min_part_fraction * parent.area:
                continue
            if _containment(child, parent) >= params.containment:
                children.append(child)

        disjoint: list[RotRect] = []
        for child in sorted(children, key=lambda r: -r.area):
            if all(_rect_iou(child, kept) < 0.2 for kept in disjoint):
                disjoint.append(child)

        # Two disjoint children tiling a parent is suggestive but not proof. The
        # sky and the ground of one landscape photograph do exactly that when
        # the edge strategy traces the horizon, and dropping the parent then
        # cuts a real photograph in half. Require a seam as well: two
        # photographs have a border between them, a landscape does not.
        covered = sum(child.area for child in disjoint)
        tiled = len(disjoint) >= 2 and covered >= params.merge_coverage * parent.area
        seamed = prepared is None or _internal_border_ratio(parent, prepared) >= params.seam_ratio
        if tiled and seamed:
            continue
        survivors.append((parent, score))
    return survivors


def _drop_contained(
    scored: list[tuple[RotRect, float]], params: DetectParams
) -> list[tuple[RotRect, float]]:
    """Discard a candidate that sits wholly inside a larger surviving one.

    A strong block of colour inside a photograph traces a perfectly good
    rectangle. It is detail, not a print. This runs after :func:`_drop_merged`,
    so by the time it is reached a genuine pair of touching prints no longer has
    a merged parent to be swallowed by.
    """
    survivors: list[tuple[RotRect, float]] = []
    for index, (candidate, score) in enumerate(scored):
        contained = any(
            other_index != index
            and candidate.area < other.area * 0.85
            and _containment(candidate, other) >= params.containment
            for other_index, (other, _) in enumerate(scored)
        )
        if not contained:
            survivors.append((candidate, score))
    return survivors


def find_candidates(
    image: np.ndarray, params: DetectParams | None = None
) -> tuple[list[Candidate], PageRegion, float]:
    """Propose photograph locations on a page.

    Returns the candidates in full-resolution coordinates, the page region at
    working scale, and the working scale factor, so a caller can draw debug maps
    without repeating the work.
    """
    params = params or DetectParams()
    working, scale = _to_working_scale(_as_bgr8(image), params.work_size)
    page = isolate_page(working, params)

    scored = _candidates_by_paper_difference(working, page, params)
    scored += _candidates_by_enclosed_edges(working, page, params)

    min_area = params.min_area * page.area
    max_area = params.max_area * page.area
    viable = [
        (rect, fill)
        for rect, fill in scored
        if min_area <= rect.area <= max_area and rect.aspect <= params.max_aspect
    ]

    # 8.1, restated: a detection that *is* the page is not a photograph. This
    # is what rejects a decorated album frame, which is a closed rectangle
    # printed on the paper and otherwise a thoroughly convincing candidate.
    viable = [
        (rect, fill) for rect, fill in viable if _rect_iou(rect, page.rect) <= params.max_page_iou
    ]

    # Collapse near-duplicates first, but only near-duplicates: a merged pair
    # overlaps each of its halves by about 0.5, and must survive to this point
    # so that _drop_merged can recognise it for what it is.
    prepared = prepare_image(working)

    viable = _suppress_overlaps(viable, 0.60)
    viable = _drop_merged(viable, params, prepared)
    viable = _drop_contained(viable, params)
    kept = _suppress_overlaps(viable, params.suppress_iou)

    inverse = 1.0 / scale if scale else 1.0
    candidates = [
        Candidate(
            rect=rect.scaled(inverse),
            rectangularity=fill,
            # Measured at working scale, where the gradients were computed.
            seam=_internal_border_ratio(rect, prepared),
        )
        for rect, fill in kept
    ]
    # A stable order in, a stable order out. The pipeline re-sorts into reading
    # order once the geometry is final.
    candidates.sort(key=lambda c: (round(c.rect.cy, 3), round(c.rect.cx, 3)))
    return candidates, page, scale
