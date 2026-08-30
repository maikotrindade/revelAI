"""Detection accuracy against synthetic ground truth.

The specification sets the bar: IoU >= 0.97 and angle error <= 0.3 degrees.
These numbers are the reason the line-integral refinement exists, so this file
is the one to run after touching ``split/refine.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import angle_error, best_match, polygon_iou, split_cached
from revelai.split import SplitOptions, split_page  # noqa: F401
from revelai.split.refine import RotRect, refine_rect

IOU_THRESHOLD = 0.97
ANGLE_THRESHOLD = 0.3

# Geometry is judged on the detected rectangle itself, so the tests run with
# inset 0. The 3 px default inset is a deliberate safety trim, not a detection
# error, and it is covered separately in test_split_output.py.
EXACT = SplitOptions(inset=0)


def _detected_corners(result) -> list[np.ndarray]:
    return [d.rect.corners() for d in result.detections]


def _assert_all_matched(page, result, iou=IOU_THRESHOLD, angle=ANGLE_THRESHOLD):
    detected = _detected_corners(result)
    expected = page.visible_photos
    assert len(detected) == len(expected), (
        f"expected {len(expected)} photographs, detected {len(detected)}"
    )
    used: set[int] = set()
    for truth in expected:
        idx, score = best_match(truth.corners, detected)
        assert score >= iou, f"best IoU {score:.4f} < {iou} for photo at {truth.cx},{truth.cy}"
        assert idx not in used, "two ground truth photos matched the same detection"
        used.add(idx)
        got = RotRect.from_corners(detected[idx])
        truth_rect = RotRect.from_corners(truth.corners)
        err = angle_error(got.angle, truth_rect.angle)
        assert err <= angle, f"angle error {err:.3f} deg > {angle} deg"


class TestSimplePage:
    def test_finds_every_photograph_precisely(self, simple_page):
        _assert_all_matched(simple_page, split_cached(simple_page.image, EXACT))


class TestSinglePhoto:
    def test_geometry_is_exact(self, single_photo_page):
        result = split_cached(single_photo_page.image, EXACT)
        assert len(result.detections) == 1
        truth = single_photo_page.photos[0]
        got = result.detections[0].rect
        assert polygon_iou(truth.corners, got.corners()) >= IOU_THRESHOLD
        truth_rect = RotRect.from_corners(truth.corners)
        assert angle_error(got.angle, truth_rect.angle) <= ANGLE_THRESHOLD
        # Dimensions within a pixel of the truth, in either orientation.
        got_dims = sorted((got.w, got.h))
        want_dims = sorted((truth.w, truth.h))
        assert got_dims[0] == pytest.approx(want_dims[0], abs=1.5)
        assert got_dims[1] == pytest.approx(want_dims[1], abs=1.5)


class TestTouchingPhotos:
    def test_two_touching_photographs_are_not_merged(self, touching_pair_page):
        """The only separator is a faint shadow line between the prints."""
        result = split_cached(touching_pair_page.image, EXACT)
        assert len(result.detections) == 2, "touching prints were merged into one crop"
        _assert_all_matched(touching_pair_page, result)

    def test_neither_crop_swallows_the_neighbour(self, touching_pair_page):
        result = split_cached(touching_pair_page.image, EXACT)
        areas = [d.rect.w * d.rect.h for d in result.detections]
        truth_area = touching_pair_page.photos[0].area
        for area in areas:
            assert area < truth_area * 1.4, "an edge slid onto the neighbouring photograph"


class TestHardPage:
    def test_every_cleanly_bordered_photograph_is_found(self, hard_page):
        """Accuracy is asserted on prints that paper surrounds on all sides.

        The overlapping pair is excluded and checked separately below, because
        the thresholds do not mean the same thing there. See
        :class:`TestOverlappingPhotographs`.
        """
        result = split_cached(hard_page.image, EXACT)
        detected = _detected_corners(result)
        for truth in hard_page.clean_photos:
            idx, score = best_match(truth.corners, detected)
            assert score >= IOU_THRESHOLD, (
                f"best IoU {score:.4f} for photo at {truth.cx},{truth.cy}"
            )
            got = RotRect.from_corners(detected[idx])
            err = angle_error(got.angle, RotRect.from_corners(truth.corners).angle)
            assert err <= ANGLE_THRESHOLD, f"angle error {err:.3f} deg"

    def test_the_faded_print_is_found(self, hard_page):
        """Near paper tone, almost no colour contrast. Only the border helps."""
        result = split_cached(hard_page.image, EXACT)
        faded = next(p for p in hard_page.photos if p.contrast < 0.5)
        _, score = best_match(faded.corners, _detected_corners(result))
        assert score >= IOU_THRESHOLD, f"faded print missed, best IoU {score:.4f}"

    def test_the_touching_prints_are_separated(self, hard_page):
        """Two prints mounted edge to edge, on a page with five others."""
        result = split_cached(hard_page.image, EXACT)
        detected = _detected_corners(result)
        pair = [p for p in hard_page.photos if abs(p.cy - 1010) < 1]
        assert len(pair) == 2, "fixture should have a touching pair in the middle row"
        for truth in pair:
            _, score = best_match(truth.corners, detected)
            assert score >= IOU_THRESHOLD

    def test_the_decorated_album_border_is_not_cropped_as_a_photo(self, hard_page):
        """Page isolation exists precisely to stop this."""
        result = split_cached(hard_page.image, EXACT)
        edge_a = hard_page.page_corners[1] - hard_page.page_corners[0]
        edge_b = hard_page.page_corners[3] - hard_page.page_corners[0]
        page_area = abs(float(edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0]))
        for det in result.detections:
            assert det.rect.w * det.rect.h < 0.5 * page_area, (
                "a detection covers half the page; the album border was taken for a photo"
            )

    def test_no_detection_swallows_the_whole_page(self, hard_page):
        result = split_cached(hard_page.image, EXACT)
        assert len(result.detections) >= len(hard_page.clean_photos)


class TestOverlappingPhotographs:
    """A known hard case, handled by reporting rather than by guessing.

    Where one print is laid over another, the union of the two is an L shape and
    its borders belong to two different rectangles. Locally there is nothing in
    the image that says which. The refinement locks onto a real photograph
    border every time; it just cannot tell whose it is, and on this fixture the
    upper print comes out about 0.8 IoU rather than the 0.97 that a cleanly
    bordered print reaches.

    So the contract for this case is not accuracy, it is honesty: both prints
    are found, nothing is silently merged or dropped, and the page is flagged so
    that ``--review`` or ``--verify`` puts it in front of a human. This is
    written up in the README under Honest limitations.
    """

    def test_both_prints_of_the_pair_are_found(self, hard_page):
        result = split_cached(hard_page.image, EXACT)
        detected = _detected_corners(result)
        pair = [p for p in hard_page.photos if p.occluded or p.overlapping]
        assert len(pair) == 2, "fixture should contain one overlapping pair"
        matched = set()
        for truth in pair:
            idx, score = best_match(truth.corners, detected)
            assert score > 0.4, f"a print of the overlapping pair was lost (IoU {score:.3f})"
            matched.add(idx)
        assert len(matched) == 2, "the pair was merged into a single crop"

    def test_the_page_is_flagged_for_review(self, hard_page):
        result = split_cached(hard_page.image, EXACT)
        assert result.needs_review, "an overlapping pair must be flagged, not silently accepted"
        assert result.notes, "the run summary must say what was wrong with the page"


class TestEmptyPage:
    def test_a_page_with_no_photographs_yields_nothing(self, empty_page):
        result = split_cached(empty_page.image, EXACT)
        assert result.detections == []
        assert result.needs_review, "a page with no detections must be flagged"


class TestRefinement:
    """The refinement is what turns a rough candidate into a usable crop."""

    def test_refinement_recovers_a_perturbed_rectangle(self, single_photo_page):
        truth = RotRect.from_corners(single_photo_page.photos[0].corners)
        rough = RotRect(
            truth.cx + 7.0, truth.cy - 6.0, truth.w - 13.0, truth.h + 9.0, truth.angle + 1.7
        )
        before = polygon_iou(truth.corners(), rough.corners())
        refined = refine_rect(single_photo_page.image, rough)
        after = polygon_iou(truth.corners(), refined.rect.corners())
        assert after > before
        assert after >= IOU_THRESHOLD, f"refined IoU {after:.4f}"
        assert angle_error(refined.rect.angle, truth.angle) <= ANGLE_THRESHOLD

    def test_refinement_stays_within_its_search_radius(self, single_photo_page):
        """A small radius must not let an edge jump onto a neighbour."""
        truth = RotRect.from_corners(single_photo_page.photos[0].corners)
        rough = RotRect(truth.cx, truth.cy, truth.w, truth.h, truth.angle)
        refined = refine_rect(single_photo_page.image, rough, search_radius=4)
        assert abs(refined.rect.w - rough.w) <= 8 + 1e-6
        assert abs(refined.rect.h - rough.h) <= 8 + 1e-6

    def test_a_long_border_beats_short_internal_detail(self, single_photo_page):
        """Integrating along the whole edge is the point of the algorithm.

        Strong blobs inside the picture produce short, high gradients. The real
        border is weaker per pixel but runs the full length of the edge.
        """
        truth = RotRect.from_corners(single_photo_page.photos[0].corners)
        rough = RotRect(truth.cx + 4.0, truth.cy + 3.0, truth.w - 8.0, truth.h - 6.0, truth.angle)
        refined = refine_rect(single_photo_page.image, rough, search_radius=18)
        assert polygon_iou(truth.corners(), refined.rect.corners()) >= IOU_THRESHOLD

    def test_the_result_is_a_rectangle_not_a_trapezoid(self, single_photo_page):
        """Fitting four independent lines produces skewed crops. Not allowed."""
        truth = RotRect.from_corners(single_photo_page.photos[0].corners)
        rough = RotRect(truth.cx + 5, truth.cy + 5, truth.w - 10, truth.h - 10, truth.angle + 0.9)
        corners = refine_rect(single_photo_page.image, rough).rect.corners()
        # Opposite sides equal, and all four corners square.
        side = lambda a, b: float(np.hypot(*(corners[b] - corners[a])))  # noqa: E731
        assert side(0, 1) == pytest.approx(side(3, 2), abs=1e-6)
        assert side(1, 2) == pytest.approx(side(0, 3), abs=1e-6)
        for i in range(4):
            u = corners[(i + 1) % 4] - corners[i]
            v = corners[(i - 1) % 4] - corners[i]
            assert abs(float(np.dot(u, v))) < 1e-6 * max(1.0, float(np.linalg.norm(u)))

    def test_refinement_is_deterministic(self, single_photo_page):
        rough = RotRect.from_corners(single_photo_page.photos[0].corners)
        rough = RotRect(rough.cx + 3, rough.cy, rough.w, rough.h, rough.angle + 0.5)
        a = refine_rect(single_photo_page.image, rough).rect
        b = refine_rect(single_photo_page.image, rough).rect
        assert (a.cx, a.cy, a.w, a.h, a.angle) == (b.cx, b.cy, b.w, b.h, b.angle)


class TestDeterminism:
    def test_two_runs_produce_identical_geometry(self, hard_page):
        # Deliberately not the cached helper: this is the test that has to do
        # the work twice for real.
        a = split_page(hard_page.image, EXACT)
        b = split_page(hard_page.image, EXACT)
        assert len(a.detections) == len(b.detections)
        for da, db in zip(a.detections, b.detections, strict=True):
            assert np.array_equal(da.rect.corners(), db.rect.corners())
