"""Interactive review, exercised without opening a window.

The window loop needs a display, so these tests drive :class:`ReviewSession`
directly with the same events OpenCV would deliver. What matters is that an edit
made by hand ends up refined, and that a headless machine degrades to automatic
mode instead of crashing.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

import synth
from conftest import polygon_iou
from revelai.split import SplitOptions, split_page
from revelai.split.refine import RotRect
from revelai.split.review import ReviewOutcome, ReviewSession, review_page

OPTIONS = SplitOptions(inset=3)


@pytest.fixture(scope="module")
def page():
    return synth.tiny_page(seed=80, photos=2)


@pytest.fixture(scope="module")
def result(page):
    return split_page(page.image, OPTIONS)


@pytest.fixture
def session(page, result):
    return ReviewSession(page.image, result, OPTIONS)


def _view(session, point):
    scaled = session.to_view(np.asarray(point, dtype=float))
    return int(round(scaled[0])), int(round(scaled[1]))


class TestSelection:
    def test_clicking_inside_a_box_selects_it(self, session):
        rect = session.rects[0]
        session.on_mouse(cv2.EVENT_LBUTTONDOWN, *_view(session, rect.centre), 0, None)
        assert session.selected == 0

    def test_clicking_empty_paper_selects_nothing(self, session):
        session.on_mouse(cv2.EVENT_LBUTTONDOWN, 2, 2, 0, None)
        assert session.selected is None


class TestEditing:
    def test_moving_a_box_snaps_it_back_onto_the_print(self, session, page):
        """Every manual edit is refined before it is cropped."""
        before = session.rects[0]
        start = _view(session, before.centre)
        end = _view(session, before.centre + np.array([9.0, 7.0]))
        session.on_mouse(cv2.EVENT_LBUTTONDOWN, *start, 0, None)
        session.on_mouse(cv2.EVENT_MOUSEMOVE, *end, 0, None)
        session.on_mouse(cv2.EVENT_LBUTTONUP, *end, 0, None)
        after = session.rects[0]
        assert abs(after.cx - before.cx) < 2.0
        assert abs(after.cy - before.cy) < 2.0

    def test_a_hand_drawn_box_is_refined_onto_the_photograph(self, page, result):
        session = ReviewSession(page.image, result, OPTIONS)
        truth = page.clean_photos[0]
        rough = RotRect.from_corners(truth.corners)
        session.delete(0)
        before = len(session.rects)
        # Drag a rough axis-aligned box a few pixels inside the real print.
        corners = rough.corners()
        start = _view(session, corners[0] + np.array([6.0, 6.0]))
        end = _view(session, corners[2] - np.array([6.0, 6.0]))
        session.on_mouse(cv2.EVENT_LBUTTONDOWN, *start, 0, None)
        session.on_mouse(cv2.EVENT_MOUSEMOVE, *end, 0, None)
        session.on_mouse(cv2.EVENT_LBUTTONUP, *end, 0, None)
        assert len(session.rects) == before + 1
        drawn = session.rects[-1]
        assert polygon_iou(truth.corners, drawn.corners()) > 0.95, (
            "a hand-drawn box must be refined onto the real borders before cropping"
        )

    def test_delete_removes_a_box(self, page, result):
        session = ReviewSession(page.image, result, OPTIONS)
        before = len(session.rects)
        session.delete(0)
        assert len(session.rects) == before - 1
        assert session.selected is None

    def test_rotation_is_in_quarter_degree_steps(self, page, result):
        session = ReviewSession(page.image, result, OPTIONS)
        before = session.rects[0].angle
        session.rotate(0, 0.25)
        assert session.rects[0].angle == pytest.approx(before + 0.25)
        session.rotate(0, -0.25)
        assert session.rects[0].angle == pytest.approx(before)

    def test_refine_on_demand_keeps_a_correct_box_correct(self, page, result):
        session = ReviewSession(page.image, result, OPTIONS)
        before = session.rects[0]
        session.refine(0)
        assert polygon_iou(before.corners(), session.rects[0].corners()) > 0.99

    def test_a_tiny_drag_does_not_create_a_box(self, session):
        before = len(session.rects)
        session.on_mouse(cv2.EVENT_LBUTTONDOWN, 3, 3, 0, None)
        session.on_mouse(cv2.EVENT_LBUTTONUP, 5, 5, 0, None)
        assert len(session.rects) == before


class TestResult:
    def test_the_edited_page_comes_back_in_reading_order(self, page, result):
        session = ReviewSession(page.image, result, OPTIONS)
        rebuilt = session.to_result()
        assert len(rebuilt.detections) == len(session.rects)
        centres = [d.rect.cx for d in rebuilt.detections]
        assert centres == sorted(centres) or len(centres) < 2

    def test_a_reviewed_page_no_longer_needs_review(self, page, result):
        rebuilt = ReviewSession(page.image, result, OPTIONS).to_result()
        assert rebuilt.needs_review is False
        assert rebuilt.notes == ["reviewed by hand"]

    def test_the_inset_is_applied_to_the_edited_boxes(self, page, result):
        session = ReviewSession(page.image, result, OPTIONS)
        detection = session.to_result().detections[0]
        outer = RotRect.from_corners(detection.rect.corners())
        inner = RotRect.from_corners(detection.crop_corners)
        assert outer.w - inner.w == pytest.approx(2 * OPTIONS.inset, abs=0.01)


class TestRendering:
    def test_the_page_renders_with_its_boxes(self, session):
        view = session.render()
        assert view.ndim == 3 and view.shape[2] == 3
        assert view.shape[0] <= 1000 and view.shape[1] <= 1000

    def test_a_sixteen_bit_page_renders(self, page, result):
        deep = page.image.astype(np.uint16) * 257
        assert ReviewSession(deep, result, OPTIONS).render().dtype == np.uint8


class TestHeadless:
    def test_it_falls_back_to_automatic_mode_without_a_display(
        self, page, result, monkeypatch, capsys
    ):
        """On a headless machine this warns and continues. It does not crash."""
        monkeypatch.setattr("revelai.split.review.display_available", lambda: False)
        outcome = review_page(page.image, result, OPTIONS)
        assert isinstance(outcome, ReviewOutcome)
        assert outcome.accepted
        assert outcome.result is result, "the automatic result must be kept as it is"
        assert "no display available" in capsys.readouterr().out
