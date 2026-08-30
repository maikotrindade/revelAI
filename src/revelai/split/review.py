"""Interactive review of a page before its crops are saved (specification 9).

Detection gets things wrong. That is an observed fact rather than a hypothesis,
and it is why this exists: a page can be corrected by hand in a few seconds,
which is much cheaper than discovering a bad crop after fifty pages.

Controls
    drag inside a box     move it
    drag a corner         resize it
    drag on empty paper   create a box
    d                     delete the selected box
    r / R                 rotate the selected box by 0.25 degrees
    f                     re-run the refinement on the selected box
    Enter                 accept the page
    s                     skip the page, saving nothing from it
    q                     quit the run

Every manual edit is put back through the refinement before it is cropped, so a
hand-drawn box still gets sub-pixel borders and a deskew.

On a machine with no display this degrades to automatic mode with a warning. It
does not crash, and it does not silently drop the page.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from revelai.split import Detection, PageResult, SplitOptions
from revelai.split.crop import inset_corners
from revelai.split.refine import PreparedImage, RotRect, prepare_image, refine_rect

__all__ = ["ReviewOutcome", "ReviewSession", "review_page", "display_available"]

_HANDLE_RADIUS = 14
_ROTATE_STEP = 0.25
_MIN_SIDE = 12.0

_ACCEPT_KEYS = (13, 10)  # Enter, on the platforms that disagree about which
_WINDOW = "RevelAI review"


@dataclass
class ReviewOutcome:
    """What the user decided about one page."""

    result: PageResult
    accepted: bool = True
    quit_run: bool = False


def display_available() -> bool:
    """Whether OpenCV can actually open a window here.

    Checked by opening one, because the alternatives all lie: a build can have
    highgui compiled in and still fail on a headless machine, and DISPLAY says
    nothing about Wayland or macOS.
    """
    try:
        cv2.namedWindow("__revelai_probe__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__revelai_probe__")
        cv2.waitKey(1)
    except cv2.error:
        return False
    except Exception:  # pragma: no cover - defensive
        return False
    return True


class ReviewSession:
    """One page on screen, with its boxes editable.

    The geometry is kept in full-resolution page coordinates and drawn through a
    display scale, so an edit made on a downscaled view still lands on the real
    pixels.
    """

    def __init__(
        self,
        image: np.ndarray,
        result: PageResult,
        options: SplitOptions,
        prepared: PreparedImage | None = None,
        max_display: int = 1000,
    ) -> None:
        self.image = image
        self.options = options
        self.prepared = prepared if prepared is not None else prepare_image(image)
        self.rects: list[RotRect] = [d.rect for d in result.detections]
        self.flags: list[list[str]] = [list(d.flags) for d in result.detections]
        self.result = result

        height, width = image.shape[:2]
        self.scale = min(1.0, max_display / float(max(height, width)))
        self.selected: int | None = None
        self._drag: tuple[str, int, np.ndarray] | None = None
        self._new_origin: np.ndarray | None = None
        self._cursor = np.zeros(2)

    # -- coordinate helpers ------------------------------------------------

    def to_view(self, point: np.ndarray) -> np.ndarray:
        return np.asarray(point, dtype=np.float64) * self.scale

    def to_page(self, x: float, y: float) -> np.ndarray:
        return np.array([x, y], dtype=np.float64) / self.scale

    # -- hit testing -------------------------------------------------------

    def _corner_at(self, point: np.ndarray) -> tuple[int, int] | None:
        radius = _HANDLE_RADIUS / max(self.scale, 1e-6)
        for index, rect in enumerate(self.rects):
            for corner_index, corner in enumerate(rect.corners()):
                if float(np.hypot(*(corner - point))) <= radius:
                    return index, corner_index
        return None

    def _box_at(self, point: np.ndarray) -> int | None:
        # Later boxes are drawn on top, so they are hit first.
        for index in reversed(range(len(self.rects))):
            contour = self.rects[index].corners().astype(np.float32)
            if cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0:
                return index
        return None

    # -- editing -----------------------------------------------------------

    def refine(self, index: int) -> None:
        """Put a box back through the refinement.

        Called after every manual edit. A box dragged by hand is approximately
        right; the refinement is what makes it exact.
        """
        result = refine_rect(
            self.prepared,
            self.rects[index],
            search_radius=self.options.search_radius,
            angle_span=self.options.angle_span,
            angle_step=self.options.angle_step,
        )
        self.rects[index] = result.rect
        self.flags[index] = []

    def delete(self, index: int) -> None:
        del self.rects[index]
        del self.flags[index]
        self.selected = None

    def rotate(self, index: int, degrees: float) -> None:
        rect = self.rects[index]
        self.rects[index] = RotRect(rect.cx, rect.cy, rect.w, rect.h, rect.angle + degrees)

    def _resize_from_corner(self, index: int, corner_index: int, point: np.ndarray) -> None:
        """Move one corner, keeping the opposite corner and the angle fixed."""
        rect = self.rects[index]
        corners = rect.corners()
        anchor = corners[(corner_index + 2) % 4]
        u, v = rect.axes
        delta = point - anchor
        width = abs(float(np.dot(delta, u)))
        height = abs(float(np.dot(delta, v)))
        if width < _MIN_SIDE or height < _MIN_SIDE:
            return
        centre = (anchor + point) / 2.0
        self.rects[index] = RotRect(centre[0], centre[1], width, height, rect.angle)

    # -- mouse -------------------------------------------------------------

    def on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        point = self.to_page(x, y)
        self._cursor = point

        if event == cv2.EVENT_LBUTTONDOWN:
            hit = self._corner_at(point)
            if hit is not None:
                index, corner_index = hit
                self.selected = index
                self._drag = ("corner", index, np.array([corner_index, 0]))
                return
            index = self._box_at(point)
            if index is not None:
                self.selected = index
                self._drag = ("move", index, point - self.rects[index].centre)
                return
            self._new_origin = point
            self.selected = None

        elif event == cv2.EVENT_MOUSEMOVE:
            if self._drag is not None:
                kind, index, payload = self._drag
                if kind == "move":
                    centre = point - payload
                    rect = self.rects[index]
                    self.rects[index] = RotRect(centre[0], centre[1], rect.w, rect.h, rect.angle)
                elif kind == "corner":
                    self._resize_from_corner(index, int(payload[0]), point)

        elif event == cv2.EVENT_LBUTTONUP:
            if self._drag is not None:
                _, index, _ = self._drag
                self._drag = None
                self.refine(index)
            elif self._new_origin is not None:
                origin = self._new_origin
                self._new_origin = None
                width = abs(float(point[0] - origin[0]))
                height = abs(float(point[1] - origin[1]))
                if width >= _MIN_SIDE and height >= _MIN_SIDE:
                    centre = (origin + point) / 2.0
                    self.rects.append(RotRect(centre[0], centre[1], width, height, 0.0))
                    self.flags.append([])
                    self.selected = len(self.rects) - 1
                    self.refine(self.selected)

    # -- drawing -----------------------------------------------------------

    def render(self) -> np.ndarray:
        canvas = self.image
        if canvas.dtype == np.uint16:
            canvas = (canvas / 257.0).astype(np.uint8)
        if canvas.ndim == 2:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        view = cv2.resize(
            canvas,
            (
                max(1, round(canvas.shape[1] * self.scale)),
                max(1, round(canvas.shape[0] * self.scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )

        for index, rect in enumerate(self.rects):
            corners = self.to_view(rect.corners()).astype(np.int32)
            chosen = index == self.selected
            colour = (
                (0, 220, 255) if chosen else ((0, 0, 255) if self.flags[index] else (0, 220, 0))
            )
            cv2.polylines(view, [corners], True, colour, 2 if not chosen else 3)
            for corner in corners:
                cv2.circle(view, tuple(corner), 5, colour, -1)
            cv2.putText(
                view,
                str(index + 1),
                tuple(corners[0] + np.array([6, 22])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                colour,
                2,
            )

        if self._new_origin is not None:
            start = self.to_view(self._new_origin).astype(int)
            end = self.to_view(self._cursor).astype(int)
            cv2.rectangle(view, tuple(start), tuple(end), (255, 200, 0), 2)

        banner = (
            f"{len(self.rects)} photo(s)   "
            "drag: move/resize   d: delete   r/R: rotate   f: refine   "
            "Enter: accept   s: skip   q: quit"
        )
        cv2.rectangle(view, (0, 0), (view.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(view, banner, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return view

    # -- result ------------------------------------------------------------

    def to_result(self) -> PageResult:
        """Rebuild a page result from the edited boxes."""
        from revelai.naming import order_on_page

        order = order_on_page(self.rects)
        detections = [
            Detection(
                rect=self.rects[i],
                crop_corners=inset_corners(self.rects[i].corners(), self.options.inset),
                edge_score=0.0,
                peak_width=0.0,
                flags=list(self.flags[i]),
            )
            for i in order
        ]
        return PageResult(
            detections=detections,
            page=self.result.page,
            scale=self.result.scale,
            notes=["reviewed by hand"],
            needs_review=False,
        )


def review_page(
    image: np.ndarray,
    result: PageResult,
    options: SplitOptions,
    *,
    title: str = "",
    prepared: PreparedImage | None = None,
) -> ReviewOutcome:
    """Show one page for review and return what the user decided."""
    if not display_available():
        # Falling back is the right answer here. Refusing to run would make
        # --review unusable over ssh, and crashing would lose the whole batch.
        print(
            "warning: no display available, so --review cannot open a window. "
            "Continuing in automatic mode."
        )
        return ReviewOutcome(result=result, accepted=True)

    session = ReviewSession(image, result, options, prepared=prepared)
    window = f"{_WINDOW} - {title}" if title else _WINDOW
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, session.on_mouse)
    try:
        while True:
            cv2.imshow(window, session.render())
            key = cv2.waitKey(20) & 0xFF
            if key == 255:
                continue
            if key in _ACCEPT_KEYS:
                return ReviewOutcome(result=session.to_result(), accepted=True)
            if key == ord("s"):
                return ReviewOutcome(
                    result=PageResult(detections=[], notes=["skipped"]), accepted=False
                )
            if key == ord("q"):
                return ReviewOutcome(result=session.to_result(), accepted=False, quit_run=True)
            if session.selected is None:
                continue
            if key == ord("d"):
                session.delete(session.selected)
            elif key == ord("r"):
                session.rotate(session.selected, _ROTATE_STEP)
            elif key == ord("R"):
                session.rotate(session.selected, -_ROTATE_STEP)
            elif key == ord("f"):
                session.refine(session.selected)
    finally:
        cv2.destroyWindow(window)
        cv2.waitKey(1)
