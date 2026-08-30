"""Synthetic album pages with exact ground truth.

Tests must never depend on photographs of real people, so every fixture used by
the suite is generated here. A page is built in three steps:

1. a dark "table" background, because a phone photograph of an album is not a
   flatbed scan and the background is the darkest thing in the frame;
2. an album page of noisy, slightly uneven paper, optionally with a decorated
   border like the ones that fooled naive detectors;
3. rectangular "photographs" pasted at known centres, sizes and angles.

The page is then placed on the table with its own small rotation. Every ground
truth corner is carried through the same affine transform analytically, so the
returned coordinates are exact rather than measured back off the pixels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

__all__ = [
    "PhotoSpec",
    "SyntheticPage",
    "build_page",
    "simple_page",
    "hard_page",
    "empty_page",
    "single_photo_page",
    "tiny_page",
    "touching_pair_page",
    "yellow_cast_image",
    "write_pages",
]


@dataclass
class PhotoSpec:
    """A photograph to paste, in page coordinates, plus its ground truth."""

    cx: float
    cy: float
    w: float
    h: float
    angle: float = 0.0
    #: Base BGR tone. A faded print sits close to the paper tone on purpose.
    tone: tuple[int, int, int] = (120, 100, 90)
    #: Content contrast. Low values simulate a washed out print.
    contrast: float = 1.0
    #: True when another photograph is pasted on top of this one afterwards.
    occluded: bool = False
    #: True when this photograph is pasted on top of another one. Its own
    #: rectangle is fully visible, but it shares borders with the print
    #: underneath rather than with paper, which is a genuinely ambiguous case.
    overlapping: bool = False
    #: Filled in by build_page: the four corners in final image coordinates.
    corners: np.ndarray | None = field(default=None, repr=False)

    @property
    def area(self) -> float:
        return float(self.w * self.h)


@dataclass
class SyntheticPage:
    """A rendered page and the ground truth of everything on it."""

    image: np.ndarray
    photos: list[PhotoSpec]
    page_corners: np.ndarray

    @property
    def visible_photos(self) -> list[PhotoSpec]:
        """Photographs whose full rectangle is actually visible."""
        return [p for p in self.photos if not p.occluded]

    @property
    def clean_photos(self) -> list[PhotoSpec]:
        """Photographs bordered by paper on all four sides.

        These are the ones the accuracy thresholds apply to. A print that
        overlaps another shares two of its borders with the print underneath,
        and locally there is nothing to say which of the two rectangles such a
        border belongs to; see the note in ``test_split_geometry.py``.
        """
        return [p for p in self.photos if not p.occluded and not p.overlapping]


def _rect_corners(cx: float, cy: float, w: float, h: float, angle: float) -> np.ndarray:
    """Four corners of a rotated rectangle, clockwise from top-left."""
    rad = math.radians(angle)
    ux, uy = math.cos(rad), math.sin(rad)
    vx, vy = -math.sin(rad), math.cos(rad)
    hw, hh = w / 2.0, h / 2.0
    return np.array(
        [
            [cx - ux * hw - vx * hh, cy - uy * hw - vy * hh],
            [cx + ux * hw - vx * hh, cy + uy * hw - vy * hh],
            [cx + ux * hw + vx * hh, cy + uy * hw + vy * hh],
            [cx - ux * hw + vx * hh, cy - uy * hw + vy * hh],
        ],
        dtype=np.float64,
    )


def _photo_content(
    w: int, h: int, tone: tuple[int, int, int], contrast: float, rng: np.random.Generator
) -> np.ndarray:
    """Plausible photographic content: soft blobs, never a flat colour.

    A flat rectangle would make detection unrealistically easy, and would hide
    the failure mode where an edge slides onto a strong detail inside the
    picture.
    """
    img = np.zeros((h, w, 3), np.float32)
    img[:] = tone
    for _ in range(max(12, (w * h) // 6000)):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        r = int(rng.integers(min(w, h) // 12 + 2, min(w, h) // 3 + 3))
        colour = rng.integers(0, 256, 3).astype(np.float32)
        cv2.circle(img, (x, y), r, tuple(float(c) for c in colour), -1)
    img = cv2.GaussianBlur(img, (0, 0), max(1.5, min(w, h) / 60.0))
    # Pull content towards its own mean to simulate a faded print.
    mean = img.mean(axis=(0, 1), keepdims=True)
    img = mean + (img - mean) * contrast
    img += rng.standard_normal(img.shape, dtype=np.float32) * 2.0
    return np.clip(img, 0, 255).astype(np.uint8)


def build_page(
    specs: list[PhotoSpec],
    *,
    page_size: tuple[int, int] = (1500, 2100),
    canvas_size: tuple[int, int] = (1800, 2400),
    page_angle: float = 1.5,
    page_offset: tuple[float, float] = (150.0, 150.0),
    decorated_border: bool = False,
    uneven_light: bool = True,
    paper_value: int = 232,
    seed: int = 0,
) -> SyntheticPage:
    """Render a page and return it with exact ground truth corners."""
    rng = np.random.default_rng(seed)
    pw, ph = page_size
    cw, ch = canvas_size

    # 1. The table. Dark, slightly noisy, matte.
    canvas = np.full((ch, cw, 3), 38.0, np.float32)
    canvas += rng.standard_normal(canvas.shape, dtype=np.float32) * 3.0

    # 2. The paper. Bright, low saturation, gently textured.
    page = np.full((ph, pw, 3), float(paper_value), np.float32)
    page += rng.standard_normal(page.shape, dtype=np.float32) * 2.5
    grain = cv2.GaussianBlur(rng.standard_normal((ph, pw), dtype=np.float32) * 6.0, (0, 0), 9.0)
    page += grain[..., None]
    if decorated_border:
        # The kind of printed frame that a naive detector mistakes for the top
        # edge of the photograph sitting just below it.
        cv2.rectangle(page, (26, 26), (pw - 26, ph - 26), (196.0, 188.0, 172.0), 6)
        cv2.rectangle(page, (44, 44), (pw - 44, ph - 44), (208.0, 201.0, 186.0), 2)

    # 3. The photographs, in the order given, so later ones occlude earlier ones.
    for spec in specs:
        w, h = int(round(spec.w)), int(round(spec.h))
        content = _photo_content(w, h, spec.tone, spec.contrast, rng)
        # OpenCV's rotation matrix turns the opposite way to the (cos, sin)
        # convention used by _rect_corners and by RotRect, so the angle is
        # negated here. Without this the rendered pixels sit at -angle while the
        # ground truth claims +angle, and every geometry test is measured
        # against a rectangle that is not the one on the page.
        rot = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -spec.angle, 1.0)
        rot[0, 2] += spec.cx - w / 2.0
        rot[1, 2] += spec.cy - h / 2.0
        warped = cv2.warpAffine(content, rot, (pw, ph), flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(np.full((h, w), 255, np.uint8), rot, (pw, ph))
        page[mask > 127] = warped[mask > 127]

    # 4. Place the page on the table.
    place = cv2.getRotationMatrix2D((pw / 2.0, ph / 2.0), page_angle, 1.0)
    place[0, 2] += page_offset[0]
    place[1, 2] += page_offset[1]
    page_u8 = np.clip(page, 0, 255).astype(np.uint8)
    warped_page = cv2.warpAffine(page_u8, place, (cw, ch), flags=cv2.INTER_LINEAR)
    page_mask = cv2.warpAffine(np.full((ph, pw), 255, np.uint8), place, (cw, ch))
    canvas_u8 = np.clip(canvas, 0, 255).astype(np.uint8)
    canvas_u8[page_mask > 127] = warped_page[page_mask > 127]

    if uneven_light:
        # Window light from one side; never flat, never a hard shadow.
        across = (0.26 * np.arange(cw, dtype=np.float32) / cw)[None, :]
        down = (0.82 + 0.10 * (1.0 - np.arange(ch, dtype=np.float32) / ch))[:, None]
        gain = across + down
        canvas_u8 = np.clip(canvas_u8 * gain[..., None], 0, 255).astype(np.uint8)

    # 5. Carry every ground truth corner through the same placement transform.
    def to_image(pts: np.ndarray) -> np.ndarray:
        homo = np.hstack([pts, np.ones((len(pts), 1))])
        return homo @ place.T

    out_specs: list[PhotoSpec] = []
    for spec in specs:
        placed = PhotoSpec(**{k: v for k, v in vars(spec).items() if k != "corners"})
        placed.corners = to_image(_rect_corners(spec.cx, spec.cy, spec.w, spec.h, spec.angle))
        out_specs.append(placed)
    page_corners = to_image(_rect_corners(pw / 2.0, ph / 2.0, pw, ph, 0.0))

    return SyntheticPage(image=canvas_u8, photos=out_specs, page_corners=page_corners)


# --------------------------------------------------------------------------
# Named fixtures
# --------------------------------------------------------------------------


def simple_page(seed: int = 0) -> SyntheticPage:
    """Four well separated photographs at mild angles. The easy case."""
    specs = [
        PhotoSpec(420, 400, 620, 460, -2.5, (120, 90, 70)),
        PhotoSpec(1090, 420, 560, 470, 1.8, (80, 110, 140)),
        PhotoSpec(430, 1100, 600, 520, 0.7, (95, 120, 105)),
        PhotoSpec(1060, 1120, 580, 500, -1.4, (140, 100, 110)),
    ]
    return build_page(specs, seed=seed)


def touching_pair_page(seed: int = 1) -> SyntheticPage:
    """Two photographs with no paper between them.

    The only separator is a faint shadow line, so the refinement search radius
    has to be small enough not to jump onto the neighbour.
    """
    specs = [
        PhotoSpec(500, 700, 620, 480, 0.0, (110, 85, 65)),
        PhotoSpec(1122, 700, 620, 480, 0.0, (70, 105, 135)),
    ]
    return build_page(specs, seed=seed)


def single_photo_page(seed: int = 2) -> SyntheticPage:
    """One photograph, used for exact geometry assertions."""
    return build_page([PhotoSpec(750, 1050, 900, 640, 2.2, (105, 95, 120))], seed=seed)


def tiny_page(seed: int = 6, photos: int = 2) -> SyntheticPage:
    """A small page, for tests about naming, ordering and file output.

    Those tests care about how many files appear and what they are called, not
    about sub-pixel geometry, and a full size page costs several seconds to
    build and split. This one is a fifth of the area, in a fixed two-column
    layout so that reading order is unambiguous.
    """
    tones = [(110, 90, 70), (70, 110, 140), (95, 125, 100), (140, 100, 110)]
    angles = [-1.4, 1.2, 0.8, -0.9]
    specs = [
        PhotoSpec(
            cx=150 + 220 * (index % 2),
            cy=140 + 200 * (index // 2),
            w=180,
            h=140,
            angle=angles[index % len(angles)],
            tone=tones[index % len(tones)],
        )
        for index in range(photos)
    ]
    return build_page(
        specs,
        page_size=(520, 700),
        canvas_size=(620, 820),
        page_offset=(50.0, 50.0),
        seed=seed,
    )


def empty_page(seed: int = 3) -> SyntheticPage:
    """A page with nothing on it. Must not raise and must not invent a crop."""
    return build_page([], decorated_border=True, seed=seed)


def hard_page(seed: int = 4) -> SyntheticPage:
    """Everything that goes wrong on a real page, on one page.

    Two touching photographs, a faded print whose tone is close to the paper,
    an overlapping pair, and a decorated album border above the top row.
    """
    specs = [
        PhotoSpec(430, 400, 620, 450, -2.2, (120, 90, 70)),
        PhotoSpec(1080, 410, 560, 450, 1.6, (80, 110, 140)),
        # Faded: near paper tone, very low contrast. Only the border saves you.
        PhotoSpec(420, 1010, 600, 500, 0.6, (214, 208, 196), contrast=0.22),
        # Touching the faded one on its right edge.
        PhotoSpec(1024, 1010, 600, 500, 0.6, (95, 125, 100)),
        # Overlapping pair: the second is pasted on top of the first.
        PhotoSpec(620, 1650, 700, 500, -1.1, (140, 100, 110), occluded=True),
        PhotoSpec(1090, 1720, 560, 430, 3.0, (90, 90, 140), overlapping=True),
    ]
    return build_page(specs, decorated_border=True, seed=seed)


def yellow_cast_image(
    size: tuple[int, int] = (256, 256), seed: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """A neutral image and the same image under a strong yellow cast.

    Returns ``(neutral_bgr, cast_bgr)``. Classical correction has to bring the
    cast version back towards the neutral one.
    """
    rng = np.random.default_rng(seed)
    w, h = size
    neutral = np.zeros((h, w, 3), np.float32)
    for _ in range(40):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(10, 60))
        grey = float(rng.integers(20, 236))
        cv2.circle(neutral, (x, y), r, (grey, grey, grey), -1)
    neutral = cv2.GaussianBlur(neutral, (0, 0), 3.0)
    # Anchor the black and white points so percentile clipping has something
    # honest to lock onto.
    neutral[:6, :6] = 4.0
    neutral[-6:, -6:] = 251.0
    neutral = np.clip(neutral, 0, 255).astype(np.uint8)

    # Yellow cast: blue channel loses the most, red gains. Typical of an aged
    # print where the cyan dye layer has faded.
    gains = np.array([0.62, 0.94, 1.12])  # BGR
    cast = np.clip(neutral.astype(np.float64) * gains + np.array([6.0, 10.0, 18.0]), 0, 255)
    return neutral, cast.astype(np.uint8)


def write_pages(directory, pages: dict[str, SyntheticPage]) -> list:
    """Write pages as lossless PNGs, returning the paths in sorted order."""
    from pathlib import Path

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, page in pages.items():
        path = directory / name
        if not cv2.imwrite(str(path), page.image):
            raise RuntimeError(f"could not write synthetic page to {path}")
        written.append(path)
    return sorted(written)
