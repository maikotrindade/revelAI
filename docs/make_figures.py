"""Generate the illustrations used in the README.

Every image here is synthetic. RevelAI never commits photographs of real people,
so the "photographs" are generated scenes from ``tests/synth.py``: a sky, a
horizon, a sun and a few silhouettes. No faces, nobody real. They are enough to
show what the tool does to a page without putting anybody's family album on the
internet.

Run from the repository root:

    python docs/make_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "src"))

import synth  # noqa: E402
from revelai.enhance.color import correct_colour_cast  # noqa: E402
from revelai.split import SplitOptions, crop_detection, split_page, write_debug_map  # noqa: E402

OUT = ROOT / "docs" / "img"
INK = (38, 38, 38)
PAPER = (250, 250, 250)


def label(canvas: np.ndarray, text: str, origin: tuple[int, int], scale: float = 0.7) -> None:
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, INK, 2, cv2.LINE_AA)


def fit(image: np.ndarray, width: int) -> np.ndarray:
    height = max(1, round(image.shape[0] * width / image.shape[1]))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def figure_split(page: synth.SyntheticPage, crops: list[np.ndarray]) -> None:
    """One photograph of a page in, four separated photographs out."""
    left = fit(page.image, 460)
    cell_w, cell_h, gap = 250, 200, 14
    grid_w, grid_h = cell_w * 2 + gap, cell_h * 2 + gap
    grid = np.full((grid_h, grid_w, 3), PAPER, np.uint8)
    for index, crop in enumerate(crops[:4]):
        thumb = fit(crop, cell_w)[:cell_h]
        y = (index // 2) * (cell_h + gap)
        x = (index % 2) * (cell_w + gap)
        grid[y : y + thumb.shape[0], x : x + thumb.shape[1]] = thumb

    pad = 14
    height = max(left.shape[0], grid_h) + 46 + pad
    canvas = np.full((height, left.shape[1] + 90 + grid_w + pad * 2, 3), PAPER, np.uint8)
    canvas[40 : 40 + left.shape[0], pad : pad + left.shape[1]] = left
    grid_x = pad + left.shape[1] + 90
    top = 40 + max(0, (left.shape[0] - grid_h) // 2)
    canvas[top : top + grid_h, grid_x : grid_x + grid_w] = grid

    label(canvas, "one photograph of a page", (pad, 26))
    label(canvas, "four photographs out, deskewed", (grid_x, 26))
    arrow_y = 40 + left.shape[0] // 2
    cv2.arrowedLine(
        canvas, (grid_x - 70, arrow_y), (grid_x - 18, arrow_y), INK, 3, cv2.LINE_AA, tipLength=0.3
    )
    cv2.imwrite(str(OUT / "split-example.png"), canvas)


def figure_restore(crop: np.ndarray) -> None:
    """Classical colour correction on a print that has yellowed."""
    faded = np.clip(
        crop.astype(np.float32) * np.array([0.58, 0.92, 1.14]) + np.array([10, 14, 26]), 0, 255
    ).astype(np.uint8)
    fixed = correct_colour_cast(faded)

    panel, pad = 330, 14
    before, after = fit(faded, panel), fit(fixed, panel)
    height = max(before.shape[0], after.shape[0]) + 46 + pad
    canvas = np.full((height, panel * 2 + 24 + pad * 2, 3), PAPER, np.uint8)
    canvas[40 : 40 + before.shape[0], pad : pad + panel] = before
    canvas[40 : 40 + after.shape[0], pad + panel + 24 : pad + panel * 2 + 24] = after
    label(canvas, "before", (pad, 26))
    label(canvas, "after: classical, no model", (pad + panel + 24, 26))
    cv2.imwrite(str(OUT / "restore-example.png"), canvas)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    page = synth.scenic_page()
    cv2.imwrite(str(OUT / "page-example.png"), fit(page.image, 700))

    result = split_page(page.image, SplitOptions())
    crops = [crop_detection(page.image, detection) for detection in result.detections]
    print(f"detected {len(crops)} photographs on the example page")

    figure_split(page, crops)
    if crops:
        figure_restore(crops[0])

    write_debug_map(OUT / "detection-map.png", page.image, result)
    cv2.imwrite(
        str(OUT / "detection-map.png"), fit(cv2.imread(str(OUT / "detection-map.png")), 620)
    )

    for name in sorted(OUT.glob("*.png")):
        print(f"  {name.name}: {cv2.imread(str(name)).shape}")


if __name__ == "__main__":
    main()
