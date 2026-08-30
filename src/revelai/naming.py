"""Global sequential numbering and the deterministic ordering rules.

Two runs over the same input must produce exactly the same filenames. That
requires three things to be pinned down: how a name is formed, in what order
pages are visited, and in what order the photographs on a page are emitted.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from revelai.split.refine import RotRect

__all__ = [
    "PHOTO_PATTERN",
    "PHOTO_DIGITS",
    "MAX_PHOTO_INDEX",
    "photo_filename",
    "parse_photo_index",
    "natural_key",
    "sort_pages",
    "order_on_page",
    "highest_existing_index",
    "resolve_start_index",
]

PHOTO_DIGITS = 8
MAX_PHOTO_INDEX = 10**PHOTO_DIGITS - 1

#: The only filename shape allowed in an output folder.
PHOTO_PATTERN = re.compile(r"^photo_(\d{8})\.png$")

_NATURAL_CHUNK = re.compile(r"(\d+)")


def photo_filename(index: int) -> str:
    """``photo_00000042.png`` for index 42."""
    if not isinstance(index, int) or isinstance(index, bool):
        raise TypeError(f"photo index must be an int, got {type(index).__name__}")
    if index < 1 or index > MAX_PHOTO_INDEX:
        raise ValueError(f"photo index {index} out of range 1..{MAX_PHOTO_INDEX}")
    return f"photo_{index:0{PHOTO_DIGITS}d}.png"


def parse_photo_index(name: str) -> int | None:
    """The index encoded in a filename, or None if it is not one of ours."""
    match = PHOTO_PATTERN.match(str(name))
    return int(match.group(1)) if match else None


def natural_key(name: Any) -> tuple:
    """Sort key where ``page_2`` comes before ``page_10``.

    Digit runs compare numerically, everything else compares case-insensitively
    with the original text as a tiebreaker so the order is total and stable.
    """
    text = name.name if isinstance(name, Path) else str(name)
    parts = _NATURAL_CHUNK.split(text)
    key: list[tuple[int, Any]] = []
    for i, part in enumerate(parts):
        if i % 2:
            key.append((0, int(part)))
        elif part:
            key.append((1, part.casefold()))
    return (tuple(key), text)


def sort_pages(paths: Iterable[Path]) -> list[Path]:
    """Input pages in natural alphabetical order of filename.

    Recursive runs are ordered by the relative directory first, so that a walk
    over subfolders is as reproducible as a flat one.
    """
    return sorted(paths, key=lambda p: (natural_key(str(p.parent)), natural_key(p.name)))


def _y_range(rect: RotRect) -> tuple[float, float]:
    ys = rect.corners()[:, 1]
    return float(ys.min()), float(ys.max())


def order_on_page(rects: Sequence[RotRect], *, band_overlap: float = 0.5) -> list[int]:
    """Indices of ``rects`` in reading order: top to bottom, then left to right.

    Photographs are grouped into horizontal bands first. Two prints belong to
    the same band when their vertical extents overlap by at least
    ``band_overlap`` of the shorter one, which is what stops a print sitting a
    few pixels higher than its neighbour from being emitted first.
    """
    if not rects:
        return []

    order = sorted(range(len(rects)), key=lambda i: (_y_range(rects[i])[0], rects[i].cx))

    bands: list[list[int]] = []
    band_spans: list[tuple[float, float]] = []
    for idx in order:
        top, bottom = _y_range(rects[idx])
        height = max(bottom - top, 1e-9)
        for band, span in zip(bands, band_spans, strict=True):
            overlap = min(bottom, span[1]) - max(top, span[0])
            shorter = min(height, max(span[1] - span[0], 1e-9))
            if overlap >= band_overlap * shorter:
                band.append(idx)
                break
        else:
            bands.append([idx])
            band_spans.append((top, bottom))

    result: list[int] = []
    for band in bands:
        result.extend(sorted(band, key=lambda i: (rects[i].cx, rects[i].cy, i)))
    return result


def highest_existing_index(directory: Path | str) -> int:
    """Largest ``photo_XXXXXXXX.png`` index already in ``directory``, else 0."""
    path = Path(directory)
    if not path.is_dir():
        return 0
    highest = 0
    for entry in path.iterdir():
        index = parse_photo_index(entry.name)
        if index is not None and index > highest:
            highest = index
    return highest


def resolve_start_index(directory: Path | str, requested: int | None) -> int:
    """Decide the first photo number for a run.

    The rule, documented in the README: with no ``--start-index`` the run
    continues from the highest number already present, so a folder is never
    overwritten and an album can be digitised over several sessions. With an
    explicit ``--start-index`` the user has taken control, so a collision is an
    error rather than something to silently work around.
    """
    from revelai.io import OutputCollisionError

    path = Path(directory)
    if requested is None:
        return highest_existing_index(path) + 1

    if not isinstance(requested, int) or isinstance(requested, bool):
        raise TypeError(f"start index must be an int, got {type(requested).__name__}")
    if requested < 1 or requested > MAX_PHOTO_INDEX:
        raise ValueError(f"start index {requested} out of range 1..{MAX_PHOTO_INDEX}")

    clash = path / photo_filename(requested)
    if clash.exists():
        raise OutputCollisionError(
            f"{clash} already exists.\n"
            f"Omit --start-index to continue from "
            f"{highest_existing_index(path) + 1}, or choose another output folder."
        )
    return requested
