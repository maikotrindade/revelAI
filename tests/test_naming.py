"""Naming, ordering and numbering rules.

Section 6 of the specification is not negotiable: eight digits, global and
continuous numbering, and an ordering that two runs cannot disagree about.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from revelai.naming import (
    PHOTO_PATTERN,
    highest_existing_index,
    natural_key,
    order_on_page,
    parse_photo_index,
    photo_filename,
    resolve_start_index,
    sort_pages,
)
from revelai.split.refine import RotRect


class TestPhotoFilename:
    def test_zero_padded_to_eight_digits(self):
        assert photo_filename(1) == "photo_00000001.png"
        assert photo_filename(42) == "photo_00000042.png"
        assert photo_filename(12345678) == "photo_12345678.png"

    def test_matches_the_mandated_pattern(self):
        for i in (1, 7, 99, 100000):
            assert re.match(r"^photo_\d{8}\.png$", photo_filename(i))
            assert PHOTO_PATTERN.match(photo_filename(i))

    def test_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            photo_filename(0)
        with pytest.raises(ValueError):
            photo_filename(-1)
        with pytest.raises(ValueError):
            photo_filename(100_000_000)

    def test_round_trips_through_parse(self):
        assert parse_photo_index("photo_00000042.png") == 42
        assert parse_photo_index("photo_00000001.png") == 1

    @pytest.mark.parametrize(
        "name",
        [
            "photo_0000042.png",  # seven digits
            "photo_000000042.png",  # nine digits
            "photo_00000042.jpg",
            "photo_00000042.PNG",
            "Photo_00000042.png",
            "photo_00000042.png.bak",
            "photo_0000004a.png",
            "00000042.png",
        ],
    )
    def test_parse_rejects_near_misses(self, name):
        assert parse_photo_index(name) is None


class TestNaturalOrdering:
    def test_page_2_sorts_before_page_10(self):
        names = ["page_10.png", "page_1.png", "page_2.png"]
        assert sorted(names, key=natural_key) == ["page_1.png", "page_2.png", "page_10.png"]

    def test_sort_pages_is_natural_and_stable(self, tmp_path):
        paths = [tmp_path / n for n in ("IMG_10.jpg", "IMG_2.jpg", "IMG_1.jpg", "IMG_20.jpg")]
        ordered = [p.name for p in sort_pages(paths)]
        assert ordered == ["IMG_1.jpg", "IMG_2.jpg", "IMG_10.jpg", "IMG_20.jpg"]

    def test_mixed_case_and_padding(self):
        names = ["a2", "A10", "a1"]
        assert sorted(names, key=natural_key) == ["a1", "a2", "A10"]


class TestOrderOnPage:
    """Top to bottom, and left to right when two photos share a vertical band."""

    def test_two_rows_of_two(self):
        rects = [
            RotRect(600, 400, 200, 150, 0.0),  # top right
            RotRect(200, 410, 200, 150, 0.0),  # top left
            RotRect(600, 900, 200, 150, 0.0),  # bottom right
            RotRect(200, 890, 200, 150, 0.0),  # bottom left
        ]
        assert order_on_page(rects) == [1, 0, 3, 2]

    def test_a_slightly_higher_neighbour_does_not_break_the_band(self):
        # Same row, but one sits 12 px higher. Reading order is still left first.
        rects = [
            RotRect(700, 388, 240, 180, 0.0),
            RotRect(300, 400, 240, 180, 0.0),
        ]
        assert order_on_page(rects) == [1, 0]

    def test_a_genuinely_lower_photo_starts_a_new_band(self):
        rects = [
            RotRect(700, 900, 240, 180, 0.0),
            RotRect(300, 400, 240, 180, 0.0),
        ]
        assert order_on_page(rects) == [1, 0]

    def test_single_and_empty(self):
        assert order_on_page([]) == []
        assert order_on_page([RotRect(10, 10, 5, 5, 0.0)]) == [0]

    def test_is_a_permutation(self, hard_page):
        rects = [RotRect.from_corners(p.corners) for p in hard_page.photos]
        assert sorted(order_on_page(rects)) == list(range(len(rects)))


class TestStartIndex:
    def test_empty_folder_starts_at_one(self, tmp_path):
        assert highest_existing_index(tmp_path) == 0
        assert resolve_start_index(tmp_path, None) == 1

    def test_missing_folder_starts_at_one(self, tmp_path):
        assert resolve_start_index(tmp_path / "nope", None) == 1

    def test_continues_from_the_highest_existing_index(self, tmp_path):
        for i in (1, 2, 7):
            (tmp_path / photo_filename(i)).write_bytes(b"x")
        assert highest_existing_index(tmp_path) == 7
        assert resolve_start_index(tmp_path, None) == 8

    def test_unrelated_files_do_not_count(self, tmp_path):
        (tmp_path / "notes.txt").write_bytes(b"x")
        (tmp_path / "photo_9999.png").write_bytes(b"x")
        assert highest_existing_index(tmp_path) == 0

    def test_explicit_start_index_is_honoured(self, tmp_path):
        assert resolve_start_index(tmp_path, 500) == 500

    def test_explicit_start_index_that_would_overwrite_is_refused(self, tmp_path):
        from revelai.io import OutputCollisionError

        (tmp_path / photo_filename(5)).write_bytes(b"x")
        with pytest.raises(OutputCollisionError) as exc:
            resolve_start_index(tmp_path, 5)
        # The message has to name the file and tell the user what to do.
        assert "photo_00000005.png" in str(exc.value)

    def test_explicit_start_index_above_the_existing_ones_is_fine(self, tmp_path):
        (tmp_path / photo_filename(5)).write_bytes(b"x")
        assert resolve_start_index(tmp_path, 6) == 6

    def test_rejects_non_positive(self, tmp_path):
        with pytest.raises(ValueError):
            resolve_start_index(tmp_path, 0)


def test_output_folder_contains_only_photo_pngs(photos_dir: Path):
    entries = list(photos_dir.iterdir())
    assert entries, "fixture should not be empty"
    for entry in entries:
        assert entry.is_file(), f"{entry.name} is not a file; no subfolders allowed"
        assert PHOTO_PATTERN.match(entry.name), f"{entry.name} does not match photo_XXXXXXXX.png"
