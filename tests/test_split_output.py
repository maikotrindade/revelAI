"""The output rules of section 6, and the quality rules of section 7.

These are the hard, non-negotiable ones: what appears in an output folder, what
it is called, and that the pixels went through exactly one resampling and no
colour adjustment at all.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

import synth
from revelai.io import OutputCollisionError, read_image, read_png_icc, read_png_text, write_png
from revelai.naming import PHOTO_PATTERN, photo_filename
from revelai.split import SplitOptions, run_split

PHOTO_NAME = re.compile(r"^photo_\d{8}\.png$")


@pytest.fixture(scope="session")
def small_pages(tmp_path_factory) -> Path:
    """Three small pages holding 4, 2 and 3 photographs, in that reading order."""
    directory = tmp_path_factory.mktemp("small_pages")
    synth.write_pages(
        directory,
        {
            "page_1.png": synth.tiny_page(seed=40, photos=4),
            "page_2.png": synth.tiny_page(seed=41, photos=2),
            "page_10.png": synth.tiny_page(seed=42, photos=3),
        },
    )
    return directory


@pytest.fixture(scope="session")
def split_once(small_pages, tmp_path_factory):
    """One real run, shared by the tests that only inspect its output."""
    from revelai.io import iter_input_images

    output = tmp_path_factory.mktemp("photos_once")
    # iter_input_images, not sorted(): plain sorting puts page_10 before page_2.
    report = run_split(iter_input_images(small_pages), output)
    return output, report


def _run(pages: Path, output: Path, **kwargs):
    from revelai.io import iter_input_images

    return run_split(iter_input_images(pages), output, **kwargs)


class TestOutputFolderIntegrity:
    def test_contains_only_photo_pngs(self, split_once):
        output, _ = split_once
        entries = list(output.iterdir())
        assert entries
        for entry in entries:
            assert entry.is_file(), f"{entry.name}: no subfolders in an output folder"
            assert PHOTO_NAME.match(entry.name), f"{entry.name} is not photo_XXXXXXXX.png"

    def test_no_reports_logs_or_thumbnails(self, split_once):
        output, _ = split_once
        for entry in output.rglob("*"):
            assert entry.suffix == ".png", f"{entry.name} should not be in the output folder"

    def test_every_file_is_a_readable_png(self, split_once):
        output, _ = split_once
        for entry in sorted(output.iterdir()):
            assert entry.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
            assert read_image(entry).pixels.size > 0


class TestNumbering:
    def test_is_global_and_continuous_across_pages(self, split_once):
        output, report = split_once
        indices = sorted(int(PHOTO_PATTERN.match(p.name).group(1)) for p in output.iterdir())
        assert indices == list(range(1, len(indices) + 1)), (
            "numbering must not restart per page and must not leave holes"
        )

    def test_page_order_is_natural(self, split_once):
        """page_2 before page_10, and numbering follows that order."""
        _, report = split_once
        assert [p.source.name for p in report.pages] == [
            "page_1.png",
            "page_2.png",
            "page_10.png",
        ]
        written = [name for page in report.pages for name in page.written]
        assert written == sorted(written), "numbers must increase in page order"

    def test_counts_match_the_pages(self, split_once):
        _, report = split_once
        assert [p.detections for p in report.pages] == [4, 2, 3]


class TestDeterminism:
    def test_two_runs_produce_identical_names_and_bytes(self, small_pages, tmp_path):
        first = _run(small_pages, tmp_path / "a")
        second = _run(small_pages, tmp_path / "b")
        names_a = sorted(p.name for p in (tmp_path / "a").iterdir())
        names_b = sorted(p.name for p in (tmp_path / "b").iterdir())
        assert names_a == names_b
        assert names_a, "the run should have produced something"
        for name in names_a:
            assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes(), (
                f"{name} differs between two runs over the same input"
            )
        assert [p.detections for p in first.pages] == [p.detections for p in second.pages]


class TestExistingFiles:
    """The documented policy: continue by default, refuse when told a number."""

    def test_continues_from_the_highest_existing_index(self, small_pages, tmp_path):
        output = tmp_path / "photos"
        _run(small_pages, output)
        before = sorted(p.name for p in output.iterdir())
        _run(small_pages, output)
        after = sorted(p.name for p in output.iterdir())
        assert len(after) == 2 * len(before)
        assert set(before) < set(after), "the first run's files must survive untouched"

    def test_nothing_is_overwritten(self, small_pages, tmp_path):
        output = tmp_path / "photos"
        _run(small_pages, output)
        keep = {p.name: p.read_bytes() for p in output.iterdir()}
        _run(small_pages, output)
        for name, data in keep.items():
            assert (output / name).read_bytes() == data

    def test_an_explicit_start_index_that_collides_is_refused(self, small_pages, tmp_path):
        output = tmp_path / "photos"
        _run(small_pages, output)
        with pytest.raises(OutputCollisionError):
            _run(small_pages, output, start_index=1)

    def test_an_explicit_start_index_is_honoured(self, small_pages, tmp_path):
        output = tmp_path / "photos"
        _run(small_pages, output, start_index=500)
        assert (output / photo_filename(500)).is_file()


class TestDryRun:
    def test_writes_nothing_but_reports_everything(self, small_pages, tmp_path):
        output = tmp_path / "photos"
        report = _run(small_pages, output, dry_run=True)
        assert report.photographs == 9
        assert all(page.written for page in report.pages)
        assert not output.exists() or not list(output.iterdir())

    def test_names_match_what_a_real_run_would_write(self, small_pages, tmp_path):
        planned = _run(small_pages, tmp_path / "x", dry_run=True)
        actual = _run(small_pages, tmp_path / "y")
        assert [n for p in planned.pages for n in p.written] == [
            n for p in actual.pages for n in p.written
        ]


class TestDebugOutput:
    def test_debug_maps_go_outside_the_output_folder(self, small_pages, tmp_path):
        output = tmp_path / "photos"
        debug = tmp_path / "debug"
        _run(small_pages, output, debug_dir=debug)
        assert list(debug.iterdir()), "debug maps should have been written"
        for entry in output.iterdir():
            assert PHOTO_NAME.match(entry.name)

    def test_refuses_to_write_debug_into_the_output_folder(self, small_pages, tmp_path):
        output = tmp_path / "photos"
        with pytest.raises(OutputCollisionError):
            _run(small_pages, output, debug_dir=output)


class TestCropQuality:
    def test_crop_size_matches_the_detected_rectangle(self, small_pages, tmp_path):
        """Output size is the real pixel size of the crop, rounded."""
        from revelai.io import iter_input_images
        from revelai.split import split_page

        output = tmp_path / "photos"
        report = _run(small_pages, output)
        source = iter_input_images(small_pages)[0]
        result = split_page(read_image(source).pixels, SplitOptions())
        expected = []
        for detection in result.detections:
            pts = detection.crop_corners
            width = max(np.hypot(*(pts[1] - pts[0])), np.hypot(*(pts[2] - pts[3])))
            height = max(np.hypot(*(pts[3] - pts[0])), np.hypot(*(pts[2] - pts[1])))
            expected.append((round(width), round(height)))
        written = report.pages[0].written
        for name, (want_w, want_h) in zip(written, expected, strict=True):
            got = read_image(output / name).pixels
            assert abs(got.shape[1] - want_w) <= 1
            assert abs(got.shape[0] - want_h) <= 1

    def test_the_inset_shrinks_the_crop_by_two_insets(self, small_pages, tmp_path):
        wide = _run(small_pages, tmp_path / "a", options=SplitOptions(inset=0))
        _run(small_pages, tmp_path / "b", options=SplitOptions(inset=5))
        for name in wide.pages[0].written:
            big = read_image(tmp_path / "a" / name).pixels
            small = read_image(tmp_path / "b" / name).pixels
            assert abs((big.shape[0] - small.shape[0]) - 10) <= 2
            assert abs((big.shape[1] - small.shape[1]) - 10) <= 2

    def test_no_upscaling_or_downscaling(self, small_pages, tmp_path):
        """A crop is never larger than the page it came from."""
        from revelai.io import iter_input_images

        output = tmp_path / "photos"
        _run(small_pages, output)
        page = read_image(iter_input_images(small_pages)[0]).pixels
        for entry in output.iterdir():
            crop = read_image(entry).pixels
            assert crop.shape[0] <= page.shape[0]
            assert crop.shape[1] <= page.shape[1]

    def test_colour_is_not_touched(self, tmp_path):
        """split crops and deskews. It does not adjust a single pixel value.

        A flat patch inside a photograph must come out of the crop with exactly
        the value it had on the page: no white balance, no contrast, no
        saturation, no sharpening.
        """
        import cv2

        from revelai.split import crop_detection, split_page

        page = synth.build_page(
            [synth.PhotoSpec(260, 350, 240, 180, 0.0, (37, 211, 88))],
            page_size=(520, 700),
            canvas_size=(620, 820),
            page_angle=0.0,
            uneven_light=False,
            seed=77,
        )
        # Paint a genuinely flat block in the middle of the print. The generated
        # content is deliberately noisy, so a known value has to be put there.
        patch = (13, 199, 61)
        centre = page.photos[0].corners.mean(axis=0).astype(int)
        cv2.rectangle(
            page.image,
            (centre[0] - 25, centre[1] - 25),
            (centre[0] + 25, centre[1] + 25),
            patch,
            -1,
        )

        result = split_page(page.image, SplitOptions(inset=6))
        assert result.detections, "the photograph should have been found"
        crop = crop_detection(page.image, result.detections[0])
        middle = crop[crop.shape[0] // 2, crop.shape[1] // 2]
        assert np.array_equal(middle, np.array(patch, dtype=np.uint8)), (
            f"crop centre {middle} is not the {patch} painted on the page; split changed the pixels"
        )


class TestMetadata:
    def test_provenance_is_written(self, split_once):
        output, report = split_once
        name = report.pages[0].written[0]
        text = read_png_text(output / name)
        assert text["revelai:stage"] == "split"
        assert text["revelai:source"] == "page_1.png"
        assert text["revelai:version"]
        assert len(text["revelai:corners"].split(";")) == 4
        float(text["revelai:angle"])

    def test_no_metadata_writes_none(self, small_pages, tmp_path):
        output = tmp_path / "photos"
        report = _run(small_pages, output, write_metadata=False)
        assert read_png_text(output / report.pages[0].written[0]) == {}

    def test_icc_profile_is_carried_through(self, tmp_path):
        profile = b"a-source-profile" * 6
        pages = tmp_path / "pages"
        pages.mkdir()
        write_png(
            pages / "page_1.png", synth.tiny_page(seed=43, photos=2).image, icc_profile=profile
        )
        output = tmp_path / "photos"
        _run(pages, output)
        written = list(output.iterdir())
        assert written, "the run should have produced something"
        for entry in written:
            assert read_png_icc(entry) == profile

    def test_flagged_crops_say_so(self, tmp_path):
        """A crop the pipeline is unsure about carries that in its metadata."""
        pages = tmp_path / "pages"
        synth.write_pages(pages, {"page_1.png": synth.hard_page(seed=44)})
        output = tmp_path / "photos"
        report = _run(pages, output)
        assert report.needing_review, "hard_page contains an overlapping pair"
        flagged = [e for e in output.iterdir() if "revelai:flags" in read_png_text(e)]
        assert flagged, "at least one crop should record why it was flagged"


class TestEdgeCases:
    """None of these may surface a raw traceback."""

    def test_a_page_with_no_photographs(self, tmp_path):
        pages = tmp_path / "pages"
        synth.write_pages(pages, {"blank.png": synth.empty_page(seed=45)})
        report = _run(pages, tmp_path / "photos")
        assert report.photographs == 0
        assert report.needing_review, "an empty page must be reported, not passed over"
        assert report.pages[0].ok

    def test_a_corrupt_image_is_reported_and_the_run_continues(self, tmp_path):
        pages = tmp_path / "pages"
        synth.write_pages(pages, {"good.png": synth.tiny_page(seed=46, photos=2)})
        (pages / "broken.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"nonsense" * 30)
        report = _run(pages, tmp_path / "photos")
        assert len(report.failed) == 1
        assert "broken.png" in report.failed[0].source.name
        assert report.photographs == 2, "the good page must still be processed"

    def test_an_unsupported_file_is_ignored(self, tmp_path):
        pages = tmp_path / "pages"
        synth.write_pages(pages, {"good.png": synth.tiny_page(seed=47, photos=2)})
        (pages / "notes.txt").write_text("not an image")
        report = _run(pages, tmp_path / "photos")
        assert [p.source.name for p in report.pages] == ["good.png"]

    def test_an_empty_folder_yields_an_empty_report(self, tmp_path):
        pages = tmp_path / "pages"
        pages.mkdir()
        report = _run(pages, tmp_path / "photos")
        assert report.pages == []
        assert report.photographs == 0
