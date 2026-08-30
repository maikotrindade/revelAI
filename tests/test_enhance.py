"""The enhance stage.

The first test in this file is the one that matters most: after a run, the input
folder must be byte-for-byte what it was before. The originals are the only copy
of the photograph nobody has altered.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

import synth
from conftest import tree_digest
from revelai.enhance import EnhanceOptions, enhance_image, run_enhance
from revelai.enhance.backends import get_backend
from revelai.enhance.backends.base import BackendUnavailable, EnhancerBackend, Operation
from revelai.enhance.backends.local import LocalBackend
from revelai.enhance.color import correct_colour_cast, grey_deviation
from revelai.io import OutputCollisionError, read_image, read_png_icc, read_png_text, write_png
from revelai.naming import photo_filename

PHOTO_NAME = re.compile(r"^photo_\d{8}\.png$")


@pytest.fixture
def photos(tmp_path) -> Path:
    """A folder that looks like the output of split: only photo_*.png."""
    directory = tmp_path / "photos"
    _, cast = synth.yellow_cast_image(size=(160, 120))
    for index in range(1, 4):
        write_png(
            directory / photo_filename(index),
            cast,
            text={"revelai:stage": "split", "revelai:source": "page_1.png"},
        )
    return directory


class TestInputIsNeverTouched:
    """The critical one. enhance must not modify its input in any way."""

    def test_the_input_folder_is_byte_for_byte_identical_afterwards(self, photos, tmp_path):
        before = tree_digest(photos)
        assert before, "the fixture should contain photographs"
        run_enhance(
            sorted(photos.iterdir()), tmp_path / "out", EnhanceOptions(denoise=True, dust=True)
        )
        assert tree_digest(photos) == before

    def test_still_identical_when_every_operation_is_requested(self, photos, tmp_path):
        before = tree_digest(photos)
        run_enhance(
            sorted(photos.iterdir()),
            tmp_path / "out",
            EnhanceOptions(denoise=True, dust=True, faces=True, colorize=True, upscale=4),
        )
        assert tree_digest(photos) == before

    def test_still_identical_after_a_failed_run(self, photos, tmp_path):
        before = tree_digest(photos)
        (photos.parent / "extra").mkdir()
        broken = photos / "photo_00000009.png"
        broken.write_bytes(b"\x89PNG\r\n\x1a\n" + b"rubbish" * 20)
        after_adding = tree_digest(photos)
        report = run_enhance(sorted(photos.iterdir()), tmp_path / "out", EnhanceOptions())
        assert report.failed
        assert tree_digest(photos) == after_adding
        assert before.items() <= after_adding.items()

    def test_writing_into_the_input_folder_is_refused(self, photos):
        with pytest.raises(OutputCollisionError) as exc:
            run_enhance(sorted(photos.iterdir()), photos, EnhanceOptions())
        assert "never writes into its input" in str(exc.value)

    def test_the_refusal_happens_before_any_work(self, photos):
        before = tree_digest(photos)
        with pytest.raises(OutputCollisionError):
            run_enhance(sorted(photos.iterdir()), photos, EnhanceOptions(denoise=True))
        assert tree_digest(photos) == before


class TestFilenameCorrespondence:
    def test_names_are_preserved_exactly(self, photos, tmp_path):
        output = tmp_path / "out"
        run_enhance(sorted(photos.iterdir()), output, EnhanceOptions())
        assert sorted(p.name for p in output.iterdir()) == sorted(p.name for p in photos.iterdir())

    def test_original_and_restored_correspond(self, photos, tmp_path):
        output = tmp_path / "out"
        run_enhance(sorted(photos.iterdir()), output, EnhanceOptions())
        for original in photos.iterdir():
            assert (output / original.name).is_file()

    def test_output_folder_holds_only_photo_pngs(self, photos, tmp_path):
        output = tmp_path / "out"
        run_enhance(sorted(photos.iterdir()), output, EnhanceOptions())
        for entry in output.rglob("*"):
            assert entry.is_file()
            assert PHOTO_NAME.match(entry.name), entry.name


class TestClassicalColour:
    def test_a_known_yellow_cast_is_corrected(self):
        """Section 12: the grey deviation must fall below a threshold."""
        neutral, cast = synth.yellow_cast_image()
        assert grey_deviation(neutral) < 1.0
        before = grey_deviation(cast)
        assert before > 30.0, "the fixture should have a strong cast"
        after = grey_deviation(correct_colour_cast(cast))
        assert after < 12.0, f"cast not corrected: {before:.1f} -> {after:.1f}"
        assert after < before / 4.0

    def test_strength_scales_the_correction(self):
        _, cast = synth.yellow_cast_image()
        scores = [grey_deviation(correct_colour_cast(cast, strength=s)) for s in (0.0, 0.5, 1.0)]
        assert scores[0] > scores[1] > scores[2]

    def test_strength_zero_changes_nothing(self):
        _, cast = synth.yellow_cast_image()
        assert np.array_equal(correct_colour_cast(cast, strength=0.0), cast)

    def test_a_flat_image_is_not_given_invented_contrast(self):
        flat = np.full((32, 32, 3), 128, np.uint8)
        assert np.array_equal(correct_colour_cast(flat), flat)

    def test_sixteen_bit_stays_sixteen_bit(self):
        _, cast = synth.yellow_cast_image(size=(64, 64))
        deep = cast.astype(np.uint16) * 257
        result = correct_colour_cast(deep)
        assert result.dtype == np.uint16
        assert result.max() > 255, "a 16-bit image must not be squashed into 8-bit range"

    def test_it_is_on_by_default(self):
        assert EnhanceOptions().color is True
        assert "color" in EnhanceOptions().requested()

    def test_it_runs_before_any_model(self):
        """Classical first is the documented order, not an implementation detail."""
        requested = EnhanceOptions(denoise=True, dust=True, faces=True, upscale=2).requested()
        assert requested.index("color") == 0
        assert requested.index("upscale") == len(requested) - 1
        assert requested.index("faces") < requested.index("upscale")
        assert requested.index("denoise") < requested.index("faces")


class TestOperationOrder:
    def test_upscale_is_always_last(self):
        for options in (
            EnhanceOptions(upscale=2, denoise=True),
            EnhanceOptions(upscale=4, faces=True, dust=True, colorize=True),
        ):
            assert options.requested()[-1] == "upscale"

    def test_colorize_precedes_faces(self):
        order = EnhanceOptions(colorize=True, faces=True).requested()
        assert order.index("colorize") < order.index("faces")

    def test_nothing_requested_means_nothing_applied(self):
        options = EnhanceOptions(color=False)
        assert options.requested() == []
        image = np.full((16, 16, 3), 90, np.uint8)
        result = enhance_image(image, options, LocalBackend())
        assert np.array_equal(result.pixels, image)
        assert not result.changed


class TestBackends:
    def test_the_default_backend_is_local_and_needs_no_key(self):
        backend = get_backend("local")
        assert backend.name == "local"
        assert backend.sends_images_away is False
        assert {"denoise", "dust"} <= backend.capabilities()

    def test_an_unavailable_operation_is_reported_not_raised(self):
        image = np.full((24, 24, 3), 120, np.uint8)
        result = enhance_image(image, EnhanceOptions(faces=True), LocalBackend())
        assert [name for name, _ in result.skipped] == ["faces"]
        assert "faces" in result.skipped[0][1]
        assert all(op.name != "faces" for op in result.operations)

    def test_an_unavailable_operation_leaves_the_pixels_alone(self):
        image = np.full((24, 24, 3), 120, np.uint8)
        result = enhance_image(image, EnhanceOptions(color=False, colorize=True), LocalBackend())
        assert np.array_equal(result.pixels, image)

    def test_a_backend_that_raises_is_handled(self):
        class Broken(EnhancerBackend):
            name = "broken"

            def capabilities(self):
                return {"denoise"}

            def denoise(self, image):
                raise BackendUnavailable("the model file is corrupt")

        result = enhance_image(
            np.full((16, 16, 3), 100, np.uint8),
            EnhanceOptions(color=False, denoise=True),
            Broken(),
        )
        assert result.skipped == [("denoise", "the model file is corrupt")]

    def test_a_fake_backend_is_used_when_it_is_available(self):
        """Models are mocked; no test may need a model file or the network."""

        class Fake(EnhancerBackend):
            name = "fake"

            def capabilities(self):
                return {"denoise", "upscale"}

            def denoise(self, image):
                return image // 2, Operation("denoise", "fake", "1.0")

            def upscale(self, image, factor):
                return np.repeat(np.repeat(image, factor, 0), factor, 1), Operation(
                    "upscale", "fake", "1.0"
                )

        image = np.full((8, 8, 3), 200, np.uint8)
        result = enhance_image(image, EnhanceOptions(color=False, denoise=True, upscale=2), Fake())
        assert result.pixels.shape[:2] == (16, 16)
        assert [op.name for op in result.operations] == ["denoise", "upscale"]

    def test_the_hosted_backend_declares_itself_unavailable_without_a_key(self, monkeypatch):
        monkeypatch.delenv("REPLICATE_API_TOKEN", raising=False)
        backend = get_backend("replicate")
        assert backend.capabilities() == set()
        assert "REPLICATE_API_TOKEN" in backend.unavailable_reason("faces")

    def test_the_hosted_backend_says_it_sends_images_away(self):
        backend = get_backend("replicate")
        assert backend.sends_images_away is True
        assert backend.destination

    def test_an_unknown_backend_is_a_clean_error(self):
        from revelai import RevelAIError

        with pytest.raises(RevelAIError):
            get_backend("does-not-exist")


class TestModelCache:
    def test_no_model_means_no_capability(self, tmp_path, monkeypatch):
        from revelai.enhance.backends.models import ModelCache

        monkeypatch.delenv("REVELAI_MODEL_UPSCALE", raising=False)
        cache = ModelCache(tmp_path)
        assert cache.available_operations() == set()
        assert cache.load("upscale") is None

    def test_a_cached_file_makes_the_operation_available(self, tmp_path, monkeypatch):
        from revelai.enhance.backends.models import REGISTRY, ModelCache

        monkeypatch.delenv("REVELAI_MODEL_UPSCALE", raising=False)
        spec = REGISTRY["upscale"][0]
        (tmp_path / spec.filename).write_bytes(b"not really a model")
        cache = ModelCache(tmp_path)
        assert "upscale" in cache.available_operations()
        assert LocalBackend(models=cache).supports("upscale")

    def test_nothing_is_downloaded_by_default(self, tmp_path, monkeypatch):
        """A normal run must never reach the network."""
        from revelai.enhance.backends import models

        def explode(*args, **kwargs):
            raise AssertionError("a test tried to download a model")

        monkeypatch.setattr(models.ModelCache, "download", explode)
        cache = models.ModelCache(tmp_path)
        backend = LocalBackend(models=cache)
        enhance_image(np.full((16, 16, 3), 120, np.uint8), EnhanceOptions(upscale=2), backend)


class TestMetadata:
    def test_operations_are_recorded(self, photos, tmp_path):
        output = tmp_path / "out"
        run_enhance(sorted(photos.iterdir()), output, EnhanceOptions(denoise=True))
        text = read_png_text(output / photo_filename(1))
        assert text["revelai:stage"] == "enhance"
        assert "color" in text["revelai:operations"]
        assert "denoise" in text["revelai:operations"]
        assert text["revelai:backend"] == "local"

    def test_split_provenance_is_carried_forward(self, photos, tmp_path):
        output = tmp_path / "out"
        run_enhance(sorted(photos.iterdir()), output, EnhanceOptions())
        text = read_png_text(output / photo_filename(1))
        assert text["revelai:source"] == "page_1.png"

    def test_generative_operations_are_labelled(self, tmp_path):
        class Generative(EnhancerBackend):
            name = "fake"

            def capabilities(self):
                return {"faces"}

            def faces(self, image):
                return image, Operation("faces", "fake-gan", "1.0", generative=True)

        photos = tmp_path / "in"
        write_png(photos / photo_filename(1), np.full((16, 16, 3), 100, np.uint8))
        output = tmp_path / "out"
        run_enhance([photos / photo_filename(1)], output, EnhanceOptions(faces=True), Generative())
        text = read_png_text(output / photo_filename(1))
        assert "faces" in text["revelai:generative"]
        assert "invented" in text["revelai:generative"]

    def test_no_metadata_writes_none(self, photos, tmp_path):
        output = tmp_path / "out"
        run_enhance(sorted(photos.iterdir()), output, EnhanceOptions(), write_metadata=False)
        assert read_png_text(output / photo_filename(1)) == {}

    def test_icc_profile_is_preserved(self, tmp_path):
        profile = b"a-profile-of-some-kind" * 4
        photos = tmp_path / "in"
        write_png(
            photos / photo_filename(1), np.full((16, 16, 3), 100, np.uint8), icc_profile=profile
        )
        output = tmp_path / "out"
        run_enhance([photos / photo_filename(1)], output, EnhanceOptions())
        assert read_png_icc(output / photo_filename(1)) == profile


class TestComparisons:
    def test_pairs_are_written_outside_the_output_folder(self, photos, tmp_path):
        output = tmp_path / "out"
        compare = tmp_path / "compare"
        run_enhance(sorted(photos.iterdir()), output, EnhanceOptions(), compare_dir=compare)
        assert len(list(compare.iterdir())) == len(list(photos.iterdir()))
        for entry in output.iterdir():
            assert PHOTO_NAME.match(entry.name)

    def test_a_pair_is_wider_than_either_side(self, photos, tmp_path):
        compare = tmp_path / "compare"
        run_enhance(
            sorted(photos.iterdir()), tmp_path / "out", EnhanceOptions(), compare_dir=compare
        )
        pair = read_image(next(iter(compare.iterdir()))).pixels
        original = read_image(next(iter(photos.iterdir()))).pixels
        assert pair.shape[1] > original.shape[1] * 1.8

    def test_compare_dir_cannot_be_the_output_folder(self, photos, tmp_path):
        output = tmp_path / "out"
        with pytest.raises(OutputCollisionError):
            run_enhance(sorted(photos.iterdir()), output, EnhanceOptions(), compare_dir=output)


class TestEdgeCases:
    def test_dry_run_writes_nothing(self, photos, tmp_path):
        output = tmp_path / "out"
        report = run_enhance(sorted(photos.iterdir()), output, EnhanceOptions(), dry_run=True)
        assert report.restored == 3
        assert not output.exists() or not list(output.iterdir())

    def test_a_corrupt_photo_is_reported_and_the_run_continues(self, photos, tmp_path):
        (photos / "photo_00000009.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"junk" * 30)
        report = run_enhance(sorted(photos.iterdir()), tmp_path / "out", EnhanceOptions())
        assert len(report.failed) == 1
        assert report.restored == 3

    def test_an_empty_input_list_is_harmless(self, tmp_path):
        report = run_enhance([], tmp_path / "out", EnhanceOptions())
        assert report.photos == []
        assert report.restored == 0

    def test_a_tiny_image_does_not_crash(self, tmp_path):
        photos = tmp_path / "in"
        write_png(photos / photo_filename(1), np.full((2, 2, 3), 40, np.uint8))
        report = run_enhance(
            [photos / photo_filename(1)],
            tmp_path / "out",
            EnhanceOptions(denoise=True, dust=True),
        )
        assert not report.failed
