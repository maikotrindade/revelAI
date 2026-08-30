"""Reading, lossless PNG writing, ICC profiles and provenance metadata."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from revelai.io import (
    SUPPORTED_SUFFIXES,
    ImageReadError,
    OutputCollisionError,
    iter_input_images,
    read_image,
    read_png_icc,
    read_png_text,
    write_png,
)


@pytest.fixture
def rgb8() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (64, 96, 3), dtype=np.uint8)


@pytest.fixture
def rgb16() -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.integers(0, 65536, (48, 72, 3), dtype=np.uint16)


class TestLosslessRoundTrip:
    def test_eight_bit_is_byte_exact(self, tmp_path, rgb8):
        path = tmp_path / "a.png"
        write_png(path, rgb8)
        back = read_image(path).pixels
        assert back.dtype == np.uint8
        assert np.array_equal(back, rgb8)

    def test_sixteen_bit_is_preserved(self, tmp_path, rgb16):
        path = tmp_path / "b.png"
        write_png(path, rgb16)
        back = read_image(path).pixels
        assert back.dtype == np.uint16, "16-bit input must not be silently reduced to 8-bit"
        assert np.array_equal(back, rgb16)

    def test_grayscale_round_trips(self, tmp_path):
        grey = np.arange(256, dtype=np.uint8).reshape(16, 16)
        path = tmp_path / "g.png"
        write_png(path, grey)
        back = read_image(path).pixels
        assert np.array_equal(back.reshape(16, 16, -1)[..., 0], grey)

    def test_no_colour_quantisation(self, tmp_path):
        """A gradient with 256 distinct levels must survive with all 256."""
        img = np.repeat(np.arange(256, dtype=np.uint8)[None, :], 8, axis=0)
        img = np.stack([img] * 3, axis=-1)
        path = tmp_path / "grad.png"
        write_png(path, img)
        back = read_image(path).pixels
        assert len(np.unique(back[..., 0])) == 256

    def test_written_file_is_a_png(self, tmp_path, rgb8):
        path = tmp_path / "c.png"
        write_png(path, rgb8)
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


class TestMetadata:
    def test_text_chunks_round_trip(self, tmp_path, rgb8):
        path = tmp_path / "m.png"
        text = {
            "revelai:version": "0.1.0",
            "revelai:source": "page_01.jpg",
            "revelai:corners": "10.5,20.5;110.5,18.0;112.0,90.0;12.0,92.0",
            "revelai:angle": "-1.250",
        }
        write_png(path, rgb8, text=text)
        assert read_png_text(path) == text

    def test_unicode_text_survives(self, tmp_path, rgb8):
        """Captions come from a VLM and are not necessarily latin-1."""
        path = tmp_path / "u.png"
        text = {"revelai:caption": "Café na varanda, três crianças — 1982"}
        write_png(path, rgb8, text=text)
        assert read_png_text(path) == text

    def test_no_metadata_writes_none(self, tmp_path, rgb8):
        path = tmp_path / "n.png"
        write_png(path, rgb8, text=None)
        assert read_png_text(path) == {}

    def test_icc_profile_is_preserved(self, tmp_path, rgb8):
        path = tmp_path / "icc.png"
        profile = b"fake-icc-profile-bytes" * 8
        write_png(path, rgb8, icc_profile=profile)
        assert read_png_icc(path) == profile

    def test_icc_survives_a_read_write_cycle(self, tmp_path, rgb8):
        profile = b"another-profile" * 4
        src = tmp_path / "src.png"
        write_png(src, rgb8, icc_profile=profile)
        loaded = read_image(src)
        assert loaded.icc_profile == profile
        dst = tmp_path / "dst.png"
        write_png(dst, loaded.pixels, icc_profile=loaded.icc_profile)
        assert read_png_icc(dst) == profile


class TestOverwriteProtection:
    def test_refuses_to_overwrite_by_default(self, tmp_path, rgb8):
        path = tmp_path / "x.png"
        write_png(path, rgb8)
        with pytest.raises(OutputCollisionError) as exc:
            write_png(path, rgb8)
        assert "x.png" in str(exc.value)

    def test_overwrite_is_possible_when_asked(self, tmp_path, rgb8):
        path = tmp_path / "x.png"
        write_png(path, rgb8)
        write_png(path, rgb8, overwrite=True)

    def test_creates_parent_directories(self, tmp_path, rgb8):
        path = tmp_path / "deep" / "deeper" / "x.png"
        write_png(path, rgb8)
        assert path.is_file()


class TestReading:
    def test_corrupt_file_raises_a_clean_error(self, tmp_path):
        path = tmp_path / "corrupt.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"garbage" * 20)
        with pytest.raises(ImageReadError):
            read_image(path)

    def test_truncated_file_raises_a_clean_error(self, tmp_path, rgb8):
        path = tmp_path / "t.png"
        write_png(path, rgb8)
        data = path.read_bytes()
        path.write_bytes(data[: len(data) // 3])
        with pytest.raises(ImageReadError):
            read_image(path)

    def test_unsupported_format_raises_a_clean_error(self, tmp_path):
        path = tmp_path / "notes.txt"
        path.write_text("this is not an image")
        with pytest.raises(ImageReadError):
            read_image(path)

    def test_missing_file_raises_a_clean_error(self, tmp_path):
        with pytest.raises(ImageReadError):
            read_image(tmp_path / "absent.png")

    def test_exif_orientation_is_applied(self, tmp_path):
        """Phone photographs carry rotation in EXIF, not in the pixels."""
        img = Image.new("RGB", (40, 20), (10, 20, 30))
        exif = Image.Exif()
        exif[274] = 6  # rotate 90 CW when displaying
        path = tmp_path / "rot.jpg"
        img.save(path, exif=exif)
        loaded = read_image(path)
        assert loaded.pixels.shape[:2] == (40, 20), "EXIF orientation must be honoured"


class TestInputListing:
    def test_lists_supported_images_in_natural_order(self, tmp_path, rgb8):
        for name in ("p_10.png", "p_2.png", "p_1.png"):
            write_png(tmp_path / name, rgb8)
        (tmp_path / "readme.txt").write_text("ignore me")
        names = [p.name for p in iter_input_images(tmp_path)]
        assert names == ["p_1.png", "p_2.png", "p_10.png"]

    def test_non_recursive_by_default(self, tmp_path, rgb8):
        write_png(tmp_path / "a.png", rgb8)
        write_png(tmp_path / "sub" / "b.png", rgb8)
        assert [p.name for p in iter_input_images(tmp_path)] == ["a.png"]

    def test_recursive_walks_subfolders(self, tmp_path, rgb8):
        write_png(tmp_path / "a.png", rgb8)
        write_png(tmp_path / "sub" / "b.png", rgb8)
        names = sorted(p.name for p in iter_input_images(tmp_path, recursive=True))
        assert names == ["a.png", "b.png"]

    def test_a_single_file_is_accepted(self, tmp_path, rgb8):
        path = tmp_path / "one.png"
        write_png(path, rgb8)
        assert iter_input_images(path) == [path]

    def test_empty_folder_yields_nothing(self, tmp_path):
        assert iter_input_images(tmp_path) == []

    def test_common_phone_formats_are_supported(self):
        for suffix in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"):
            assert suffix in SUPPORTED_SUFFIXES
