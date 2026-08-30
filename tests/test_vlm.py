"""Crop verification and description. Every model call here is mocked.

No test in this file may reach the network or need an API key, so the client is
always a stand-in. What is being tested is the plumbing around the model: that a
verdict becomes a review flag, that an unreachable model is reported rather than
assumed good, and that an unavailable key never stops a run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

import synth
from revelai.split import SplitOptions, run_split
from revelai.split.inspect import build_inspector, rotate_upright
from revelai.vlm.client import (
    AnthropicVLM,
    NullVLM,
    VLMClient,
    VLMUnavailable,
    encode_for_vlm,
    get_vlm_client,
)
from revelai.vlm.describe import DESCRIBE_SCHEMA, Description, describe_photo
from revelai.vlm.verify import VERIFY_SCHEMA, verify_crop

GOOD = {
    "complete": True,
    "cut_off_edges": [],
    "contains_multiple": False,
    "contains_page_background": False,
    "orientation": 0,
    "confidence": 0.95,
}


@dataclass
class FakeVLM(VLMClient):
    """Answers with whatever the test tells it to."""

    name: str = "fake"
    payload: dict = field(default_factory=lambda: dict(GOOD))
    calls: list = field(default_factory=list)
    fail_with: str = ""

    def available(self) -> bool:
        return True

    def ask_json(self, image, prompt, schema, *, system=""):
        self.calls.append((np.asarray(image).shape, schema))
        if self.fail_with:
            raise VLMUnavailable(self.fail_with)
        return dict(self.payload)


@pytest.fixture
def crop() -> np.ndarray:
    return np.full((120, 160, 3), 130, np.uint8)


class TestVerdict:
    def test_a_clean_crop_passes(self, crop):
        verdict = verify_crop(crop, FakeVLM())
        assert verdict.checked and verdict.ok
        assert verdict.problems() == []

    def test_a_cut_off_crop_is_flagged(self, crop):
        verdict = verify_crop(
            crop, FakeVLM(payload={**GOOD, "complete": False, "cut_off_edges": ["top", "left"]})
        )
        assert not verdict.ok
        assert any("cut off at the top, left" in p for p in verdict.problems())

    def test_two_photographs_in_one_crop_are_flagged(self, crop):
        verdict = verify_crop(crop, FakeVLM(payload={**GOOD, "contains_multiple": True}))
        assert not verdict.ok
        assert any("more than one photograph" in p for p in verdict.problems())

    def test_leftover_page_background_is_flagged(self, crop):
        verdict = verify_crop(crop, FakeVLM(payload={**GOOD, "contains_page_background": True}))
        assert not verdict.ok
        assert any("album paper" in p for p in verdict.problems())

    def test_orientation_is_read(self, crop):
        assert verify_crop(crop, FakeVLM(payload={**GOOD, "orientation": 270})).orientation == 270

    def test_a_sideways_but_complete_crop_still_passes(self, crop):
        """Orientation is not a defect; it is something to fix silently."""
        verdict = verify_crop(crop, FakeVLM(payload={**GOOD, "orientation": 90}))
        assert verdict.ok

    def test_a_missing_field_does_not_crash(self, crop):
        assert verify_crop(crop, FakeVLM(payload={"complete": True})).checked

    def test_the_schema_is_well_formed(self):
        for schema in (VERIFY_SCHEMA, DESCRIBE_SCHEMA):
            assert schema["type"] == "object"
            assert schema["additionalProperties"] is False
            for name in schema["required"]:
                assert name in schema["properties"], name


class TestUnavailableModel:
    def test_an_unreachable_model_leaves_the_crop_unchecked(self, crop):
        verdict = verify_crop(crop, FakeVLM(fail_with="no key"))
        assert not verdict.checked
        assert verdict.unavailable == "no key"

    def test_an_unchecked_crop_is_not_reported_as_flagged(self, crop):
        """It also must not be reported as verified. It was simply not checked."""
        verdict = verify_crop(crop, FakeVLM(fail_with="no key"))
        assert verdict.ok is True
        assert verdict.problems() == []

    def test_no_key_means_no_client_is_built(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        client = AnthropicVLM()
        assert not client.available()
        assert "ANTHROPIC_API_KEY" in client.unavailable_reason()
        with pytest.raises(VLMUnavailable):
            client.ask_json(np.zeros((4, 4, 3), np.uint8), "hello", VERIFY_SCHEMA)

    def test_the_null_client_refuses_politely(self):
        client = NullVLM()
        assert not client.available()
        with pytest.raises(VLMUnavailable):
            client.ask_json(np.zeros((4, 4, 3), np.uint8), "hello", VERIFY_SCHEMA)

    def test_get_vlm_client_never_raises_for_a_missing_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert get_vlm_client() is not None
        assert get_vlm_client("none").name == "none"

    def test_an_unknown_provider_is_a_clean_error(self):
        from revelai import RevelAIError

        with pytest.raises(RevelAIError):
            get_vlm_client("telepathy")


class TestEncoding:
    def test_a_large_crop_is_downscaled_for_the_request(self):
        media_type, data = encode_for_vlm(np.zeros((4000, 3000, 3), np.uint8), max_edge=800)
        assert media_type == "image/jpeg"
        assert data

    def test_the_original_array_is_not_modified(self):
        image = np.full((300, 200, 3), 77, np.uint8)
        copy = image.copy()
        encode_for_vlm(image)
        assert np.array_equal(image, copy)

    def test_sixteen_bit_and_grayscale_are_accepted(self):
        assert encode_for_vlm(np.full((40, 40), 100, np.uint8))[1]
        assert encode_for_vlm(np.full((40, 40, 3), 1000, np.uint16))[1]


class TestRotation:
    def test_quarter_turns_are_lossless(self):
        image = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
        assert np.array_equal(rotate_upright(rotate_upright(image, 90), 270), image)
        assert np.array_equal(rotate_upright(rotate_upright(image, 180), 180), image)

    def test_ninety_degrees_swaps_the_axes(self):
        image = np.zeros((10, 20, 3), np.uint8)
        assert rotate_upright(image, 90).shape[:2] == (20, 10)
        assert rotate_upright(image, 180).shape[:2] == (10, 20)

    def test_zero_and_nonsense_are_no_ops(self):
        image = np.arange(12, dtype=np.uint8).reshape(3, 4)
        assert np.array_equal(rotate_upright(image, 0), image)
        assert np.array_equal(rotate_upright(image, 45), image)

    def test_no_pixel_values_are_invented(self):
        image = np.random.default_rng(0).integers(0, 255, (17, 23, 3), dtype=np.uint8)
        rotated = rotate_upright(image, 90)
        assert sorted(rotated.flatten()) == sorted(image.flatten())


class TestInspector:
    def test_it_counts_what_it_actually_checked(self, crop):
        inspector = build_inspector(verify=True, client=FakeVLM())
        for index in range(3):
            inspector(crop, None, None, f"photo_{index}.png")
        assert inspector.checked == 3
        assert inspector.flagged == 0
        assert inspector.ran

    def test_an_unavailable_model_counts_nothing(self, crop):
        seen = []
        inspector = build_inspector(
            verify=True, client=FakeVLM(fail_with="no key"), on_unavailable=seen.append
        )
        for _ in range(3):
            inspector(crop, None, None, "photo.png")
        assert inspector.checked == 0
        assert not inspector.ran
        assert seen == ["no key"], "the reason should be reported once, not once per crop"

    def test_flags_come_back_for_the_review_list(self, crop):
        inspector = build_inspector(
            verify=True, client=FakeVLM(payload={**GOOD, "contains_multiple": True})
        )
        _, metadata, problems = inspector(crop, None, None, "photo.png")
        assert problems
        assert metadata["revelai:verified"] == "flagged"
        assert inspector.flagged == 1

    def test_auto_orient_rotates_the_pixels(self, crop):
        inspector = build_inspector(
            auto_orient=True, client=FakeVLM(payload={**GOOD, "orientation": 90})
        )
        pixels, metadata, _ = inspector(crop, None, None, "photo.png")
        assert pixels.shape[:2] == (crop.shape[1], crop.shape[0])
        assert "90" in metadata["revelai:auto-orient"]
        assert inspector.reoriented == 1

    def test_verify_and_auto_orient_share_one_call(self, crop):
        """Asking twice would double the cost of a batch for nothing."""
        client = FakeVLM(payload={**GOOD, "orientation": 180})
        inspector = build_inspector(verify=True, auto_orient=True, client=client)
        inspector(crop, None, None, "photo.png")
        assert len(client.calls) == 1

    def test_doing_neither_makes_no_call(self, crop):
        client = FakeVLM()
        inspector = build_inspector(verify=False, auto_orient=False, client=client)
        pixels, metadata, problems = inspector(crop, None, None, "photo.png")
        assert client.calls == []
        assert np.array_equal(pixels, crop)
        assert metadata == {} and problems == []


class TestVerificationInASplitRun:
    def test_flagged_crops_reach_the_run_summary(self, tmp_path):
        pages = tmp_path / "pages"
        synth.write_pages(pages, {"page_1.png": synth.tiny_page(seed=60, photos=2)})
        inspector = build_inspector(
            verify=True, client=FakeVLM(payload={**GOOD, "cut_off_edges": ["left"]})
        )
        report = run_split(
            sorted(pages.iterdir()), tmp_path / "out", SplitOptions(), inspect=inspector
        )
        assert report.photographs == 2
        assert len(report.flagged_crops) == 2
        assert report.needing_review

    def test_verification_metadata_is_written(self, tmp_path):
        from revelai.io import read_png_text

        pages = tmp_path / "pages"
        synth.write_pages(pages, {"page_1.png": synth.tiny_page(seed=61, photos=2)})
        output = tmp_path / "out"
        run_split(
            sorted(pages.iterdir()),
            output,
            SplitOptions(),
            inspect=build_inspector(verify=True, client=FakeVLM()),
        )
        text = read_png_text(next(iter(sorted(output.iterdir()))))
        assert text["revelai:verified"] == "yes"
        assert float(text["revelai:verify-confidence"]) == pytest.approx(0.95)

    def test_a_missing_key_does_not_stop_the_run(self, tmp_path):
        pages = tmp_path / "pages"
        synth.write_pages(pages, {"page_1.png": synth.tiny_page(seed=62, photos=2)})
        report = run_split(
            sorted(pages.iterdir()),
            tmp_path / "out",
            SplitOptions(),
            inspect=build_inspector(verify=True, client=FakeVLM(fail_with="no key")),
        )
        assert report.photographs == 2
        assert report.flagged_crops == []

    def test_dry_run_still_verifies(self, tmp_path):
        """Checking a batch before committing to it is when this matters most."""
        pages = tmp_path / "pages"
        synth.write_pages(pages, {"page_1.png": synth.tiny_page(seed=63, photos=2)})
        client = FakeVLM()
        run_split(
            sorted(pages.iterdir()),
            tmp_path / "out",
            SplitOptions(),
            dry_run=True,
            inspect=build_inspector(verify=True, client=client),
        )
        assert len(client.calls) == 2


class TestDescribe:
    def test_a_description_becomes_metadata(self, crop):
        payload = {
            "caption": "Three people on a beach",
            "tags": ["beach", "summer", "family"],
            "people_count": 3,
            "estimated_decade": "1970s",
            "decade_confidence": 0.7,
            "date_stamp": "MAR 82",
            "handwriting": "Praia, verao",
            "is_black_and_white": False,
        }
        description = describe_photo(crop, FakeVLM(payload=payload))
        metadata = description.as_metadata()
        assert metadata["revelai:caption"] == "Three people on a beach"
        assert metadata["revelai:people-count"] == "3"
        assert metadata["revelai:date-stamp"] == "MAR 82"
        assert "beach" in metadata["revelai:tags"]

    def test_an_estimated_decade_is_labelled_as_an_estimate(self, crop):
        description = Description(estimated_decade="1960s", decade_confidence=0.4)
        assert "estimated" in description.as_metadata()["revelai:estimated-decade"]

    def test_an_unavailable_model_yields_no_metadata(self, crop):
        assert describe_photo(crop, FakeVLM(fail_with="no key")).as_metadata() == {}

    def test_it_counts_people_rather_than_identifying_them(self):
        """Face recognition is out of scope; the schema must not invite it."""
        assert "people_count" in DESCRIBE_SCHEMA["properties"]
        text = str(DESCRIBE_SCHEMA).lower()
        assert "identify" not in text and "name" not in text
