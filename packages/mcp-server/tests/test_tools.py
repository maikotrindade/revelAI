"""The tool behaviour that matters: what survives into a result, and what a
model is not allowed to decide.
"""

from __future__ import annotations

from conftest import tree_digest
from revelai_mcp import tools
from revelai_mcp.config import ServerConfig
from revelai_mcp.messages import COLORIZE_WARNING, FACE_WARNING


class TestSplitPages:
    def test_a_dry_run_reports_without_writing(self, config, pages, root):
        out = root / "photos"
        result = tools.split_pages(config, str(pages), str(out), dry_run=True)
        assert result["photographs_found"] == 3
        assert result["dry_run"] is True
        assert not out.exists() or not list(out.iterdir())

    def test_a_real_run_writes_numbered_photographs(self, config, pages, root):
        out = root / "photos"
        result = tools.split_pages(config, str(pages), str(out))
        assert result["photographs_found"] == 3
        assert result["numbering"] == {
            "first": "photo_00000001.png",
            "last": "photo_00000003.png",
        }
        assert sorted(p.name for p in out.iterdir()) == [
            "photo_00000001.png",
            "photo_00000002.png",
            "photo_00000003.png",
        ]

    def test_paths_outside_the_roots_are_refused(self, config, pages, tmp_path):
        result = tools.split_pages(config, str(pages), str(tmp_path / "elsewhere"))
        assert "error" in result
        assert "outside" in result["error"]

    def test_an_empty_folder_is_an_error_not_an_empty_success(self, config, root):
        empty = root / "empty"
        empty.mkdir()
        result = tools.split_pages(config, str(empty), str(root / "out"))
        assert "error" in result and "No images" in result["error"]

    def test_a_page_that_needs_review_says_so(self, config, root):
        import synth

        directory = root / "hard"
        synth.write_pages(directory, {"page_1.png": synth.hard_page(seed=92)})
        result = tools.split_pages(config, str(directory), str(root / "out"))
        assert result["pages_needing_review"] == ["page_1.png"]
        assert result["pages"][0]["notes"]

    def test_nothing_is_overwritten_on_a_second_run(self, config, pages, root):
        out = root / "photos"
        first = tools.split_pages(config, str(pages), str(out))
        kept = {p.name: p.read_bytes() for p in out.iterdir()}
        second = tools.split_pages(config, str(pages), str(out))
        assert second["numbering"]["first"] == "photo_00000004.png"
        for name, data in kept.items():
            assert (out / name).read_bytes() == data
        assert first["numbering"]["first"] == "photo_00000001.png"


class TestVerificationHonesty:
    def test_verify_is_refused_when_the_server_disables_it(self, config, pages, root):
        result = tools.split_pages(config, str(pages), str(root / "out"), verify=True)
        assert "error" in result
        assert "disabled" in result["error"]

    def test_an_unreachable_model_is_not_reported_as_verified(self, root, pages, monkeypatch):
        """The batch must not look verified when nothing was checked."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        config = ServerConfig(roots=[root], allow_vlm=True)
        result = tools.split_pages(config, str(pages), str(root / "out"), verify=True)
        assert result["verification"]["ran"] is False
        assert result["verification"]["crops_checked"] == 0
        assert "not" in result["verification"]["note"]

    def test_flagged_crops_reach_the_result(self, root, pages, monkeypatch):
        """A flag raised by the model has to survive into the tool result."""
        from revelai.vlm.client import VLMClient

        class Flagging(VLMClient):
            def available(self):
                return True

            def ask_json(self, image, prompt, schema, *, system=""):
                return {
                    "complete": False,
                    "cut_off_edges": ["left"],
                    "contains_multiple": False,
                    "contains_page_background": False,
                    "orientation": 0,
                    "confidence": 0.9,
                }

        # build_inspector binds get_vlm_client at import time, so the name has
        # to be replaced where it is used, not where it is defined.
        monkeypatch.setattr("revelai.split.inspect.get_vlm_client", lambda *a, **k: Flagging())
        config = ServerConfig(roots=[root], allow_vlm=True)
        result = tools.split_pages(config, str(pages), str(root / "out"), verify=True)
        assert result["verification"]["ran"] is True
        assert result["flagged_crops"], "a flagged crop must appear in the result"
        assert "cut off" in result["flagged_crops"][0]["reasons"][0]


class TestEnhancePhotos:
    def test_the_input_tree_is_byte_identical_afterwards(self, config, photos, root):
        """The most important test in the engine, repeated at this layer."""
        before = tree_digest(photos)
        assert before
        tools.enhance_photos(config, str(photos), str(root / "restored"), denoise=True, dust=True)
        assert tree_digest(photos) == before

    def test_writing_into_the_input_folder_is_refused(self, config, photos):
        result = tools.enhance_photos(config, str(photos), str(photos))
        assert "error" in result
        assert "never writes into its input" in result["error"]

    def test_the_refusal_leaves_the_input_untouched(self, config, photos):
        before = tree_digest(photos)
        tools.enhance_photos(config, str(photos), str(photos), denoise=True)
        assert tree_digest(photos) == before

    def test_filenames_are_preserved(self, config, photos, root):
        out = root / "restored"
        tools.enhance_photos(config, str(photos), str(out))
        assert sorted(p.name for p in out.iterdir()) == sorted(p.name for p in photos.iterdir())

    def test_operations_applied_are_reported_with_versions(self, config, photos, root):
        result = tools.enhance_photos(config, str(photos), str(root / "out"), denoise=True)
        names = {op["name"] for op in result["photographs"][0]["operations"]}
        assert {"color", "denoise"} <= names
        assert all(op["engine"] for op in result["photographs"][0]["operations"])

    def test_skipped_operations_are_reported_with_reasons(self, config, photos, root):
        """An operation that did not run must never look like one that did."""
        result = tools.enhance_photos(config, str(photos), str(root / "out"), faces=True)
        skipped = {s["operation"] for s in result["skipped_operations"]}
        assert "faces" in skipped
        assert all(s["reason"] for s in result["skipped_operations"])
        applied = {op["name"] for op in result["photographs"][0]["operations"]}
        assert "faces" not in applied

    def test_the_result_states_the_originals_were_not_modified(self, config, photos, root):
        result = tools.enhance_photos(config, str(photos), str(root / "out"))
        assert result["originals_modified"] is False

    def test_a_comparison_folder_outside_the_output_is_written(self, config, photos, root):
        compare = root / "compare"
        tools.enhance_photos(config, str(photos), str(root / "out"), compare_path=str(compare))
        assert len(list(compare.iterdir())) == 2


class TestUploadsAreNotAModelsDecision:
    def test_there_is_no_backend_parameter(self):
        """A model must not be able to choose to upload family photographs."""
        import inspect

        assert "backend" not in inspect.signature(tools.enhance_photos).parameters

    def test_the_backend_comes_from_the_configuration(self, config, photos, root):
        result = tools.enhance_photos(config, str(photos), str(root / "out"))
        assert result["backend"] == "local"

    def test_a_local_server_never_reports_an_upload(self, config, photos, root):
        result = tools.enhance_photos(config, str(photos), str(root / "out"), denoise=True)
        assert not any("uploaded" in w for w in result["warnings"])

    def test_a_hosted_configuration_declares_that_it_uploads(self, root):
        assert ServerConfig(roots=[root], backend="replicate").uploads_photographs
        assert not ServerConfig(roots=[root]).uploads_photographs

    def test_vlm_tools_are_refused_when_disabled(self, config, photos):
        photo = next(iter(photos.iterdir()))
        for call in (tools.verify_crop, tools.describe_photo):
            result = call(config, str(photo))
            assert "error" in result and "disabled" in result["error"]


class TestGenerativeWarnings:
    def _fake_backend(self, monkeypatch, *names):
        from revelai.enhance.backends.base import EnhancerBackend, Operation

        class Fake(EnhancerBackend):
            name = "local"

            def capabilities(self):
                return set(names)

            def faces(self, image):
                return image, Operation("faces", "fake-gan", "1.0", generative=True)

            def colorize(self, image):
                return image, Operation("colorize", "fake-deoldify", "1.0", generative=True)

        monkeypatch.setattr("revelai.enhance.backends.get_backend", lambda *a, **k: Fake())

    def test_face_restoration_carries_its_warning(self, config, photos, root, monkeypatch):
        self._fake_backend(monkeypatch, "faces")
        result = tools.enhance_photos(config, str(photos), str(root / "out"), faces=True)
        assert FACE_WARNING in result["warnings"]

    def test_colourisation_carries_its_warning(self, config, photos, root, monkeypatch):
        self._fake_backend(monkeypatch, "colorize")
        result = tools.enhance_photos(config, str(photos), str(root / "out"), colorize=True)
        assert COLORIZE_WARNING in result["warnings"]

    def test_a_generative_operation_is_marked_in_the_result(
        self, config, photos, root, monkeypatch
    ):
        self._fake_backend(monkeypatch, "faces")
        result = tools.enhance_photos(config, str(photos), str(root / "out"), faces=True)
        ops = result["photographs"][0]["operations"]
        assert any(op["name"] == "faces" and op["generative"] for op in ops)

    def test_no_warning_when_nothing_generative_ran(self, config, photos, root):
        result = tools.enhance_photos(config, str(photos), str(root / "out"))
        assert result["warnings"] == []


class TestInspectPhoto:
    def test_it_reads_revelai_provenance(self, config, pages, root):
        out = root / "photos"
        tools.split_pages(config, str(pages), str(out))
        result = tools.inspect_photo(config, str(out / "photo_00000001.png"))
        assert result["has_revelai_metadata"] is True
        assert result["metadata"]["stage"] == "split"
        assert result["metadata"]["source"] == "page_1.png"
        assert "corners" in result["metadata"] and "angle" in result["metadata"]

    def test_a_restored_photograph_says_what_was_done_to_it(self, config, photos, root):
        out = root / "restored"
        tools.enhance_photos(config, str(photos), str(out), denoise=True)
        result = tools.inspect_photo(config, str(next(iter(out.iterdir()))))
        assert result["metadata"]["stage"] == "enhance"
        assert "denoise" in result["metadata"]["operations"]

    def test_a_non_png_is_refused_clearly(self, config, root):
        target = root / "note.txt"
        target.write_text("hello")
        result = tools.inspect_photo(config, str(target))
        assert "error" in result and "PNG" in result["error"]

    def test_a_path_outside_the_roots_is_refused(self, config, tmp_path):
        outside = tmp_path / "x.png"
        outside.write_bytes(b"x")
        assert "error" in tools.inspect_photo(config, str(outside))


class TestErrorsAreCleanNotTracebacks:
    def test_a_corrupt_image_is_reported(self, config, root):
        directory = root / "broken"
        directory.mkdir()
        (directory / "bad.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"junk" * 30)
        result = tools.split_pages(config, str(directory), str(root / "out"))
        assert result["failed_pages"], "a corrupt page must be reported"
        assert result["failed_pages"][0]["error"]

    def test_a_missing_input_is_reported(self, config, root):
        result = tools.split_pages(config, str(root / "absent"), str(root / "out"))
        assert "error" in result

    def test_an_unreadable_photo_is_reported_by_enhance(self, config, root):
        directory = root / "photos2"
        directory.mkdir()
        (directory / "photo_00000001.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"junk" * 20)
        result = tools.enhance_photos(config, str(directory), str(root / "out"))
        assert result["failed"] and result["failed"][0]["error"]
