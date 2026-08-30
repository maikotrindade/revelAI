"""The command line surface.

The rule these tests exist to enforce: no user mistake may surface a raw
traceback. Every failure has to come out as a sentence and a non-zero exit code.
"""

from __future__ import annotations

import numpy as np
import pytest

import synth
from revelai.cli import EXIT_ERROR, EXIT_OK, build_parser, main
from revelai.io import write_png
from revelai.naming import photo_filename


@pytest.fixture
def pages(tmp_path):
    directory = tmp_path / "pages"
    synth.write_pages(
        directory,
        {
            "page_1.png": synth.tiny_page(seed=70, photos=2),
            "page_2.png": synth.tiny_page(seed=71, photos=1),
        },
    )
    return directory


@pytest.fixture
def photos(tmp_path):
    directory = tmp_path / "photos"
    _, cast = synth.yellow_cast_image(size=(80, 60))
    for index in range(1, 3):
        write_png(directory / photo_filename(index), cast)
    return directory


class TestParser:
    def test_the_three_commands_exist(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["--help"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        for command in ("split", "enhance", "run"):
            assert command in out

    def test_no_command_prints_help(self, capsys):
        assert main([]) == EXIT_OK
        assert "usage" in capsys.readouterr().out

    def test_version(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["--version"])
        assert exc.value.code == 0
        assert "RevelAI" in capsys.readouterr().out

    def test_documented_split_defaults(self):
        args = build_parser().parse_args(["split", "pages"])
        assert str(args.output) == "photos"
        assert args.inset == 3
        assert args.min_area == 0.005
        assert args.max_area == 0.9
        assert args.start_index is None

    def test_documented_enhance_defaults(self):
        args = build_parser().parse_args(["enhance", "photos"])
        assert str(args.output) == "photos_enhanced"
        assert args.color is True, "classical colour correction is on by default"
        assert args.faces is False, "face restoration must be off by default"
        assert args.colorize is False, "colourisation must be off by default"
        assert args.backend == "local", "the local backend is the default"
        assert args.upscale == 0

    def test_colour_can_be_switched_off(self):
        assert build_parser().parse_args(["enhance", "p", "--no-color"]).color is False


class TestSplitCommand:
    def test_dry_run_reports_and_writes_nothing(self, pages, tmp_path, capsys):
        output = tmp_path / "out"
        assert main(["split", str(pages), "-o", str(output), "--dry-run"]) == EXIT_OK
        out = capsys.readouterr().out
        assert "photographs found    3" in out
        assert not output.exists() or not list(output.iterdir())

    def test_a_real_run_writes_the_photographs(self, pages, tmp_path):
        output = tmp_path / "out"
        assert main(["split", str(pages), "-o", str(output)]) == EXIT_OK
        assert sorted(p.name for p in output.iterdir()) == [photo_filename(i) for i in (1, 2, 3)]

    def test_an_empty_folder_is_an_error_not_a_traceback(self, tmp_path, capsys):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert main(["split", str(empty), "-o", str(tmp_path / "out")]) == EXIT_ERROR
        assert "No images found" in capsys.readouterr().err

    def test_a_missing_folder_is_an_error_not_a_traceback(self, tmp_path, capsys):
        assert main(["split", str(tmp_path / "nope"), "-o", str(tmp_path / "out")]) == EXIT_ERROR
        assert "error:" in capsys.readouterr().err

    def test_debug_dir_inside_the_output_is_refused(self, pages, tmp_path, capsys):
        output = tmp_path / "out"
        code = main(["split", str(pages), "-o", str(output), "--debug-dir", str(output)])
        assert code == EXIT_ERROR
        assert "debug-dir" in capsys.readouterr().err

    def test_verify_without_a_key_says_so_and_still_runs(
        self, pages, tmp_path, capsys, monkeypatch
    ):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        output = tmp_path / "out"
        assert main(["split", str(pages), "-o", str(output), "--verify"]) == EXIT_OK
        out = capsys.readouterr().out
        assert "crops verified       none" in out, "an unverified batch must not look verified"
        assert len(list(output.iterdir())) == 3


class TestEnhanceCommand:
    def test_a_real_run(self, photos, tmp_path):
        output = tmp_path / "restored"
        assert main(["enhance", str(photos), "-o", str(output)]) == EXIT_OK
        assert len(list(output.iterdir())) == 2

    def test_writing_into_the_input_is_refused(self, photos, capsys):
        assert main(["enhance", str(photos), "-o", str(photos)]) == EXIT_ERROR
        assert "never writes into its input" in capsys.readouterr().err

    def test_an_unknown_backend_is_an_error_not_a_traceback(self, photos, tmp_path, capsys):
        code = main(["enhance", str(photos), "-o", str(tmp_path / "out"), "--backend", "nope"])
        assert code == EXIT_ERROR
        assert "unknown backend" in capsys.readouterr().err

    def test_the_face_warning_is_printed_before_running(self, photos, tmp_path, capsys):
        main(["enhance", str(photos), "-o", str(tmp_path / "out"), "--faces"])
        out = capsys.readouterr().out
        assert "reconstructs faces" in out
        # The warning is wrapped, so match a phrase that fits on one line.
        assert "may not be that person" in out
        assert "not touched" in out

    def test_the_colorize_warning_is_printed(self, photos, tmp_path, capsys):
        main(["enhance", str(photos), "-o", str(tmp_path / "out"), "--colorize"])
        assert "invents the colour" in capsys.readouterr().out

    def test_no_warning_when_neither_is_requested(self, photos, tmp_path, capsys):
        main(["enhance", str(photos), "-o", str(tmp_path / "out")])
        out = capsys.readouterr().out
        assert "WARNING" not in out

    def test_unavailable_operations_are_reported(self, photos, tmp_path, capsys):
        main(["enhance", str(photos), "-o", str(tmp_path / "out"), "--faces"])
        assert "operations skipped" in capsys.readouterr().out

    def test_the_summary_says_the_originals_were_not_modified(self, photos, tmp_path, capsys):
        main(["enhance", str(photos), "-o", str(tmp_path / "out")])
        assert "originals were not modified" in capsys.readouterr().out

    def test_an_index_file_inside_the_output_is_refused(self, photos, tmp_path, capsys):
        output = tmp_path / "out"
        code = main(
            [
                "enhance",
                str(photos),
                "-o",
                str(output),
                "--index-file",
                str(output / "index.csv"),
            ]
        )
        assert code == EXIT_ERROR
        assert "index-file" in capsys.readouterr().err

    def test_an_index_file_outside_the_output_is_written(self, photos, tmp_path):
        index = tmp_path / "index.csv"
        assert (
            main(["enhance", str(photos), "-o", str(tmp_path / "out"), "--index-file", str(index)])
            == EXIT_OK
        )
        assert index.is_file()
        assert "photo_00000001.png" in index.read_text()


class TestHostedBackendConfirmation:
    def test_it_says_where_the_images_go_and_stops_without_a_yes(
        self, photos, tmp_path, capsys, monkeypatch
    ):
        monkeypatch.setenv("REPLICATE_API_TOKEN", "fake-token-for-the-prompt")
        monkeypatch.setattr("builtins.input", lambda *_: "n")
        code = main(["enhance", str(photos), "-o", str(tmp_path / "out"), "--backend", "replicate"])
        out = capsys.readouterr()
        assert "uploads your photographs" in out.out
        assert "Replicate" in out.out
        assert code == EXIT_ERROR
        assert "Nothing was uploaded" in out.err

    def test_yes_skips_the_prompt_for_automation(self, photos, tmp_path, capsys, monkeypatch):
        monkeypatch.setenv("REPLICATE_API_TOKEN", "fake-token")

        def refuse(*_):  # pragma: no cover - must not be reached
            raise AssertionError("--yes must not prompt")

        monkeypatch.setattr("builtins.input", refuse)
        main(
            [
                "enhance",
                str(photos),
                "-o",
                str(tmp_path / "out"),
                "--backend",
                "replicate",
                "--yes",
                "--no-color",
            ]
        )
        assert "Continuing because --yes was given" in capsys.readouterr().out

    def test_the_local_backend_never_prompts(self, photos, tmp_path, monkeypatch):
        def refuse(*_):  # pragma: no cover
            raise AssertionError("the local backend must not prompt")

        monkeypatch.setattr("builtins.input", refuse)
        assert main(["enhance", str(photos), "-o", str(tmp_path / "out")]) == EXIT_OK


class TestRunCommand:
    def test_both_stages_in_sequence(self, pages, tmp_path, capsys):
        split_out = tmp_path / "photos"
        enhanced = tmp_path / "restored"
        code = main(
            [
                "run",
                str(pages),
                "--out-split",
                str(split_out),
                "--out-enhanced",
                str(enhanced),
            ]
        )
        assert code == EXIT_OK
        out = capsys.readouterr().out
        assert "Stage 1 of 2" in out and "Stage 2 of 2" in out
        assert len(list(split_out.iterdir())) == 3
        assert sorted(p.name for p in enhanced.iterdir()) == sorted(
            p.name for p in split_out.iterdir()
        )

    def test_the_split_output_is_untouched_by_the_enhance_stage(self, pages, tmp_path):
        from conftest import tree_digest

        split_out = tmp_path / "photos"
        main(["split", str(pages), "-o", str(split_out)])
        before = tree_digest(split_out)
        main(["enhance", str(split_out), "-o", str(tmp_path / "restored")])
        assert tree_digest(split_out) == before


class TestBadInput:
    @pytest.mark.parametrize(
        "argv",
        [
            ["split", "does-not-exist"],
            ["enhance", "does-not-exist"],
            ["run", "does-not-exist"],
        ],
    )
    def test_a_missing_input_never_raises(self, argv, tmp_path, capsys):
        argv = [*argv]
        if argv[0] == "run":
            argv += ["--out-split", str(tmp_path / "a"), "--out-enhanced", str(tmp_path / "b")]
        else:
            argv += ["-o", str(tmp_path / "out")]
        assert main(argv) == EXIT_ERROR
        assert capsys.readouterr().err

    def test_a_corrupt_page_is_reported_not_raised(self, tmp_path, capsys):
        pages = tmp_path / "pages"
        pages.mkdir()
        (pages / "broken.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"junk" * 40)
        assert main(["split", str(pages), "-o", str(tmp_path / "out")]) == EXIT_ERROR
        assert "broken.png" in capsys.readouterr().out

    def test_an_unreadable_photo_does_not_stop_the_enhance_run(self, tmp_path, capsys):
        photos = tmp_path / "photos"
        write_png(photos / photo_filename(1), np.full((16, 16, 3), 90, np.uint8))
        (photos / photo_filename(2)).write_bytes(b"\x89PNG\r\n\x1a\n" + b"junk" * 20)
        assert main(["enhance", str(photos), "-o", str(tmp_path / "out")]) == EXIT_OK
        assert "failed" in capsys.readouterr().out
