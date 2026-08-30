"""Filesystem confinement.

Every path a tool receives was chosen by a model, which takes its instructions
from whatever it has read - including the files it is being asked to process.
These tests are the boundary.
"""

from __future__ import annotations

import os

import pytest

from revelai_mcp.config import ConfigError, ServerConfig
from revelai_mcp.paths import PathNotAllowed, resolve_output_within, resolve_within


class TestServerRefusesUnsafeConfigurations:
    def test_no_roots_is_refused(self):
        with pytest.raises(ConfigError) as exc:
            ServerConfig(roots=[])
        assert "allowed root" in str(exc.value)

    def test_the_refusal_says_how_to_fix_it(self):
        with pytest.raises(ConfigError) as exc:
            ServerConfig(roots=[])
        assert "--root" in str(exc.value)

    def test_a_root_that_is_not_a_directory_is_refused(self, tmp_path):
        target = tmp_path / "a-file"
        target.write_text("not a directory")
        with pytest.raises(ConfigError):
            ServerConfig(roots=[target])

    def test_roots_are_resolved(self, tmp_path):
        (tmp_path / "real").mkdir()
        config = ServerConfig(roots=[tmp_path / "real" / ".." / "real"])
        assert config.roots == [(tmp_path / "real").resolve()]

    def test_environment_without_roots_is_refused(self, monkeypatch):
        monkeypatch.delenv("REVELAI_MCP_ROOTS", raising=False)
        with pytest.raises(ConfigError):
            ServerConfig.from_environment()

    def test_environment_roots_are_read(self, tmp_path, monkeypatch):
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        monkeypatch.setenv(
            "REVELAI_MCP_ROOTS", os.pathsep.join([str(tmp_path / "a"), str(tmp_path / "b")])
        )
        assert len(ServerConfig.from_environment().roots) == 2


class TestConfinement:
    def test_a_path_inside_a_root_is_allowed(self, root):
        (root / "x.png").write_bytes(b"x")
        assert resolve_within(root / "x.png", [root]).name == "x.png"

    def test_dot_dot_escape_is_refused(self, root):
        with pytest.raises(PathNotAllowed):
            resolve_within(root / ".." / "escaped.png", [root])

    def test_an_absolute_path_outside_is_refused(self, root):
        with pytest.raises(PathNotAllowed):
            resolve_within("/etc/passwd", [root])

    def test_the_refusal_names_the_allowed_directories(self, root):
        with pytest.raises(PathNotAllowed) as exc:
            resolve_within("/etc/passwd", [root])
        assert str(root) in str(exc.value)

    @pytest.mark.skipif(os.name == "nt", reason="symlink creation needs privileges on Windows")
    def test_a_symlink_escaping_the_root_is_refused(self, root, tmp_path):
        """The case a naive prefix check misses."""
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.png").write_bytes(b"secret")
        (root / "link.png").symlink_to(outside / "secret.png")
        with pytest.raises(PathNotAllowed):
            resolve_within(root / "link.png", [root])

    @pytest.mark.skipif(os.name == "nt", reason="symlink creation needs privileges on Windows")
    def test_a_symlinked_directory_escaping_the_root_is_refused(self, root, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        (root / "link").symlink_to(outside, target_is_directory=True)
        with pytest.raises(PathNotAllowed):
            resolve_within(root / "link", [root])

    def test_a_missing_path_is_refused_when_it_must_exist(self, root):
        with pytest.raises(PathNotAllowed):
            resolve_within(root / "absent.png", [root])

    def test_several_roots_are_honoured(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        a.mkdir(), b.mkdir()
        (b / "x.png").write_bytes(b"x")
        assert resolve_within(b / "x.png", [a, b]).exists()


class TestOutputConfinement:
    def test_a_new_folder_inside_a_root_is_allowed(self, root):
        assert resolve_output_within(root / "new" / "deeper", [root]).name == "deeper"

    def test_a_new_folder_outside_a_root_is_refused(self, root, tmp_path):
        with pytest.raises(PathNotAllowed):
            resolve_output_within(tmp_path / "elsewhere", [root])

    def test_an_existing_file_is_not_an_output_folder(self, root):
        (root / "afile").write_bytes(b"x")
        with pytest.raises(PathNotAllowed):
            resolve_output_within(root / "afile", [root])
