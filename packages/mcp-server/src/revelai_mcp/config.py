"""Server configuration, and the decisions an assistant is not allowed to make.

Two settings here are safety boundaries rather than preferences, and both are
set by the operator when the server starts, never by a tool call:

* which directories the server may touch;
* whether photographs may be uploaded to anybody else's computer.

An MCP server takes its instructions from a model, which takes its instructions
from whatever it has read. Neither of those decisions can be delegated to it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["ServerConfig", "ConfigError"]

ENV_ROOTS = "REVELAI_MCP_ROOTS"
ENV_BACKEND = "REVELAI_MCP_BACKEND"
ENV_ALLOW_VLM = "REVELAI_MCP_ALLOW_VLM"


class ConfigError(Exception):
    """The server was asked to start in a configuration it will not accept."""


@dataclass
class ServerConfig:
    """What this server is permitted to do."""

    #: Directories the server may read from and write to. Never empty.
    roots: list[Path] = field(default_factory=list)

    #: Restoration backend. ``local`` keeps every photograph on this machine.
    #: A hosted backend uploads them, so it is opt-in by the operator and is not
    #: reachable from any tool parameter.
    backend: str = "local"

    #: Whether the vision-language tools may run. They transmit the image to a
    #: model provider, so they are off unless the operator turns them on *and*
    #: a key is configured.
    allow_vlm: bool = False

    def __post_init__(self) -> None:
        if not self.roots:
            raise ConfigError(
                "revelai-mcp will not start without at least one allowed root.\n"
                "Every path a tool receives comes from a model, so the server "
                "confines itself to directories you name explicitly.\n"
                f"Pass --root /path/to/photos, or set {ENV_ROOTS} to a "
                f"{os.pathsep!r}-separated list."
            )
        resolved: list[Path] = []
        for root in self.roots:
            path = Path(root).expanduser().resolve()
            if not path.is_dir():
                raise ConfigError(f"allowed root {path} is not a directory")
            resolved.append(path)
        self.roots = resolved

    @property
    def uploads_photographs(self) -> bool:
        """True when the configured backend sends images to a third party."""
        return self.backend != "local"

    def describe(self) -> str:
        """A line for the log, so the operator can see what they enabled."""
        roots = ", ".join(str(r) for r in self.roots)
        vlm = "enabled" if self.allow_vlm else "disabled"
        return (
            f"roots: {roots} | backend: {self.backend} "
            f"({'uploads images' if self.uploads_photographs else 'local only'}) "
            f"| vision-language tools: {vlm}"
        )

    @classmethod
    def from_environment(cls, **overrides) -> ServerConfig:
        """Build from environment variables, with explicit overrides winning."""
        raw_roots = os.environ.get(ENV_ROOTS, "")
        roots = [Path(p) for p in raw_roots.split(os.pathsep) if p.strip()]
        config = {
            "roots": roots,
            "backend": os.environ.get(ENV_BACKEND, "local").strip().lower() or "local",
            "allow_vlm": os.environ.get(ENV_ALLOW_VLM, "").strip().lower()
            in ("1", "true", "yes", "on"),
        }
        config.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**config)
