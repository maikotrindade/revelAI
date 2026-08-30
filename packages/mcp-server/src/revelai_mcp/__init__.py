"""A Model Context Protocol server exposing the RevelAI engine to AI assistants.

This package is a transport. It contains no detection, cropping or restoration
logic: everything it does is call into ``revelai`` and serialise the structured
reports that come back. If a tool needs behaviour the engine does not have, that
behaviour belongs in the engine, where the tests are.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__", "ServerConfig", "build_server"]


def __getattr__(name: str):  # pragma: no cover - thin lazy re-export
    if name == "ServerConfig":
        from revelai_mcp.config import ServerConfig

        return ServerConfig
    if name == "build_server":
        from revelai_mcp.server import build_server

        return build_server
    raise AttributeError(name)
