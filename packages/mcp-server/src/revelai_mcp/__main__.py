"""Command line entry point: ``revelai-mcp``.

The server will not start without at least one allowed root. That is deliberate
and is explained in the error: every path a tool receives is chosen by a model,
so the operator says up front which directories are in play.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from revelai_mcp import __version__
from revelai_mcp.config import ENV_ROOTS, ConfigError, ServerConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="revelai-mcp",
        description="Expose the RevelAI engine to AI assistants over MCP.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "The server confines itself to the directories given with --root.\n"
            "A hosted backend uploads photographs to a third party, so it is\n"
            "chosen here and never by a tool call."
        ),
    )
    parser.add_argument("--version", action="version", version=f"revelai-mcp {__version__}")
    parser.add_argument(
        "--root",
        action="append",
        type=Path,
        metavar="DIR",
        help=(
            "a directory the server may read and write. Repeatable. "
            f"Required; may also be set as {ENV_ROOTS}."
        ),
    )
    parser.add_argument(
        "--backend",
        default=None,
        metavar="NAME",
        help=(
            "restoration backend: local (default, nothing leaves this machine) "
            "or a hosted provider, which uploads your photographs"
        ),
    )
    parser.add_argument(
        "--allow-vlm",
        action="store_true",
        default=None,
        help=(
            "enable the vision-language tools (verify_crop, describe_photo). "
            "They transmit images to a model provider and need an API key."
        ),
    )
    parser.add_argument(
        "--transport",
        default="stdio",
        choices=("stdio", "streamable-http", "sse"),
        help="MCP transport (default: stdio)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = ServerConfig.from_environment(
            roots=args.root, backend=args.backend, allow_vlm=args.allow_vlm
        )
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # stderr, not stdout: stdout is the MCP transport.
    print(f"revelai-mcp {__version__} | {config.describe()}", file=sys.stderr)
    if config.uploads_photographs:
        print(
            "warning: this server is configured with a hosted backend, so "
            "photographs will be uploaded to a third party.",
            file=sys.stderr,
        )

    from revelai_mcp.server import build_server

    build_server(config).run(transport=args.transport)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
