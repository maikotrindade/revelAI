"""Filesystem confinement.

Every path argument reaching this server was chosen by a model, and a model
takes its instructions from whatever it has read - including the contents of the
files it is being asked to process. So no path is trusted, and none is used
until it has been resolved and checked against the allowed roots.

The check resolves symlinks before comparing, because a symlink inside an
allowed root pointing at ``/etc`` is exactly the case a naive prefix test misses.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["PathNotAllowed", "resolve_within", "resolve_output_within"]


class PathNotAllowed(Exception):
    """A path fell outside every configured root."""


def _relative_to_any(path: Path, roots: list[Path]) -> bool:
    for root in roots:
        try:
            path.relative_to(root)
        except ValueError:
            continue
        return True
    return False


def _describe(roots: list[Path]) -> str:
    return ", ".join(str(r) for r in roots)


def resolve_within(candidate: str | Path, roots: list[Path], *, must_exist: bool = True) -> Path:
    """Resolve ``candidate`` and confirm it sits inside one of ``roots``.

    ``strict=False`` resolution is deliberate: it follows the symlinks that do
    exist while still returning a fully-resolved path for one that does not,
    which is what an output directory usually is.
    """
    if not roots:  # pragma: no cover - ServerConfig refuses to build this
        raise PathNotAllowed("no allowed roots are configured")

    try:
        path = Path(candidate).expanduser().resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise PathNotAllowed(f"{candidate!r} is not a usable path: {exc}") from exc

    if not _relative_to_any(path, roots):
        raise PathNotAllowed(
            f"{path} is outside the directories this server may touch "
            f"({_describe(roots)}). Ask the operator to add it with --root."
        )

    if must_exist and not path.exists():
        raise PathNotAllowed(f"{path} does not exist")

    # A path that exists must still be checked after its own symlinks are
    # followed: resolve(strict=False) above already did that, but a component
    # created between the two calls would not have been. Re-check the real path.
    if path.exists():
        real = path.resolve(strict=True)
        if not _relative_to_any(real, roots):
            raise PathNotAllowed(
                f"{path} resolves to {real}, which is outside the allowed directories "
                f"({_describe(roots)})."
            )
    return path


def resolve_output_within(candidate: str | Path, roots: list[Path]) -> Path:
    """Resolve a directory the tool is about to write into.

    The directory need not exist yet, but its nearest existing ancestor must be
    inside an allowed root - otherwise a caller could create a new tree anywhere
    the process has permission to write.
    """
    path = resolve_within(candidate, roots, must_exist=False)
    ancestor = path
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    if not _relative_to_any(ancestor.resolve(strict=True), roots):
        raise PathNotAllowed(
            f"{path} would be created outside the allowed directories ({_describe(roots)})."
        )
    if path.exists() and not path.is_dir():
        raise PathNotAllowed(f"{path} exists and is not a directory")
    return path
