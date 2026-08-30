"""The tool implementations, as plain functions.

Kept separate from the MCP wiring so that the behaviour that matters - path
confinement, the refusal to upload, the warnings riding along with a result -
is testable without standing up a client and a transport.

Every function here takes a :class:`ServerConfig` and returns a JSON-able dict.
None of them raises for an ordinary failure: engine errors come back as a
``{"error": ...}`` result so the assistant can read what went wrong and say so.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from revelai_mcp import serialize
from revelai_mcp.config import ServerConfig
from revelai_mcp.messages import warnings_for
from revelai_mcp.paths import PathNotAllowed, resolve_output_within, resolve_within

__all__ = [
    "split_pages",
    "enhance_photos",
    "verify_crop",
    "describe_photo",
    "inspect_photo",
]


def _error(message: str, **extra) -> dict:
    return {"error": message, **extra}


def _inputs(path: Path, recursive: bool) -> list[Path]:
    from revelai.io import iter_input_images

    return iter_input_images(path, recursive=recursive)


def split_pages(
    config: ServerConfig,
    input_path: str,
    output_path: str,
    *,
    recursive: bool = False,
    inset: int = 3,
    min_area: float = 0.005,
    max_area: float = 0.9,
    start_index: int | None = None,
    dry_run: bool = False,
    verify: bool = False,
) -> dict:
    """Cut individual photographs out of album page images."""
    from revelai import RevelAIError
    from revelai.split import SplitOptions, run_split

    try:
        source = resolve_within(input_path, config.roots)
        destination = resolve_output_within(output_path, config.roots)
    except PathNotAllowed as exc:
        return _error(str(exc))

    if verify and not config.allow_vlm:
        return _error(
            "Crop verification sends each crop to a vision-language model, and "
            "this server was started with that disabled. Ask the operator to "
            "enable it, or call split_pages without verify."
        )

    try:
        pages = _inputs(source, recursive)
    except RevelAIError as exc:
        return _error(str(exc))
    if not pages:
        return _error(f"No images found in {source}.")

    inspector = None
    if verify:
        from revelai.split.inspect import build_inspector

        inspector = build_inspector(verify=True)

    try:
        report = run_split(
            pages,
            destination,
            SplitOptions(min_area=min_area, max_area=max_area, inset=inset),
            start_index=start_index,
            dry_run=dry_run,
            inspect=inspector,
        )
    except RevelAIError as exc:
        return _error(str(exc))

    result = serialize.split_report(report)
    if verify:
        checked = getattr(inspector, "checked", 0)
        result["verification"] = {
            "ran": bool(checked),
            "crops_checked": checked,
            "crops_flagged": getattr(inspector, "flagged", 0),
            "reason": getattr(inspector, "unavailable_reason", "") or None,
        }
        if not checked:
            result["verification"]["note"] = (
                "No crop was verified. Do not describe this batch as verified."
            )
    return result


def enhance_photos(
    config: ServerConfig,
    input_path: str,
    output_path: str,
    *,
    recursive: bool = False,
    color: bool = True,
    color_strength: float = 1.0,
    denoise: bool = False,
    dust: bool = False,
    upscale: int = 0,
    faces: bool = False,
    colorize: bool = False,
    dry_run: bool = False,
    compare_path: str | None = None,
) -> dict:
    """Restore photographs that have already been separated.

    There is deliberately no ``backend`` parameter. A hosted backend uploads
    family photographs to a third party, and that is the operator's decision,
    made when the server starts.
    """
    from revelai import RevelAIError
    from revelai.enhance import EnhanceOptions, run_enhance
    from revelai.enhance.backends import get_backend

    try:
        source = resolve_within(input_path, config.roots)
        destination = resolve_output_within(output_path, config.roots)
        comparisons = resolve_output_within(compare_path, config.roots) if compare_path else None
    except PathNotAllowed as exc:
        return _error(str(exc))

    try:
        photos = _inputs(source, recursive)
    except RevelAIError as exc:
        return _error(str(exc))
    if not photos:
        return _error(f"No images found in {source}.")

    options = EnhanceOptions(
        color=color,
        color_strength=color_strength,
        denoise=denoise,
        dust=dust,
        upscale=upscale,
        faces=faces,
        colorize=colorize,
    )

    try:
        backend = get_backend(config.backend)
        report = run_enhance(
            photos,
            destination,
            options,
            backend,
            dry_run=dry_run,
            compare_dir=comparisons,
        )
    except RevelAIError as exc:
        # The engine refuses to write into its own input before doing any work.
        # That refusal is a promise to the user, so it is surfaced, not worked
        # around by quietly choosing a different folder.
        return _error(str(exc))

    result = serialize.enhance_report(report)
    applied = {op.name for photo in report.photos for op in photo.operations}
    result["warnings"] = warnings_for(
        faces="faces" in applied,
        colorize="colorize" in applied,
        uploaded=config.uploads_photographs and bool(applied - {"color"}),
    )
    result["originals_modified"] = False
    return result


def verify_crop(config: ServerConfig, image_path: str) -> dict:
    """Ask a vision-language model whether one crop is a complete photograph."""
    from revelai import RevelAIError
    from revelai.io import read_image
    from revelai.vlm.verify import verify_crop as engine_verify

    if not config.allow_vlm:
        return _error(
            "This tool sends the image to a vision-language model, and this "
            "server was started with the vision-language tools disabled."
        )
    try:
        path = resolve_within(image_path, config.roots)
    except PathNotAllowed as exc:
        return _error(str(exc))
    try:
        pixels = read_image(path).pixels
    except RevelAIError as exc:
        return _error(str(exc))
    return {"file": path.name, **serialize.verdict(engine_verify(pixels))}


def describe_photo(config: ServerConfig, image_path: str) -> dict:
    """Caption, tag and date one photograph. It counts people, never names them."""
    from revelai import RevelAIError
    from revelai.io import read_image
    from revelai.vlm.describe import describe_photo as engine_describe

    if not config.allow_vlm:
        return _error(
            "This tool sends the image to a vision-language model, and this "
            "server was started with the vision-language tools disabled."
        )
    try:
        path = resolve_within(image_path, config.roots)
    except PathNotAllowed as exc:
        return _error(str(exc))
    try:
        pixels = read_image(path).pixels
    except RevelAIError as exc:
        return _error(str(exc))
    return {"file": path.name, **serialize.description(engine_describe(pixels))}


def inspect_photo(config: ServerConfig, image_path: str) -> dict:
    """Read a photograph's RevelAI provenance metadata.

    Reads only what RevelAI wrote: the source page, the corner coordinates on
    that page, the deskew angle, and for a restored file the exact operations
    and model versions applied to it.
    """
    from revelai.io import read_png_text

    try:
        path = resolve_within(image_path, config.roots)
    except PathNotAllowed as exc:
        return _error(str(exc))
    if path.suffix.lower() != ".png":
        return _error(
            f"{path.name} is not a PNG; RevelAI writes its provenance into PNG text chunks."
        )
    try:
        text = read_png_text(path)
    except Exception as exc:  # noqa: BLE001 - a malformed PNG must not crash the server
        return _error(f"{path.name} could not be read: {exc}")
    return serialize.provenance(text, path)


def available_tools(config: ServerConfig) -> dict[str, Any]:
    """Which tools this configuration exposes, and why any are missing."""
    vlm_reason = None if config.allow_vlm else "vision-language tools are disabled on this server"
    return {
        "split_pages": None,
        "enhance_photos": None,
        "inspect_photo": None,
        "verify_crop": vlm_reason,
        "describe_photo": vlm_reason,
    }
