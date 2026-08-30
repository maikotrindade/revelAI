"""MCP wiring: turns the functions in :mod:`revelai_mcp.tools` into MCP tools.

Nothing here decides anything. The tool descriptions matter more than usual,
though, because they are the only thing an assistant reads before choosing what
to call - so they say plainly which tools transmit an image and which ones write
files.
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from revelai_mcp import __version__, tools
from revelai_mcp.config import ServerConfig

__all__ = ["build_server", "INSTRUCTIONS"]

INSTRUCTIONS = """\
RevelAI separates the individual photographs on a scanned album page, and
restores them as a separate step that never touches the originals.

Two things to carry into any answer you give about a run:

* A crop that was flagged is not a crop that succeeded. `split_pages` returns
  `flagged_crops` and `pages_needing_review`; report them.
* An operation that was skipped did not run. `enhance_photos` returns
  `skipped_operations` with reasons; never imply a skipped operation happened.

Face restoration reconstructs faces rather than revealing them, and
colourisation invents colour. When a result carries `warnings`, pass them on.
"""


def build_server(config: ServerConfig) -> MCPServer:
    """Build the MCP server for one configuration."""
    server = MCPServer(
        name="revelai",
        title="RevelAI",
        version=__version__,
        instructions=INSTRUCTIONS,
        website_url="https://github.com/maikotrindade/revelAI",
    )

    writes = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False)
    reads = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True)

    @server.tool(
        name="split_pages",
        title="Split album pages",
        description=(
            "Cut the individual photographs out of one or more album page images, "
            "cropped and deskewed with no quality loss. Writes numbered PNG files "
            "into output_path, which must be a different folder from input_path. "
            "Nothing is overwritten: a run continues from the highest number "
            "already present. Returns the files written, plus any crops flagged "
            "for review - report those, they are part of the result."
        ),
        annotations=writes,
    )
    def split_pages(
        input_path: str,
        output_path: str,
        recursive: bool = False,
        inset: int = 3,
        min_area: float = 0.005,
        max_area: float = 0.9,
        start_index: int | None = None,
        dry_run: bool = False,
        verify: bool = False,
    ) -> dict:
        return tools.split_pages(
            config,
            input_path,
            output_path,
            recursive=recursive,
            inset=inset,
            min_area=min_area,
            max_area=max_area,
            start_index=start_index,
            dry_run=dry_run,
            verify=verify,
        )

    @server.tool(
        name="enhance_photos",
        title="Restore separated photographs",
        description=(
            "Restore photographs that have already been separated. The originals "
            "are never modified and output_path must not be the input folder. "
            "Classical colour correction runs by default and needs no model. "
            "faces and colorize are off by default because they invent detail; "
            "when enabled, the result carries a warning that must be passed on. "
            "Returns the operations actually applied per file and the ones that "
            "were skipped, with reasons."
        ),
        annotations=writes,
    )
    def enhance_photos(
        input_path: str,
        output_path: str,
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
        return tools.enhance_photos(
            config,
            input_path,
            output_path,
            recursive=recursive,
            color=color,
            color_strength=color_strength,
            denoise=denoise,
            dust=dust,
            upscale=upscale,
            faces=faces,
            colorize=colorize,
            dry_run=dry_run,
            compare_path=compare_path,
        )

    @server.tool(
        name="inspect_photo",
        title="Read a photograph's provenance",
        description=(
            "Read the RevelAI metadata written into a PNG: which page it was cut "
            "from, the corner coordinates on that page, the deskew angle, and for "
            "a restored file the exact operations and model versions applied. "
            "Reads the file only."
        ),
        annotations=reads,
    )
    def inspect_photo(image_path: str) -> dict:
        return tools.inspect_photo(config, image_path)

    # The two vision-language tools transmit the image to a model provider, so
    # they exist only when the operator turned them on. Registering them and
    # then refusing at call time would advertise a capability this server does
    # not have.
    if config.allow_vlm:

        @server.tool(
            name="verify_crop",
            title="Verify one crop",
            description=(
                "Ask a vision-language model whether an image is exactly one "
                "complete photograph: whether it is cut off at any edge, whether "
                "more than one photograph is visible, whether album paper is "
                "showing, and which way up it is. TRANSMITS THE IMAGE to the "
                "configured model provider. If the model is unavailable the "
                "result says checked: false, which is not a pass."
            ),
            annotations=ToolAnnotations(read_only_hint=True, open_world_hint=True),
        )
        def verify_crop(image_path: str) -> dict:
            return tools.verify_crop(config, image_path)

        @server.tool(
            name="describe_photo",
            title="Describe a photograph",
            description=(
                "Caption and tag a photograph, count how many people are visible, "
                "estimate the decade, and read any date stamp or handwritten "
                "caption. It counts people; it does not identify them. TRANSMITS "
                "THE IMAGE to the configured model provider."
            ),
            annotations=ToolAnnotations(read_only_hint=True, open_world_hint=True),
        )
        def describe_photo(image_path: str) -> dict:
            return tools.describe_photo(config, image_path)

    return server
