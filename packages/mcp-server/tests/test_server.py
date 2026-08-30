"""The MCP layer itself: tool registration, descriptions, and a real round trip.

The tests above call the tool functions directly. These go through
``MCPServer.call_tool``, which is what a client actually does.
"""

from __future__ import annotations

import json

import pytest

from revelai_mcp.config import ServerConfig
from revelai_mcp.server import build_server


def _payload(result):
    """The structured content of a CallToolResult."""
    if getattr(result, "structuredContent", None):
        return result.structuredContent
    block = result.content[0]
    return json.loads(block.text)


class TestRegistration:
    async def test_the_default_server_exposes_three_tools(self, config):
        names = {t.name for t in await build_server(config).list_tools()}
        assert names == {"split_pages", "enhance_photos", "inspect_photo"}

    async def test_the_vlm_tools_appear_only_when_enabled(self, root):
        server = build_server(ServerConfig(roots=[root], allow_vlm=True))
        names = {t.name for t in await server.list_tools()}
        assert {"verify_crop", "describe_photo"} <= names

    async def test_a_disabled_vlm_tool_is_not_advertised(self, config):
        """Advertising a tool and then refusing would claim a capability we lack."""
        names = {t.name for t in await build_server(config).list_tools()}
        assert "verify_crop" not in names

    async def test_the_tools_that_transmit_images_say_so(self, root):
        server = build_server(ServerConfig(roots=[root], allow_vlm=True))
        for tool in await server.list_tools():
            if tool.name in ("verify_crop", "describe_photo"):
                assert "TRANSMITS THE IMAGE" in tool.description

    async def test_no_tool_exposes_a_backend_parameter(self, root):
        """A model must not be able to choose to upload."""
        server = build_server(ServerConfig(roots=[root], allow_vlm=True))
        for tool in await server.list_tools():
            assert "backend" not in (tool.input_schema.get("properties") or {})

    async def test_the_writing_tools_are_not_marked_read_only(self, config):
        by_name = {t.name: t for t in await build_server(config).list_tools()}
        assert by_name["split_pages"].annotations.read_only_hint is False
        assert by_name["enhance_photos"].annotations.read_only_hint is False
        assert by_name["inspect_photo"].annotations.read_only_hint is True

    async def test_the_instructions_mention_flags_and_warnings(self, config):
        server = build_server(config)
        assert "flagged" in server.instructions
        assert "skipped" in server.instructions


class TestRoundTrip:
    async def test_split_then_inspect_over_the_mcp_layer(self, config, pages, root):
        server = build_server(config)
        out = root / "photos"

        split = _payload(
            await server.call_tool(
                "split_pages", {"input_path": str(pages), "output_path": str(out)}
            )
        )
        assert split["photographs_found"] == 3
        assert split["numbering"]["first"] == "photo_00000001.png"

        inspected = _payload(
            await server.call_tool("inspect_photo", {"image_path": str(out / "photo_00000001.png")})
        )
        assert inspected["metadata"]["source"] == "page_1.png"

    async def test_enhance_over_the_mcp_layer_leaves_the_input_alone(self, config, photos, root):
        from conftest import tree_digest

        before = tree_digest(photos)
        server = build_server(config)
        result = _payload(
            await server.call_tool(
                "enhance_photos",
                {"input_path": str(photos), "output_path": str(root / "restored")},
            )
        )
        assert result["restored"] == 2
        assert tree_digest(photos) == before

    async def test_a_path_outside_the_roots_is_refused_over_the_mcp_layer(
        self, config, pages, tmp_path
    ):
        server = build_server(config)
        result = _payload(
            await server.call_tool(
                "split_pages",
                {"input_path": str(pages), "output_path": str(tmp_path / "escape")},
            )
        )
        assert "error" in result and "outside" in result["error"]

    async def test_an_unknown_tool_is_a_clean_error(self, config):
        from mcp.server.mcpserver.exceptions import ToolError

        server = build_server(config)
        with pytest.raises(ToolError, match="Unknown tool"):
            await server.call_tool("delete_everything", {})
