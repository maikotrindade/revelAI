# RevelAI — MCP server

A [Model Context Protocol](https://modelcontextprotocol.io) server that exposes
the [RevelAI engine](../engine) to AI assistants, so an assistant can split an
album page or restore a photograph on your behalf.

This package is a **transport**. It contains no detection, cropping or
restoration logic — everything it does is call into `revelai` and serialise the
structured reports that come back.

> The implementation brief this was built to is [PROMPT.md](PROMPT.md).

## Install

```bash
pip install revelai-mcp
```

From the monorepo:

```bash
pip install -e packages/engine -e packages/mcp-server
```

## Run it

The server **will not start without at least one allowed root**:

```bash
revelai-mcp --root ~/Pictures/album
```

That is deliberate. Every path a tool receives is chosen by a model, which takes
its instructions from whatever it has read — including the files it is being
asked to process. So you say up front which directories are in play, and the
server refuses anything outside them.

### Configure an assistant

```json
{
  "mcpServers": {
    "revelai": {
      "command": "revelai-mcp",
      "args": ["--root", "/Users/you/Pictures/album"]
    }
  }
}
```

### Options

| Flag | Effect |
| --- | --- |
| `--root DIR` | a directory the server may read and write. Repeatable. **Required.** |
| `--backend NAME` | `local` (default, nothing leaves your machine) or a hosted provider |
| `--allow-vlm` | enable `verify_crop` and `describe_photo`, which transmit images |
| `--transport` | `stdio` (default), `streamable-http`, `sse` |

Equivalent environment variables: `REVELAI_MCP_ROOTS` (path-separated),
`REVELAI_MCP_BACKEND`, `REVELAI_MCP_ALLOW_VLM`.

## Tools

| Tool | What it does | Writes files | Transmits images |
| --- | --- | --- | --- |
| `split_pages` | Cut the individual photographs out of album page images | yes | no |
| `enhance_photos` | Restore already-separated photographs | yes | only on a hosted backend |
| `inspect_photo` | Read a photograph's RevelAI provenance metadata | no | no |
| `verify_crop` | Ask a VLM whether a crop is one complete photograph | no | **yes** |
| `describe_photo` | Caption, tags, people count, estimated decade | no | **yes** |

`verify_crop` and `describe_photo` are only registered when `--allow-vlm` is
given. Advertising a tool and then refusing at call time would claim a
capability the server does not have.

## A worked example

Driving the server over stdio, as a client does:

```
connected to: revelai 0.1.0
tools: ['split_pages', 'enhance_photos', 'inspect_photo']

split_pages    -> 2 photographs, numbering photo_00000001.png .. photo_00000002.png
enhance_photos -> 2 restored, backend local | originals_modified: False
inspect_photo  -> stage: enhance | source: page_1.png
escape attempt -> /etc/revelai is outside the directories this server may touch
```

A `split_pages` result, trimmed:

```json
{
  "output_folder": "/home/you/album/photos",
  "pages_read": 2,
  "photographs_found": 3,
  "numbering": { "first": "photo_00000001.png", "last": "photo_00000003.png" },
  "pages_needing_review": ["page_2.png"],
  "flagged_crops": [
    { "file": "photo_00000003.png", "reasons": ["may be two photographs mounted edge to edge"] }
  ],
  "failed_pages": []
}
```

The flags are part of the result, not a footnote. A crop that was flagged is not
a crop that succeeded, and an assistant summarising the run has to say so.

## What a model is not allowed to decide

Two decisions belong to whoever starts the server, and no tool parameter can
reach them.

**Where files may be read and written.** Paths are resolved, symlinks followed,
and anything landing outside an allowed root is refused — including a symlink
inside a root that points out of it, which is the case a naive prefix check
misses.

**Whether photographs are uploaded.** A hosted restoration backend sends family
photographs to a third party. The CLI handles that by asking the user; an MCP
tool cannot prompt, so it does not get to choose. There is deliberately **no
`backend` parameter** on any tool, and the vision-language tools — which
transmit the image by definition — are absent unless you enable them.

## Promises it inherits

These come from the engine and this layer does not get to relax them:

- **`enhance` never writes into its input folder.** The engine refuses before
  doing any work; that refusal is surfaced, not worked around.
- **Nothing is overwritten.** A split run continues from the highest number
  already present.
- **No crop is ever both wrong and unflagged**, and the flags survive into the
  tool result.
- **Face restoration and colourisation are off by default.** When enabled, the
  result carries the warning — that face restoration reconstructs rather than
  reveals, and that colourisation invents colour — so an assistant relaying the
  answer can relay the caveat too.

## Development

```bash
cd packages/mcp-server
pip install -e ../engine -e ".[dev]"
pytest -q
ruff check .
```

61 tests, all offline: none touches the network or needs an API key. Images come
from the engine's `tests/synth.py` rather than being committed.

## Licence

MIT. Copyright (c) 2026 Maiko Trindade. See [LICENSE](../../LICENSE).
