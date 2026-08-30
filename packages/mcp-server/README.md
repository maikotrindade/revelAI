# RevelAI — MCP server

**Status: not implemented.** This directory is the reserved space for it.

A [Model Context Protocol](https://modelcontextprotocol.io) server that exposes
the [RevelAI engine](../engine) to AI assistants, so an assistant can split an
album page or restore a photograph on the user's behalf.

## Scope

Wrapping the engine's existing entry points as MCP tools. The likely surface:

| Tool | Wraps |
| --- | --- |
| `split_pages` | `revelai.split.run_split` |
| `enhance_photos` | `revelai.enhance.run_enhance` |
| `verify_crops` | `revelai.vlm.verify` |
| `describe_photo` | `revelai.vlm.describe` |

The engine already returns structured reports — `SplitReport`, `EnhanceReport`,
`Verdict`, `Description` — so the tool results have a shape to serialise
rather than needing one invented for them.

## Out of scope

Reimplementing any detection, cropping or restoration logic. This package is a
transport. If a tool needs behaviour the engine does not have, that behaviour
belongs in the engine, where the tests are.

## Constraints it inherits

These are not negotiable at this layer, because they are the product's promises:

- **`split` never changes a pixel** beyond one crop and deskew.
- **`enhance` never writes into its input folder.** An MCP client asking for
  that must get the same error a CLI user gets.
- **The local backend is the default.** A tool call must not silently upload
  family photographs to a hosted backend; the confirmation the CLI performs
  needs an equivalent here, and `--yes` has no business being implicit.
- **Face restoration and colourisation stay off by default**, and a tool that
  enables them must return the warning alongside the result.

## Intended stack

| Concern | Choice |
| --- | --- |
| Language | Python, so it imports the engine directly |
| SDK | the official `mcp` Python package |
| Transport | stdio first; HTTP only if something needs it |
| Dependency | `revelai` from `packages/engine`, as a path dependency |

Python is the obvious choice here: the engine is Python, and a server in another
language would need to shell out to the CLI and parse its output, which throws
away the structured reports the engine already produces.

## Getting started

When work begins:

1. Scaffold inside this directory with its own `pyproject.toml`.
2. Depend on the engine as a path dependency so the two move together.
3. Add `.github/workflows/mcp-server-ci.yml`, path-filtered on
   `packages/mcp-server/**`.
4. Mock the engine in tests the way `packages/engine/tests/test_vlm.py` mocks
   the model calls — no test should need the network or an API key.
