# Implementation brief — RevelAI MCP server

This is the working brief for `packages/mcp-server`. Read it end to end before
writing anything. It assumes no prior knowledge of the repository.

## 1. What RevelAI is

People digitising old photo albums photograph a whole album page with a phone.
The page holds several printed photographs at different sizes and angles.
RevelAI separates them (`split`) and, as an independent second stage that never
touches the originals, restores them (`enhance`).

The engine is built and shipping, in [`packages/engine`](../engine). Its README
is the authoritative technical reference. **Read it before starting.**

## 2. What you are building

A [Model Context Protocol](https://modelcontextprotocol.io) server that exposes
the engine to AI assistants, so an assistant can split an album page or restore
a photograph on the user's behalf.

**This package is a transport.** It contains no detection, cropping or
restoration logic. If a tool needs behaviour the engine does not have, that
behaviour goes in the engine, where the tests are.

## 3. Stack

| Concern | Choice |
| --- | --- |
| Language | Python 3.10+ — the engine is Python, so import it directly |
| SDK | the official `mcp` package |
| Transport | stdio first. Add HTTP only when something concrete needs it |
| Engine dependency | path dependency on `../engine`, so the two move together |
| Tests | pytest, engine mocked, no network |
| Lint | ruff, same config style as the engine |

Python is not a preference here. A server in another language would have to
shell out to the CLI and parse its stdout, throwing away the structured reports
the engine already returns.

## 4. The engine API you are wrapping

These signatures are current. Verify them against the source before relying on
them; do not invent parameters.

```python
from revelai.split import run_split, SplitOptions, SplitReport
from revelai.enhance import run_enhance, EnhanceOptions, EnhanceReport
from revelai.enhance.backends import get_backend
from revelai.vlm.verify import verify_crop, Verdict
from revelai.vlm.describe import describe_photo, Description
from revelai.io import read_image, iter_input_images

run_split(inputs: list[Path], output: Path, options: SplitOptions | None = None, *,
          start_index: int | None = None, dry_run: bool = False,
          debug_dir: Path | None = None, write_metadata: bool = True,
          jobs: int = 1, on_page=None, review=None, inspect=None) -> SplitReport

run_enhance(inputs: list[Path], output: Path, options: EnhanceOptions | None = None,
            backend: EnhancerBackend | None = None, *, dry_run: bool = False,
            compare_dir: Path | None = None, write_metadata: bool = True,
            jobs: int = 1, describe=None, on_photo=None) -> EnhanceReport
```

Report shapes, which is what makes serialising results easy:

```
SplitReport   pages: list[PageReport], output, dry_run, start_index
              .photographs  .failed  .needing_review  .flagged_crops
PageReport    source, detections, written: list[str], notes: list[str],
              needs_review, error, flagged_crops: list[(name, reasons)]  .ok

EnhanceReport photos: list[PhotoReport], output, dry_run, backend
              .restored  .failed  .skipped_operations
PhotoReport   source, written, operations: list[Operation],
              skipped: list[(name, reason)], error  .ok
Operation     name, engine, version, generative

Verdict       complete, cut_off_edges, contains_multiple,
              contains_page_background, orientation, confidence, note,
              unavailable  .checked  .ok  .problems()
Description   caption, tags, people_count, estimated_decade,
              decade_confidence, date_stamp, handwriting,
              is_black_and_white, unavailable  .checked  .as_metadata()
```

Options:

```
SplitOptions    min_area, max_area, inset, search_radius, angle_span,
                angle_step, overlap_warn, area_deviation, soft_edge_width,
                seam_warning, refine_passes
EnhanceOptions  color, color_strength, denoise, dust, upscale, faces, colorize
```

Every deliberate error the engine raises subclasses `revelai.RevelAIError`.
Anything else escaping is an engine bug — report it, do not paper over it.

## 5. Tools to expose

Five. Resist adding more until someone asks.

### `split_pages`
Cut individual photographs out of album page images.

Input: `input_path` (file or directory), `output_path`, optional `recursive`,
`inset`, `min_area`, `max_area`, `start_index`, `dry_run`, `verify`.

Returns: pages read, photographs found, the filenames written, the numbering
range, and — this is the part that matters — **every flagged crop with its
reasons**, plus `needs_review` per page.

### `enhance_photos`
Restore already-separated photographs.

Input: `input_path`, `output_path`, optional `color`, `color_strength`,
`denoise`, `dust`, `upscale`, `faces`, `colorize`, `dry_run`, `compare_dir`.

Returns: photographs restored, the operations actually applied per file with
model names and versions, **and the operations that were skipped with the
reason**. A caller must never be led to believe an operation ran when it did
not.

### `verify_crop`
Ask a vision-language model whether one crop is a single complete photograph.

Input: `image_path`. Returns the `Verdict` fields. When the model is
unavailable, say so — `checked: false` — rather than returning a pass.

### `describe_photo`
Caption, tags, people count, estimated decade, date stamp, handwriting.

Input: `image_path`. It **counts** people; it never identifies them. Do not add
a tool, a parameter or a prompt that invites identification.

### `inspect_photo`
Read a photograph's RevelAI provenance metadata: source page, corner
coordinates, angle applied, and for restored files the exact operations and
model versions. Wraps `revelai.io.read_png_text`.

## 6. Hard rules

These are the product's promises. The MCP layer inherits every one of them, and
an assistant calling a tool must not be able to route around any of them.

1. **`enhance` never writes into its input folder.** The engine raises
   `OutputCollisionError` before doing any work. Surface that error; do not
   catch it and pick a different folder.
2. **Nothing is overwritten.** With no `start_index`, a split run continues from
   the highest number already present. Keep that behaviour.
3. **Output folders contain only `photo_XXXXXXXX.png`.** Comparisons, debug maps
   and index files go to separate directories the caller names explicitly.
4. **No crop is ever both wrong and unflagged.** The engine guarantees this and
   the tool result must carry the flags through. Never summarise a run as
   successful while dropping the flags on the floor.
5. **Face restoration and colourisation stay off by default.** When a call
   enables them, the tool result must include the warning text alongside the
   output — that face restoration reconstructs rather than reveals, and that
   colourisation invents colour. An assistant relaying the result to a user has
   to be able to relay the warning too.
6. **Hosted backends are opt-in at the server level, not per call.** See §7.

## 7. Safety model

An MCP server takes instructions from a model, which takes instructions from
whatever it has read. Two things need real care, and neither is optional.

### 7.1 Filesystem confinement

Every path argument is attacker-influenced. The server takes an **allowed roots**
list at startup — command-line argument or environment variable — and:

- resolves every incoming path with `Path.resolve()`,
- rejects anything that does not sit inside an allowed root,
- rejects symlinks that escape,
- refuses to run at all if no roots are configured.

There is no default of "the whole filesystem". A server started without roots
should exit with a message explaining how to configure them.

### 7.2 Uploads

A hosted restoration backend uploads family photographs to a third party. The
CLI handles this by asking the user; **an MCP tool cannot prompt, so it must not
decide**. Therefore:

- The local backend is the only one available unless the operator explicitly
  enables a hosted one at startup.
- `backend` is not a tool parameter. A model cannot choose to upload.
- The same applies to `verify_crop` and `describe_photo`, which send images to a
  vision-language model: available only when the operator has configured a key,
  and the tool description must state plainly that they transmit the image.

If you find yourself adding a `--yes`-equivalent that a model can set, stop.

## 8. Error handling

- Catch `RevelAIError` and return it as a tool error with the message intact.
  The engine's messages are written for humans and say what to do next.
- Never return a partial success as a success. A run where three of ten
  photographs failed reports both numbers.
- A missing or unreadable file is an error, not an empty result.
- Long runs: `run_split` and `run_enhance` accept `on_page` / `on_photo`
  callbacks. Use them for progress reporting if the transport supports it.

## 9. Testing

Mirror how the engine tests itself. Look at
`packages/engine/tests/test_vlm.py` for the mocking pattern.

- **No test touches the network or needs an API key.** Mock the VLM client and
  any hosted backend.
- Generate fixtures with `packages/engine/tests/synth.py` rather than committing
  images.
- Cover, at minimum: path confinement rejects escapes (`..`, absolute paths
  outside roots, escaping symlinks); a hosted backend is unreachable through any
  tool parameter; flags survive into the tool result; `faces`/`colorize` results
  carry the warning; `enhance` into its own input is refused; a `RevelAIError`
  becomes a clean tool error rather than a traceback.
- Add a test that the input tree is byte-for-byte identical after an
  `enhance_photos` call, the way the engine does. It is the most important test
  in the engine and it is worth repeating at this layer.

## 10. CI

Add `.github/workflows/mcp-server-ci.yml`, modelled on
[`engine-ci.yml`](../../.github/workflows/engine-ci.yml) and **path-filtered** to
`packages/mcp-server/**` plus its own workflow file. Lint and test on Linux,
macOS and Windows across Python 3.10–3.13, matching the engine.

## 11. Definition of done

- [ ] The five tools in §5 work over stdio against a real MCP client.
- [ ] The server refuses to start with no allowed roots configured.
- [ ] A path outside the allowed roots is rejected, with a test proving it.
- [ ] No tool parameter can select a hosted backend or enable uploads.
- [ ] `faces` and `colorize` results carry the warning text.
- [ ] Flagged crops appear in the `split_pages` result.
- [ ] Skipped operations appear in the `enhance_photos` result with reasons.
- [ ] The input tree is byte-identical after an enhance run, under test.
- [ ] `mcp-server-ci.yml` is green and does not run when only the engine changes.
- [ ] A worked example is in this package's README: config snippet plus one
      round trip.

## 12. Out of scope

Reimplementing anything the engine does. Interactive review — it needs a window
and a human, and there is no sensible MCP equivalent. Face recognition or
identification of any kind. A hosted deployment. Writing outside the allowed
roots for any reason.
