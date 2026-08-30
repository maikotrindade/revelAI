# RevelAI

**Split album page scans into individual photos, then restore them with AI.**

You photograph a page of a family album with your phone. The page holds five
prints at different sizes and angles, and they are faded, yellowed and scratched.
RevelAI separates them and, as a separate step that never touches your
originals, restores them.

![One photograph of an album page becomes four separate, deskewed photographs](packages/engine/docs/img/split-example.png)

This repository is a monorepo with three packages.

| Package | What it is | Status |
| --- | --- | --- |
| [`packages/engine`](packages/engine) | The Python library and the `revelai` CLI. Detection, cropping, restoration, VLM verification, the local `serve` API. | 322 tests, CI on Linux/macOS/Windows × Python 3.10–3.13 |
| [`packages/mcp-server`](packages/mcp-server) | An MCP server exposing the engine to AI assistants. | 61 tests, all offline. [Brief](packages/mcp-server/PROMPT.md) |
| [`packages/website`](packages/website) | The public site: what the tool does, the honest limitations on the front page, and the [run page](https://maikotrindade.com/revelAI/run/) that drives the engine on your own machine. | Live at [maikotrindade.com/revelAI](https://maikotrindade.com/revelAI/). [Brief](packages/website/PROMPT.md) |

## Quick start

```bash
pip install revelai
```

```bash
revelai split ./album-pages -o ./photos
```

```bash
revelai enhance ./photos -o ./photos-restored --compare-dir ./compare
```

Or, if a terminal is not where you want to be:

```bash
revelai serve --open
```

That opens the [run page](https://maikotrindade.com/revelAI/run/): choose a
folder, watch the pages being separated, download the photographs as a zip. The
page is a static file with no server of its own — it drives the copy of RevelAI
you just started, on `127.0.0.1`, so the photographs still never leave your
machine.

Full documentation, the command reference and the algorithm write-up are in the
**[engine README](packages/engine/README.md)**.

## The two stages

| Stage | What it does |
| --- | --- |
| `split` | Takes photographs of album pages and returns each individual print, cropped and deskewed, **with no quality loss whatsoever**. One resampling per photograph, lossless PNG, no colour adjustment of any kind. |
| `enhance` | Takes separated photographs and returns restored versions, **never touching the originals**. Classical correction first; a model is called only for what is left. |

They are independent. Either runs on its own, in whatever order you like.

## Distribution

The CLI ships through **GitHub Packages**, as a container image on `ghcr.io`:

```bash
docker run --rm -v "$PWD:/work" ghcr.io/maikotrindade/revelai split ./album-pages
```

One thing worth being clear about: GitHub Packages hosts npm, RubyGems, Maven,
NuGet and containers — but it has **no Python registry**. The container registry
is therefore the GitHub Packages route for a Python CLI. The wheel and sdist are
attached to each GitHub Release in the same workflow, so `pip install` from a
release asset works too.

Both are published by
[`engine-release.yml`](.github/workflows/engine-release.yml) when a `v*` tag is
pushed.

## Privacy

These are family photographs, and that shapes the design rather than being a
footnote.

The local backend is the default, and the whole tool works **with no API key at
all** — nothing uploaded, no account needed. A hosted backend uploads your
photographs to a third party, so the first time one is used in a run the CLI
says where the images are going and asks. `--verify` and `--describe` also send
images, and are off unless you ask for them.

The browser route keeps the same promise rather than making an exception to it.
`revelai serve` binds to the loopback interface, the run page is a static file
with no backend of its own, and the backend a run uses is fixed by the flag the
server started with — so no web page can arrange for your photographs to be
uploaded anywhere.

## Honest limitations

Written up in full in the [engine README](packages/engine/README.md#honest-limitations).
The short version:

- **Face restoration reconstructs faces, it does not reveal them.** On a
  low-resolution photograph the face that comes out may not be that person's
  face. Off by default, warns before running, never replaces the original.
- **Colourisation invents the colour.** A plausible guess, not a record.
- **Detection fails on low-contrast pages**, which is why `--review` exists.
- **Prints that touch or overlap** may come out as one crop — reported rather
  than guessed at. The invariant the tests enforce is that no crop is ever both
  wrong and unflagged.

## Use it from an AI assistant

The MCP server exposes the engine as tools, so an assistant can split a page or
restore a photograph for you:

```bash
revelai-mcp --root ~/Pictures/album
```

It will not start without an allowed root — every path a tool receives comes
from a model, so you say up front which directories are in play. And no tool
parameter can select a hosted backend: whether photographs are uploaded is
decided when the server starts, not by the model. See
[`packages/mcp-server`](packages/mcp-server).

## Repository layout

```
revelAI/
├── packages/
│   ├── engine/          Python engine and CLI
│   │   ├── src/revelai/
│   │   │   ├── split/   detection, refinement, cropping, review
│   │   │   ├── enhance/ restoration and its backends
│   │   │   ├── vlm/     crop verification and description
│   │   │   └── server/  the loopback API behind `revelai serve`
│   │   ├── tests/
│   │   ├── docs/        figures, generated by make_figures.py
│   │   ├── Dockerfile
│   │   └── pyproject.toml
│   ├── mcp-server/      Python MCP server
│   │   ├── src/revelai_mcp/
│   │   ├── tests/
│   │   └── pyproject.toml
│   └── website/         Next.js static site
│       ├── app/
│       ├── components/
│       └── package.json
├── .github/workflows/
│   ├── engine-ci.yml       lint + tests, path-filtered
│   ├── engine-release.yml  wheel + ghcr.io container on tag
│   ├── mcp-server-ci.yml   lint + tests
│   ├── website-ci.yml      typecheck, lint, static build
│   └── website-deploy.yml  GitHub Pages
├── pnpm-workspace.yaml
├── CONTRIBUTING.md
└── LICENSE
```

Each package is self-contained: its own build config, its own tests, its own
CI workflow filtered to its own paths. Nothing outside `packages/engine/` is
needed to build, test or run the engine.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) — it covers the monorepo conventions and
the ground rules that are not style preferences but product promises.

## Licence

MIT. Copyright (c) 2026 Maiko Trindade. See [LICENSE](LICENSE).
