# RevelAI — website

**Status: not implemented.** This directory is the reserved space for it.

The public face of [RevelAI](../../README.md): what the tool does, what it
refuses to do, and why the honest limitations are on the front page rather than
buried. The engine and the CLI are the product; this explains them.

## Scope

- A landing page built around the real before/after figures in
  [`packages/engine/docs/img`](../engine/docs/img).
- An explanation of the two stages, and of why `split` is lossless.
- The privacy position stated plainly: the default backend is local and the tool
  works with no API key at all.
- The "Honest limitations" material — face restoration reconstructs rather than
  reveals, colourisation invents — given the same prominence it has in the
  engine README.
- Install and quick-start instructions pointing at the published package.

## Out of scope

Running the engine. This is a static marketing and documentation site, not a
web application: no uploads, no processing in the browser, no accounts. Family
photographs stay on the owner's machine, which is the whole privacy position and
a website that accepted uploads would undermine it.

## Intended stack

Not yet chosen for certain. The current thinking:

| Concern | Choice |
| --- | --- |
| Framework | Next.js (App Router), static export |
| Language | TypeScript |
| Styling | Tailwind CSS |
| Components | shadcn/ui |
| Hosting | GitHub Pages or Vercel, built from this directory |

Nothing here is fixed until the first commit lands. Whatever is chosen, it
should build to static files, so the site can be served from anywhere and cannot
quietly grow a backend.

## Getting started

When work begins:

1. Scaffold inside this directory, so `packages/website/` stays self-contained.
2. Add `.github/workflows/website-ci.yml`, path-filtered on
   `packages/website/**`, mirroring how `engine-ci.yml` is scoped.
3. Add the Node entries this repository's `.gitignore` already reserves.
4. Reference the engine's figures rather than copying them, so one regeneration
   of `make_figures.py` updates both the README and the site.
