# The engine, packaged as a container image.
#
# GitHub Packages has no Python registry - it hosts npm, RubyGems, Maven, NuGet
# and containers, but not PyPI - so the container registry (ghcr.io) is how a
# Python CLI is distributed through GitHub Packages. Wheels are attached to the
# GitHub Release alongside it, for anyone who would rather pip install.
#
#   docker build -t revelai packages/engine
#   docker run --rm -v "$PWD:/work" ghcr.io/maikotrindade/revelai split ./pages
#
# The build context is this package alone, so a contributor can build the image
# without the rest of the monorepo. The licence therefore travels as an OCI
# label rather than as a copied file: the LICENSE lives at the repository root
# and duplicating it here would leave two copies to drift apart.

FROM python:3.12-slim AS build

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir --upgrade pip build \
    && python -m build --wheel --outdir /dist

FROM python:3.12-slim

LABEL org.opencontainers.image.title="RevelAI" \
      org.opencontainers.image.description="Split album page scans into individual photos, then restore them with AI" \
      org.opencontainers.image.source="https://github.com/maikotrindade/revelAI" \
      org.opencontainers.image.licenses="MIT"

# opencv-python links against libGL, which the slim image does not carry.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /dist/*.whl /tmp/
RUN python -m pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl

# Run as a normal user: this tool reads and writes the caller's photographs and
# has no reason to do that as root.
RUN useradd --create-home --uid 1000 revelai
USER revelai
WORKDIR /work

ENTRYPOINT ["revelai"]
CMD ["--help"]
