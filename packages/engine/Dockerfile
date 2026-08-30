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
# `revelai serve` works too, but note what changes inside a container: the
# server has to bind the container's own interface rather than its loopback,
# and the port is then published back to the *host's* loopback so it is still
# only reachable from the machine you are sitting at.
#
#   docker run --rm -p 127.0.0.1:8765:8765 ghcr.io/maikotrindade/revelai \
#       serve --host 0.0.0.0
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

# The default `revelai serve` port. Publishing it is opt-in, and the comment at
# the top of this file explains why it should be published to 127.0.0.1 only.
EXPOSE 8765

ENTRYPOINT ["revelai"]
CMD ["--help"]
