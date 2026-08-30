"""``revelai serve``: run the engine from a page in your own browser.

This exists because the terminal is not where most people who own a shoebox of
album pages live, and it is built so that using a browser costs none of the
privacy the command line version has. The server binds to the loopback
interface, so the only machine that can reach it is the one it is running on.
The page that drives it talks to ``http://127.0.0.1`` and nowhere else, which is
why a *static* website - one with no backend of its own - can offer a working
run button without a single photograph leaving the house.

Two decisions are the operator's and not the page's, and they are fixed when the
process starts: which restoration backend is used, and whether the operations
that reconstruct or invent detail may run at all. See :mod:`revelai.server.config`.
"""

from __future__ import annotations

from revelai.server.config import DEFAULT_PORT, ServeConfig, ServeConfigError
from revelai.server.folder import FolderCheck, check_listing, verify_image_bytes
from revelai.server.jobs import STAGES, Job, JobError, JobStore

__all__ = [
    "DEFAULT_PORT",
    "ServeConfig",
    "ServeConfigError",
    "FolderCheck",
    "check_listing",
    "verify_image_bytes",
    "Job",
    "JobStore",
    "JobError",
    "STAGES",
    "build_server",
]


def build_server(config: ServeConfig, **kwargs):
    """A configured server and job store, for tests and for embedding."""
    from revelai.server.app import build_server as _build

    return _build(config, **kwargs)
