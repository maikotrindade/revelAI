"""What a running ``revelai serve`` is permitted to do.

Everything in here is decided by the person who starts the server, and none of
it can be changed by a request. That split is the whole security model, and it
is the same one ``packages/mcp-server`` uses: a browser page is no more
trustworthy than a language model, so the two decisions that matter - whether
photographs leave this machine, and whether anything generative may run - are
taken at startup by a human typing a flag.

The server binds to the loopback interface. It is a local appliance that a page
in your own browser drives; it is not something to put on a network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import urlsplit

from revelai import RevelAIError

__all__ = [
    "ServeConfig",
    "ServeConfigError",
    "DEFAULT_PORT",
    "DEFAULT_ALLOWED_ORIGINS",
    "LOOPBACK_HOSTS",
]

#: Arbitrary, memorable, and outside the range anything common squats on.
DEFAULT_PORT = 8765

#: The published site, so that the page at /run works with no extra flag. It is
#: an *origin*: the browser sends it, the server compares it, and nothing about
#: it lets the site read a byte it was not handed by the person using it.
DEFAULT_ALLOWED_ORIGINS: tuple[str, ...] = ("https://maikotrindade.com",)

#: Hostnames that resolve to this machine and nowhere else. A request whose
#: Host header is not one of these is a DNS-rebinding attempt: some other name
#: has been pointed at 127.0.0.1 so that a page on that name counts as
#: same-origin. Checking Host is what closes that door.
LOOPBACK_HOSTS: frozenset[str] = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})


class ServeConfigError(RevelAIError):
    """The server was asked to start in a configuration it will not accept."""


def _origin_of(value: str) -> str:
    """Normalise a URL to a bare ``scheme://host[:port]`` origin."""
    parts = urlsplit(value.strip())
    if not parts.scheme or not parts.netloc:
        raise ServeConfigError(
            f"{value!r} is not an origin. An origin looks like "
            f"https://example.com or http://localhost:3000 - scheme and host, "
            f"no path."
        )
    return f"{parts.scheme}://{parts.netloc}".rstrip("/")


def is_loopback_host(host_header: str) -> bool:
    """True when a ``Host`` header names this machine and only this machine."""
    host = (host_header or "").strip()
    if not host:
        return False
    # Strip the port, keeping bracketed IPv6 literals intact.
    if host.startswith("["):
        host = host.partition("]")[0] + "]"
    elif ":" in host:
        host = host.rpartition(":")[0]
    return host.lower() in LOOPBACK_HOSTS


@dataclass
class ServeConfig:
    """The settings a request cannot touch."""

    host: str = "127.0.0.1"
    port: int = DEFAULT_PORT

    #: Extra browser origins allowed to drive this server. Loopback origins are
    #: always allowed and are not listed here.
    allow_origins: tuple[str, ...] = DEFAULT_ALLOWED_ORIGINS

    #: Restoration backend. ``local`` keeps every photograph on this machine.
    #: No request field can change this, which is the point of it living here.
    backend: str = "local"

    #: Face restoration and colourisation reconstruct and invent. They stay off
    #: unless the operator turns them on, exactly as they do in the CLI.
    allow_generative: bool = False

    #: A directory of static files to serve at ``/``. Optional: the hosted page
    #: works without it, and this is for running the whole thing offline.
    ui_dir: str | None = None

    #: How many pages one job may hold, and how large they may be. A local
    #: appliance still should not fall over because a folder had 40,000 files
    #: in it, and a bound the client can read makes a good error message.
    max_files: int = 500
    max_file_bytes: int = 128 * 1024 * 1024
    max_total_bytes: int = 4 * 1024 * 1024 * 1024

    #: How long a finished job's files stay on disk before being swept.
    job_ttl_seconds: float = 60 * 60.0

    #: Worker threads inside one run, matching the CLI's ``--jobs``.
    jobs: int = 1

    _allowed: frozenset[str] = field(init=False, repr=False, default=frozenset())

    def __post_init__(self) -> None:
        # 0 is allowed and means "any free port", which is how the tests bind
        # without racing each other for a fixed one.
        if not 0 <= int(self.port) <= 65535:
            raise ServeConfigError(f"port {self.port} is outside 0..65535")
        self.port = int(self.port)
        self.jobs = max(1, int(self.jobs))

        self._allowed = frozenset(_origin_of(o) for o in self.allow_origins if o.strip())

        if self.max_files < 1:
            raise ServeConfigError("--max-files must be at least 1")
        if self.max_file_bytes < 1 or self.max_total_bytes < 1:
            raise ServeConfigError("size limits must be positive")

    # ------------------------------------------------------------------
    # Questions a request handler asks
    # ------------------------------------------------------------------

    @property
    def uploads_photographs(self) -> bool:
        """True when the configured backend sends images to a third party."""
        return self.backend != "local"

    @property
    def bound_to_loopback(self) -> bool:
        return self.host in LOOPBACK_HOSTS

    def origin_allowed(self, origin: str | None) -> bool:
        """Whether a browser at ``origin`` may drive this server.

        A missing ``Origin`` is allowed: same-origin navigations and plain
        ``GET``s from the page this server itself serves do not send one, and
        no browser omits it on a cross-origin request.
        """
        if not origin or origin == "null":
            return True
        candidate = origin.strip().rstrip("/")
        if candidate in self._allowed:
            return True
        parts = urlsplit(candidate)
        return parts.scheme in ("http", "https") and (parts.hostname or "").lower() in (
            "localhost",
            "127.0.0.1",
            "::1",
        )

    def host_allowed(self, host_header: str | None) -> bool:
        """Whether a ``Host`` header may be trusted, blocking DNS rebinding."""
        if not self.bound_to_loopback:
            # Bound somewhere reachable on purpose; the operator owns the name.
            return True
        return is_loopback_host(host_header or "")

    # ------------------------------------------------------------------

    @property
    def base_url(self) -> str:
        host = "localhost" if self.host in ("0.0.0.0", "::", "127.0.0.1") else self.host
        return f"http://{host}:{self.port}"

    def describe(self) -> str:
        """One line for the terminal, so the operator sees what they enabled."""
        where = "uploads images" if self.uploads_photographs else "local only"
        gen = "allowed" if self.allow_generative else "refused"
        return (
            f"backend: {self.backend} ({where}) | "
            f"faces and colourisation: {gen} | "
            f"limits: {self.max_files} files, "
            f"{self.max_total_bytes // (1024 * 1024)} MB total"
        )

    def public_limits(self) -> dict:
        """The bounds a client is entitled to know before it uploads anything."""
        return {
            "maxFiles": self.max_files,
            "maxFileBytes": self.max_file_bytes,
            "maxTotalBytes": self.max_total_bytes,
        }
