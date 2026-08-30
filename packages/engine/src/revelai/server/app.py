"""The HTTP surface of ``revelai serve``.

Written against the standard library on purpose. RevelAI's dependency list is
numpy, OpenCV and Pillow, and a local convenience should not be the thing that
adds a web framework to a photo tool - or that stops ``revelai serve`` working
from the container image and a bare ``pip install revelai``.

Three things about the shape of this API are deliberate.

*Uploads are one file per request*, raw bytes with the name in the URL, rather
than one multipart form. That removes a multipart parser from the trusted path,
gives the browser accurate per-file upload progress for free, and means a
failure names the file that failed.

*Progress is server-sent events*, with a plain polling endpoint carrying exactly
the same numbered log for anything that cannot hold a stream open.

*Every guard is at the front door.* Host and Origin are checked on every request
before the route is even looked at, because a local server that any page on the
internet can drive is a worse thing to have running than no local server.
"""

from __future__ import annotations

import json
import mimetypes
import re
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from revelai import PRODUCT_NAME, __version__
from revelai.io import SUPPORTED_SUFFIXES
from revelai.server.config import ServeConfig
from revelai.server.folder import check_listing
from revelai.server.jobs import JobError, JobStore, stage_sequence

__all__ = ["RevelAIHandler", "build_server"]

_JOB_ID = r"(?P<job>[0-9a-f]{32})"
_ROUTES = {
    "job": re.compile(rf"^/api/jobs/{_JOB_ID}$"),
    "file": re.compile(rf"^/api/jobs/{_JOB_ID}/files/(?P<index>\d+)$"),
    "start": re.compile(rf"^/api/jobs/{_JOB_ID}/start$"),
    "events": re.compile(rf"^/api/jobs/{_JOB_ID}/events$"),
    "result": re.compile(rf"^/api/jobs/{_JOB_ID}/result\.zip$"),
}

#: How long a stream waits for something to happen before sending a keep-alive.
_HEARTBEAT_SECONDS = 15.0

#: A JSON body is a listing or a set of options. Neither is large, and a cap
#: here means a malformed Content-Length cannot ask us to allocate a gigabyte.
_MAX_JSON_BYTES = 4 * 1024 * 1024


class RevelAIHandler(BaseHTTPRequestHandler):
    """One request. The config and the job store are shared, and both are safe
    to touch from several threads."""

    protocol_version = "HTTP/1.1"
    server_version = f"{PRODUCT_NAME}/{__version__}"
    sys_version = ""

    def __init__(self, *args, config: ServeConfig, store: JobStore, log=None, **kwargs) -> None:
        self.config = config
        self.store = store
        self._log = log
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Replies
    # ------------------------------------------------------------------

    def _cors(self, *, preflight: bool = False) -> None:
        origin = self.headers.get("Origin")
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        if preflight:
            self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Max-Age", "600")
            # Chrome's Private Network Access check: a page on the public web
            # asking to reach a server on the loopback interface has to be told
            # in the preflight that the server expects it.
            if self.headers.get("Access-Control-Request-Private-Network") == "true":
                self.send_header("Access-Control-Allow-Private-Network", "true")

    def _common(self) -> None:
        # This server answers from a temporary directory that is gone shortly
        # after; nothing it says should ever be cached.
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")

    def send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._common()
        self._cors()
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def send_error_json(self, status: int, message: str, **extra) -> None:
        self.send_json(status, {"error": message, **extra})

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def _content_length(self, limit: int) -> int | None:
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            self.send_error_json(
                HTTPStatus.LENGTH_REQUIRED,
                "send the body with a Content-Length; this server does not read chunked bodies",
            )
            return None
        raw = self.headers.get("Content-Length")
        if raw is None:
            self.send_error_json(HTTPStatus.LENGTH_REQUIRED, "Content-Length is required")
            return None
        try:
            length = int(raw)
        except ValueError:
            self.send_error_json(HTTPStatus.BAD_REQUEST, "Content-Length is not a number")
            return None
        if length < 0:
            self.send_error_json(HTTPStatus.BAD_REQUEST, "Content-Length is negative")
            return None
        if length > limit:
            self.send_error_json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                f"the body is {length} bytes, over this server's limit of {limit}",
            )
            return None
        return length

    def _read_body(self, limit: int) -> bytes | None:
        length = self._content_length(limit)
        if length is None:
            return None
        data = b""
        while len(data) < length:
            chunk = self.rfile.read(min(1 << 20, length - len(data)))
            if not chunk:
                break
            data += chunk
        if len(data) != length:
            self.send_error_json(HTTPStatus.BAD_REQUEST, "the body ended early")
            return None
        return data

    def _read_json(self) -> dict | None:
        body = self._read_body(_MAX_JSON_BYTES)
        if body is None:
            return None
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self.send_error_json(HTTPStatus.BAD_REQUEST, f"the body is not valid JSON: {exc}")
            return None
        if not isinstance(payload, dict):
            self.send_error_json(HTTPStatus.BAD_REQUEST, "expected a JSON object")
            return None
        return payload

    # ------------------------------------------------------------------
    # The front door
    # ------------------------------------------------------------------

    def _guard(self) -> bool:
        """Host and Origin, checked before anything else looks at the path."""
        if not self.config.host_allowed(self.headers.get("Host")):
            self.send_error_json(
                HTTPStatus.MISDIRECTED_REQUEST,
                "this server answers on localhost only. A request arrived under "
                "another name, which is how DNS rebinding works, so it was refused.",
            )
            return False
        origin = self.headers.get("Origin")
        if not self.config.origin_allowed(origin):
            self.send_error_json(
                HTTPStatus.FORBIDDEN,
                f"{origin} is not allowed to drive this server. Start it with "
                f"--allow-origin {origin} if that is really where your page is.",
                code="origin_not_allowed",
            )
            return False
        return True

    def do_OPTIONS(self) -> None:  # noqa: N802 - the stdlib names the method
        if not self._guard():
            return
        self.send_response(HTTPStatus.NO_CONTENT)
        self._common()
        self._cors(preflight=True)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if not self._guard():
            return
        path = self.path.split("?", 1)[0]
        query = self.path.split("?", 1)[1] if "?" in self.path else ""

        if path == "/api/health":
            return self._health()

        match = _ROUTES["events"].match(path)
        if match:
            return self._events(match.group("job"), query)
        match = _ROUTES["result"].match(path)
        if match:
            return self._result(match.group("job"))
        match = _ROUTES["job"].match(path)
        if match:
            return self._job_state(match.group("job"))

        if path.startswith("/api/"):
            return self.send_error_json(HTTPStatus.NOT_FOUND, f"no route for GET {path}")
        return self._static(path)

    do_HEAD = do_GET

    def do_POST(self) -> None:  # noqa: N802
        if not self._guard():
            return
        path = self.path.split("?", 1)[0]
        if path == "/api/folder/check":
            return self._folder_check()
        if path == "/api/jobs":
            return self._create_job()
        match = _ROUTES["start"].match(path)
        if match:
            return self._start_job(match.group("job"))
        return self.send_error_json(HTTPStatus.NOT_FOUND, f"no route for POST {path}")

    def do_PUT(self) -> None:  # noqa: N802
        if not self._guard():
            return
        path = self.path.split("?", 1)[0]
        match = _ROUTES["file"].match(path)
        if match:
            return self._upload(match.group("job"), int(match.group("index")))
        return self.send_error_json(HTTPStatus.NOT_FOUND, f"no route for PUT {path}")

    def do_DELETE(self) -> None:  # noqa: N802
        if not self._guard():
            return
        match = _ROUTES["job"].match(self.path.split("?", 1)[0])
        if not match:
            return self.send_error_json(HTTPStatus.NOT_FOUND, "no route for DELETE")
        removed = self.store.discard(match.group("job"))
        return self.send_json(HTTPStatus.OK, {"discarded": removed})

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    def _health(self) -> None:
        """What this server is and what it will let you ask for.

        The client reads this before showing a single control, so anything the
        UI must not offer - a hosted backend, face restoration - is absent from
        the interface rather than refused after the user has chosen it.
        """
        self.send_json(
            HTTPStatus.OK,
            {
                "ok": True,
                "product": PRODUCT_NAME,
                "version": __version__,
                "stages": ["split", "enhance", "run"],
                "backend": self.config.backend,
                "uploadsPhotographs": self.config.uploads_photographs,
                "allowGenerative": self.config.allow_generative,
                "limits": self.config.public_limits(),
                "acceptedSuffixes": sorted(SUPPORTED_SUFFIXES),
                "jobs": len(self.store),
            },
        )

    def _check(self, entries) -> dict:
        return check_listing(
            entries,
            max_files=self.config.max_files,
            max_file_bytes=self.config.max_file_bytes,
            max_total_bytes=self.config.max_total_bytes,
        ).as_dict()

    def _folder_check(self) -> None:
        """Say whether a folder is acceptable, before a byte is uploaded."""
        payload = self._read_json()
        if payload is None:
            return
        result = self._check(payload.get("files"))
        self.send_json(HTTPStatus.OK, result)

    def _refuse_generative(self, options: dict) -> str | None:
        if self.config.allow_generative:
            return None
        asked = [name for name in ("faces", "colorize") if options.get(name)]
        if not asked:
            return None
        return (
            f"{' and '.join(asked)} reconstruct and invent detail that was not in "
            f"the photograph, so this server refuses them unless it was started "
            f"with --allow-generative."
        )

    def _create_job(self) -> None:
        payload = self._read_json()
        if payload is None:
            return

        stage = str(payload.get("stage", "split"))
        try:
            stage_sequence(stage)
        except JobError as exc:
            return self.send_error_json(HTTPStatus.BAD_REQUEST, str(exc))

        options = payload.get("options") or {}
        if not isinstance(options, dict):
            return self.send_error_json(HTTPStatus.BAD_REQUEST, "options must be an object")
        if "backend" in options:
            return self.send_error_json(
                HTTPStatus.BAD_REQUEST,
                "the backend is chosen when the server starts, not by a request. "
                "Whether your photographs leave this machine is not something a "
                "web page gets to decide.",
                code="backend_not_selectable",
            )
        refusal = self._refuse_generative(options)
        if refusal:
            return self.send_error_json(HTTPStatus.FORBIDDEN, refusal, code="generative_refused")

        check = self._check(payload.get("files"))
        if not check["ok"]:
            return self.send_json(
                HTTPStatus.BAD_REQUEST,
                {
                    "error": "this folder is not one RevelAI will run on",
                    "code": "folder_rejected",
                    **check,
                },
            )

        try:
            job = self.store.create(stage=stage, options=options, expected=check["accepted"])
        except JobError as exc:
            return self.send_error_json(HTTPStatus.BAD_REQUEST, str(exc))

        self.log_line(f"job {job.id[:8]} created: {stage}, {len(check['accepted'])} files")
        self.send_json(
            HTTPStatus.CREATED,
            {
                "id": job.id,
                "stage": stage,
                "stages": list(job.stages),
                "files": check["accepted"],
                "skipped": check["skipped"],
                "uploadUrl": f"/api/jobs/{job.id}/files/{{index}}",
            },
        )

    def _upload(self, job_id: str, index: int) -> None:
        job = self.store.get(job_id)
        if job is None:
            return self.send_error_json(HTTPStatus.NOT_FOUND, "no such run")
        body = self._read_body(self.config.max_file_bytes)
        if body is None:
            return
        try:
            name = job.accept_file(index, body)
        except JobError as exc:
            return self.send_error_json(
                HTTPStatus.BAD_REQUEST, str(exc), code="not_an_image", index=index
            )
        self.send_json(
            HTTPStatus.OK,
            {"name": name, "index": index, "received": job.received, "expected": len(job.expected)},
        )

    def _start_job(self, job_id: str) -> None:
        job = self.store.get(job_id)
        if job is None:
            return self.send_error_json(HTTPStatus.NOT_FOUND, "no such run")
        try:
            self.store.start(job, backend=self.config.backend, jobs=self.config.jobs)
        except JobError as exc:
            return self.send_error_json(HTTPStatus.CONFLICT, str(exc))
        self.log_line(f"job {job.id[:8]} started: {job.stage}")
        self.send_json(HTTPStatus.ACCEPTED, job.snapshot())

    def _job_state(self, job_id: str) -> None:
        job = self.store.get(job_id)
        if job is None:
            return self.send_error_json(HTTPStatus.NOT_FOUND, "no such run")
        after = self._after_param()
        snapshot = job.snapshot()
        if after is not None:
            snapshot["events"] = job.events_after(after)
        self.send_json(HTTPStatus.OK, snapshot)

    def _after_param(self) -> int | None:
        if "?" not in self.path:
            return None
        from urllib.parse import parse_qs

        values = parse_qs(self.path.split("?", 1)[1]).get("after")
        if not values:
            return None
        try:
            return max(0, int(values[0]))
        except ValueError:
            return None

    def _events(self, job_id: str, query: str) -> None:
        """Stream the job's event log as server-sent events.

        The stream replays from ``?after=`` rather than starting from now, so a
        client that reconnects has no window in which it can miss a page.
        """
        job = self.store.get(job_id)
        if job is None:
            return self.send_error_json(HTTPStatus.NOT_FOUND, "no such run")

        from urllib.parse import parse_qs

        try:
            seq = max(0, int(parse_qs(query).get("after", ["0"])[0]))
        except ValueError:
            seq = 0

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self._cors()
        self.end_headers()
        self.close_connection = True

        try:
            while True:
                events = job.wait_for_events(seq, _HEARTBEAT_SECONDS)
                if not events:
                    if job.finished or self.store.get(job_id) is None:
                        break
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                    continue
                for event in events:
                    seq = event["seq"]
                    frame = (
                        f"id: {seq}\nevent: {event['type']}\ndata: {json.dumps(event)}\n\n"
                    ).encode()
                    self.wfile.write(frame)
                self.wfile.flush()
                if job.finished and seq >= job.last_seq:
                    break
        except (BrokenPipeError, ConnectionResetError):
            # The page navigated away mid-run. Not an error worth a line.
            pass

    def _result(self, job_id: str) -> None:
        job = self.store.get(job_id)
        if job is None:
            return self.send_error_json(HTTPStatus.NOT_FOUND, "no such run")
        if job.state != "done" or not job.result_path.exists():
            return self.send_error_json(
                HTTPStatus.CONFLICT,
                f"this run is {job.state}; there is nothing to download yet",
            )
        size = job.result_path.stat().st_size
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Length", str(size))
        self.send_header("Content-Disposition", f'attachment; filename="{job.result_path.name}"')
        self._common()
        self._cors()
        self.end_headers()
        if self.command == "HEAD":
            return
        # Streamed, not read into memory: a folder of fifty phone photographs
        # produces a zip large enough that holding a whole copy per download
        # would be a silly way to run out of memory.
        try:
            with job.result_path.open("rb") as handle:
                while chunk := handle.read(1 << 20):
                    self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    # ------------------------------------------------------------------
    # Optional static UI
    # ------------------------------------------------------------------

    def _static(self, path: str) -> None:
        """Serve ``--ui-dir``, so the whole thing can run with no internet.

        The resolved file has to still be inside the directory that was named.
        That is checked after resolution rather than by inspecting the URL,
        because a symlink is invisible to any amount of string cleaning.
        """
        if not self.config.ui_dir:
            return self.send_error_json(
                HTTPStatus.NOT_FOUND,
                "this server has no user interface of its own. Open the RevelAI "
                "run page in your browser, or start it with --ui-dir pointing at "
                "a built copy of the site.",
            )

        root = Path(self.config.ui_dir).resolve()
        relative = path.lstrip("/") or "index.html"
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return self.send_error_json(HTTPStatus.FORBIDDEN, "that path is outside the UI folder")

        if candidate.is_dir():
            candidate = candidate / "index.html"
        if not candidate.is_file():
            fallback = root / "404.html"
            if fallback.is_file():
                candidate = fallback
            else:
                return self.send_error_json(HTTPStatus.NOT_FOUND, f"{path} is not in the UI folder")

        data = candidate.read_bytes()
        kind = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self._common()
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    # ------------------------------------------------------------------

    def log_line(self, message: str) -> None:
        if self._log is not None:
            self._log(message)

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003 - stdlib name
        """Quieter than the default, which prints a line per request.

        A local server's terminal is where the operator learns what happened,
        and one line per uploaded file would bury the two that matter. Requests
        that succeeded are dropped; everything else - refusals, failures, and
        anything the stdlib logs as an error - is kept.
        """
        if self._log is None:
            return
        text = fmt % args
        if '" 2' in text or '" 3' in text:
            return
        self._log(text)


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request, client_address) -> None:
        """A page that navigated away mid-upload is not an error.

        The default prints a traceback per dropped connection, which during a
        long run is both alarming and wrong: the browser closing a stream is
        the normal end of a stream.
        """
        import sys

        kind = sys.exc_info()[0]
        if kind is not None and issubclass(kind, (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


def build_server(config: ServeConfig, *, log=None) -> tuple[_Server, JobStore]:
    """A configured server and its job store, not yet serving."""
    store = JobStore(ttl_seconds=config.job_ttl_seconds)
    handler = partial(RevelAIHandler, config=config, store=store, log=log)
    server = _Server((config.host, config.port), handler)
    return server, store
