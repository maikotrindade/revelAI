"""One run of RevelAI, driven from a browser instead of a terminal.

A job owns a temporary directory, the files uploaded into it, the worker thread
that runs the two stages, an ordered log of progress events, and the zip that
comes out. Nothing here reaches outside that directory: the server never reads
a path a request gave it, because a request never gives it one. The browser
hands over bytes, and the bytes are written to names this module chose.

Progress is a log rather than a snapshot. Every change appends a numbered event,
so a client that connects late, or reconnects after its stream dropped, asks for
everything after the last sequence number it saw and misses nothing. The live
stream and the polling fallback are then the same data read two ways.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from revelai import PRODUCT_NAME, RevelAIError, __version__

__all__ = [
    "Job",
    "JobStore",
    "JobError",
    "JobCancelled",
    "STAGES",
    "stage_sequence",
]

#: What a job may be asked to do, mirroring the three CLI commands.
STAGES: tuple[str, ...] = ("split", "enhance", "run")

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

#: Sub-directories inside a job. Named here so nothing else invents one.
_INPUT = "input"
_SPLIT_OUT = "photos"
_ENHANCE_OUT = "photos_restored"
_RESULT = "revelai-photos.zip"


class JobError(RevelAIError):
    """The job cannot do what was asked of it."""


class JobCancelled(Exception):
    """Raised inside a worker to unwind a run the user discarded.

    Deliberately not a ``RevelAIError``: it is control flow, not a failure, and
    it must not be reported to the user as one.
    """


def stage_sequence(stage: str) -> tuple[str, ...]:
    """The stages a request for ``stage`` actually runs, in order."""
    if stage == "run":
        return ("split", "enhance")
    if stage in ("split", "enhance"):
        return (stage,)
    raise JobError(f"unknown stage {stage!r}; expected one of {', '.join(STAGES)}")


@dataclass
class _Event:
    seq: int
    type: str
    payload: dict

    def as_dict(self) -> dict:
        return {"seq": self.seq, "type": self.type, **self.payload}


@dataclass
class Job:
    """A single run: its files, its progress, and its result."""

    id: str
    root: Path
    stage: str
    options: dict
    expected: list[dict]

    created_at: float = field(default_factory=time.monotonic)
    state: str = "collecting"
    error: str | None = None
    summary: dict = field(default_factory=dict)

    _events: list[_Event] = field(default_factory=list, repr=False)
    _changed: threading.Condition = field(default_factory=threading.Condition, repr=False)
    _received: dict[int, str] = field(default_factory=dict, repr=False)
    _cancelled: bool = False

    # ------------------------------------------------------------------
    # Directories
    # ------------------------------------------------------------------

    @property
    def input_dir(self) -> Path:
        return self.root / _INPUT

    @property
    def split_dir(self) -> Path:
        return self.root / _SPLIT_OUT

    @property
    def enhance_dir(self) -> Path:
        return self.root / _ENHANCE_OUT

    @property
    def result_path(self) -> Path:
        return self.root / _RESULT

    @property
    def stages(self) -> tuple[str, ...]:
        return stage_sequence(self.stage)

    @property
    def finished(self) -> bool:
        return self.state in ("done", "failed", "cancelled")

    # ------------------------------------------------------------------
    # Uploads
    # ------------------------------------------------------------------

    def accept_file(self, index: int, data: bytes) -> str:
        """Store one uploaded page, after proving it decodes as an image.

        The index refers to the listing this job was created from, so the name
        written is the one already validated at creation. A request cannot
        introduce a new name here, which is why no path handling is needed.
        """
        from revelai.server.folder import verify_image_bytes

        if self.state != "collecting":
            raise JobError(f"this run is {self.state}; it is no longer taking files")
        if not 0 <= index < len(self.expected):
            raise JobError(f"there is no file {index} in this run")

        entry = self.expected[index]
        problem = verify_image_bytes(data)
        if problem is not None:
            raise JobError(f"{entry['name']} {problem}")

        self.input_dir.mkdir(parents=True, exist_ok=True)
        target = self.input_dir / entry["name"]
        target.write_bytes(data)
        with self._changed:
            self._received[index] = entry["name"]
            self._changed.notify_all()
        return entry["name"]

    @property
    def received(self) -> int:
        return len(self._received)

    @property
    def complete(self) -> bool:
        return self.received == len(self.expected)

    def missing(self) -> list[str]:
        return [e["name"] for i, e in enumerate(self.expected) if i not in self._received]

    # ------------------------------------------------------------------
    # The event log
    # ------------------------------------------------------------------

    def emit(self, type_: str, **payload) -> None:
        with self._changed:
            event = _Event(seq=len(self._events) + 1, type=type_, payload=payload)
            self._events.append(event)
            self._changed.notify_all()

    def events_after(self, seq: int) -> list[dict]:
        with self._changed:
            return [e.as_dict() for e in self._events if e.seq > seq]

    @property
    def last_seq(self) -> int:
        with self._changed:
            return self._events[-1].seq if self._events else 0

    def wait_for_events(self, seq: int, timeout: float) -> list[dict]:
        """Block until there is something after ``seq``, or ``timeout`` passes.

        An empty list means "nothing yet" rather than "nothing ever", so the
        caller sends a keep-alive and asks again. That is what keeps a stream
        open through a long page without inventing progress that has not
        happened.
        """
        deadline = time.monotonic() + timeout
        with self._changed:
            while True:
                fresh = [e.as_dict() for e in self._events if e.seq > seq]
                if fresh or self._cancelled:
                    return fresh
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return []
                self._changed.wait(remaining)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def set_state(self, state: str, **payload) -> None:
        self.state = state
        self.emit("state", state=state, **payload)

    def cancel(self) -> None:
        with self._changed:
            self._cancelled = True
            self._changed.notify_all()

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def raise_if_cancelled(self) -> None:
        if self._cancelled:
            raise JobCancelled

    def snapshot(self) -> dict:
        return {
            "id": self.id,
            "state": self.state,
            "stage": self.stage,
            "stages": list(self.stages),
            "files": len(self.expected),
            "received": self.received,
            "error": self.error,
            "summary": self.summary or None,
            "lastSeq": self.last_seq,
            "resultReady": self.state == "done" and self.result_path.exists(),
        }


# --------------------------------------------------------------------------
# Running the stages
# --------------------------------------------------------------------------


class _Progress:
    """Turns per-page callbacks into the percentage a progress bar wants.

    Two stages have to share one bar, and the second stage's total is not known
    until the first has finished - splitting eight pages might yield thirty
    photographs or none. So the overall figure is the fraction of stages done
    plus the fraction of the current one, which is honest at every moment
    rather than accurate only in hindsight.
    """

    def __init__(self, job: Job) -> None:
        self.job = job
        self.stages = job.stages
        self.index = 0
        self.total = 0
        self.done = 0

    def begin(self, stage: str, total: int, unit: str) -> None:
        self.index = self.stages.index(stage)
        self.total = max(0, int(total))
        self.done = 0
        self.job.emit(
            "progress",
            stage=stage,
            stageIndex=self.index,
            stageCount=len(self.stages),
            done=0,
            total=self.total,
            unit=unit,
            percent=self._percent(),
            current=None,
            message=f"{self.total} {unit}{'s' if self.total != 1 else ''} to go",
        )

    def step(self, *, current: str, message: str, unit: str, **extra) -> None:
        self.done += 1
        self.job.emit(
            "progress",
            stage=self.stages[self.index],
            stageIndex=self.index,
            stageCount=len(self.stages),
            done=self.done,
            total=self.total,
            unit=unit,
            percent=self._percent(),
            current=current,
            message=message,
            **extra,
        )

    def _percent(self) -> float:
        fraction = (self.done / self.total) if self.total else 1.0
        return round(100.0 * (self.index + fraction) / len(self.stages), 1)


def _split_options(options: dict):
    from revelai.split import SplitOptions

    return SplitOptions(
        min_area=float(options.get("minArea", 0.005)),
        max_area=float(options.get("maxArea", 0.9)),
        inset=int(options.get("inset", 3)),
    )


def _enhance_options(options: dict):
    from revelai.enhance import EnhanceOptions

    return EnhanceOptions(
        color=bool(options.get("color", True)),
        color_strength=float(options.get("colorStrength", 1.0)),
        denoise=bool(options.get("denoise", False)),
        dust=bool(options.get("dust", False)),
        upscale=int(options.get("upscale", 0) or 0),
        faces=bool(options.get("faces", False)),
        colorize=bool(options.get("colorize", False)),
    )


def _run_split_stage(job: Job, progress: _Progress, output: Path, jobs: int) -> dict:
    from revelai.io import iter_input_images
    from revelai.split import run_split

    sources = iter_input_images(job.input_dir)
    if not sources:
        raise JobError("no readable images arrived; nothing to split")

    progress.begin("split", len(sources), "page")

    def on_page(page) -> None:
        job.raise_if_cancelled()
        if page.error:
            message = f"{page.source.name}: {page.error}"
        else:
            count = page.detections
            message = f"{page.source.name}: {count} photograph{'s' if count != 1 else ''}" + (
                "  needs review" if page.needs_review else ""
            )
        progress.step(
            current=page.source.name,
            message=message,
            unit="page",
            photographs=page.detections,
            needsReview=bool(page.needs_review),
            failed=bool(page.error),
        )

    report = run_split(
        sources,
        output,
        _split_options(job.options),
        jobs=jobs,
        on_page=on_page,
    )
    job.raise_if_cancelled()

    return {
        "pagesRead": len(report.pages),
        "photographs": report.photographs,
        "pagesNeedingReview": [
            {"name": p.source.name, "reasons": list(dict.fromkeys(p.notes))}
            for p in report.needing_review
        ],
        "pagesFailed": [{"name": p.source.name, "reason": p.error} for p in report.failed],
    }


def _run_enhance_stage(
    job: Job, progress: _Progress, source: Path, output: Path, backend_name: str, jobs: int
) -> dict:
    from revelai.enhance import run_enhance
    from revelai.enhance.backends import get_backend
    from revelai.io import iter_input_images

    photos = iter_input_images(source)
    if not photos:
        raise JobError("there are no photographs to restore")

    options = _enhance_options(job.options)
    backend = get_backend(backend_name)
    progress.begin("enhance", len(photos), "photograph")

    def on_photo(photo) -> None:
        job.raise_if_cancelled()
        if photo.error:
            message = f"{photo.source.name}: {photo.error}"
        else:
            applied = ", ".join(str(op) for op in photo.operations) or "nothing to do"
            message = f"{photo.source.name}: {applied}"
        progress.step(
            current=photo.source.name,
            message=message,
            unit="photograph",
            failed=bool(photo.error),
        )

    report = run_enhance(
        photos,
        output,
        options,
        backend,
        jobs=jobs,
        on_photo=on_photo,
    )
    job.raise_if_cancelled()

    return {
        "restored": report.restored,
        "backend": backend.name,
        "operations": options.requested(),
        "operationsSkipped": [
            {"name": name, "reason": reason} for name, reason in report.skipped_operations.items()
        ],
        "photosFailed": [{"name": p.source.name, "reason": p.error} for p in report.failed],
    }


def build_result_zip(source: Path, destination: Path) -> dict:
    """Zip the photographs in ``source``, and nothing else.

    Two rules are enforced here rather than assumed. Only files RevelAI wrote
    go in - checked against the PNG signature, not the name - and every entry
    is stored with a ``.png`` suffix because every byte RevelAI writes is a
    PNG, whatever the input was called. A zip that claimed to hold JPEGs and
    held PNGs would be a small lie that breaks somebody's importer later.

    Stored, not deflated: a PNG is already compressed, so deflating it again
    buys under a per cent and costs the whole archive's worth of CPU.
    """
    entries: list[str] = []
    total = 0
    destination.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in source.iterdir() if p.is_file()) if source.is_dir() else []
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in files:
            data = path.read_bytes()
            if not data.startswith(_PNG_SIGNATURE):
                continue
            name = path.name if path.suffix.lower() == ".png" else f"{path.stem}.png"
            archive.writestr(name, data)
            entries.append(name)
            total += len(data)

    if not entries:
        destination.unlink(missing_ok=True)
        raise JobError("the run produced no photographs, so there is nothing to download")

    return {
        "file": destination.name,
        "images": len(entries),
        "names": entries,
        "bytesUncompressed": total,
        "bytes": destination.stat().st_size,
    }


def run_job(job: Job, *, backend: str, jobs: int) -> None:
    """Execute a job to completion. Runs on the job's own worker thread."""
    started = time.monotonic()
    summary: dict = {"stages": list(job.stages)}
    progress = _Progress(job)

    try:
        job.set_state("running")
        job.raise_if_cancelled()

        final_dir = job.input_dir
        if "split" in job.stages:
            summary["split"] = _run_split_stage(job, progress, job.split_dir, jobs)
            final_dir = job.split_dir
        if "enhance" in job.stages:
            summary["enhance"] = _run_enhance_stage(
                job, progress, final_dir, job.enhance_dir, backend, jobs
            )
            final_dir = job.enhance_dir

        summary["result"] = build_result_zip(final_dir, job.result_path)
        summary["seconds"] = round(time.monotonic() - started, 2)
        summary["product"] = f"{PRODUCT_NAME} {__version__}"

        job.summary = summary
        job.set_state("done")
        job.emit("done", summary=summary)

    except JobCancelled:
        job.state = "cancelled"
        job.emit("state", state="cancelled")
    except RevelAIError as exc:
        job.error = str(exc)
        job.state = "failed"
        job.emit("state", state="failed", error=job.error)
        job.emit("error", error=job.error)
    except Exception as exc:  # noqa: BLE001 - a failed run is a message, not a traceback
        job.error = f"the run stopped: {exc}"
        job.state = "failed"
        job.emit("state", state="failed", error=job.error)
        job.emit("error", error=job.error)


# --------------------------------------------------------------------------
# The store
# --------------------------------------------------------------------------


class JobStore:
    """Every job this server knows about, and the temporary space they own."""

    def __init__(self, *, ttl_seconds: float = 3600.0) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds
        self._base = Path(tempfile.mkdtemp(prefix="revelai-serve-"))

    @property
    def base(self) -> Path:
        return self._base

    def create(self, *, stage: str, options: dict, expected: list[dict]) -> Job:
        stage_sequence(stage)  # validates, raising JobError on a bad stage
        job_id = uuid.uuid4().hex
        root = self._base / job_id
        (root / _INPUT).mkdir(parents=True, exist_ok=True)
        job = Job(id=job_id, root=root, stage=stage, options=dict(options), expected=expected)
        job.emit("state", state="collecting", files=len(expected))
        with self._lock:
            self._jobs[job_id] = job
        self.sweep()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def start(self, job: Job, *, backend: str, jobs: int) -> None:
        if job.state != "collecting":
            raise JobError(f"this run is already {job.state}")
        if not job.complete:
            missing = ", ".join(job.missing()[:5])
            raise JobError(f"{len(job.missing())} file(s) never arrived: {missing}")
        job.set_state("queued")
        threading.Thread(
            target=run_job,
            args=(job,),
            kwargs={"backend": backend, "jobs": jobs},
            name=f"revelai-job-{job.id[:8]}",
            daemon=True,
        ).start()

    def discard(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.pop(job_id, None)
        if job is None:
            return False
        job.cancel()
        # Do not join the worker: a discard has to answer immediately. The
        # thread notices the flag at its next page boundary and unwinds, and
        # the directory below is removed whether or not it has stopped yet.
        shutil.rmtree(job.root, ignore_errors=True)
        return True

    def sweep(self) -> int:
        """Remove jobs whose time is up. Called whenever a new one is made."""
        now = time.monotonic()
        stale = []
        with self._lock:
            for job_id, job in list(self._jobs.items()):
                if now - job.created_at > self._ttl:
                    stale.append(self._jobs.pop(job_id))
        for job in stale:
            job.cancel()
            shutil.rmtree(job.root, ignore_errors=True)
        return len(stale)

    def close(self) -> None:
        """Cancel everything and take the temporary directory with it."""
        with self._lock:
            jobs = list(self._jobs.values())
            self._jobs.clear()
        for job in jobs:
            job.cancel()
        shutil.rmtree(self._base, ignore_errors=True)

    def __len__(self) -> int:
        with self._lock:
            return len(self._jobs)
