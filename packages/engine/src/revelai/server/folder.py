"""Deciding whether a chosen folder is one RevelAI will accept.

The check happens twice on purpose, and the two halves answer different
questions.

*Before* anything is uploaded, the client sends a listing - names and sizes -
and this module says whether the folder looks right. That is what turns "the
run failed on file 34 of 80" into "this folder has a PDF and two subfolders in
it, here they are", before a single byte has moved.

*After* each file arrives, :func:`verify_image_bytes` decides whether it really
is an image, by decoding it. A name is a claim; the bytes are the evidence. A
folder that passed the listing check can still contain ``holiday.jpg`` that is
actually a spreadsheet, and only the decoder knows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from revelai.io import SUPPORTED_SUFFIXES

__all__ = [
    "FolderProblem",
    "FolderCheck",
    "check_listing",
    "safe_name",
    "verify_image_bytes",
    "IGNORED_NAMES",
]

#: Files every desktop scatters through a folder. Their presence is not the
#: user doing anything wrong, so they are skipped rather than reported: a
#: folder of photographs is still a folder of photographs with a .DS_Store in
#: it, and refusing it over that would be pedantry rather than a safeguard.
IGNORED_NAMES: frozenset[str] = frozenset(
    {".ds_store", "thumbs.db", "desktop.ini", ".picasa.ini", "picasa.ini", ".directory"}
)

_MAX_NAME = 255


@dataclass(frozen=True)
class FolderProblem:
    """One reason a folder was not accepted, in words a person can act on."""

    name: str
    reason: str

    def as_dict(self) -> dict:
        return {"name": self.name, "reason": self.reason}


@dataclass
class FolderCheck:
    """The verdict on a folder listing."""

    accepted: list[dict] = field(default_factory=list)
    skipped: list[FolderProblem] = field(default_factory=list)
    problems: list[FolderProblem] = field(default_factory=list)
    total_bytes: int = 0

    @property
    def ok(self) -> bool:
        return not self.problems and bool(self.accepted)

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "accepted": self.accepted,
            "skipped": [p.as_dict() for p in self.skipped],
            "problems": [p.as_dict() for p in self.problems],
            "totalBytes": self.total_bytes,
        }


def safe_name(raw: str) -> str | None:
    """The bare filename of ``raw``, or None if it is not one we will write.

    Names arrive from a browser and are used to build a path, so this is a
    security boundary rather than tidying. Anything with a separator, a drive
    letter, a ``..`` segment, a NUL or a control character is refused outright
    instead of being sanitised into something that looks fine and is not.
    """
    name = (raw or "").strip()
    if not name or len(name) > _MAX_NAME:
        return None
    if name in (".", ".."):
        return None
    if any(ch in name for ch in ("/", "\\", "\x00")):
        return None
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in name):
        return None
    if ":" in name:  # C:file, and NTFS alternate data streams
        return None
    return name


def _suffix(name: str) -> str:
    _, dot, ext = name.rpartition(".")
    return f".{ext.lower()}" if dot else ""


def check_listing(
    entries,
    *,
    max_files: int,
    max_file_bytes: int,
    max_total_bytes: int,
) -> FolderCheck:
    """Check a folder listing before anything is uploaded.

    ``entries`` is a sequence of mappings with ``name`` and ``size``. A
    ``relativePath`` may be present; when it names a subfolder the entry is
    refused, because a folder of album pages is flat and quietly flattening a
    tree would produce collisions the user never asked for.
    """
    check = FolderCheck()
    seen: set[str] = set()

    listing = list(entries or [])
    if not listing:
        check.problems.append(FolderProblem("", "the folder is empty"))
        return check

    for raw in listing:
        if not isinstance(raw, dict):
            check.problems.append(FolderProblem("", "malformed entry in the listing"))
            continue

        original = str(raw.get("name", ""))
        relative = str(raw.get("relativePath") or original)

        if original.lower() in IGNORED_NAMES or original.startswith("._"):
            check.skipped.append(FolderProblem(original, "a system file; ignored"))
            continue

        name = safe_name(original)
        if name is None:
            check.problems.append(
                FolderProblem(original or "(unnamed)", "the name is not one RevelAI will write")
            )
            continue

        # relativePath is "folder/sub/file.jpg" when the picker walked a tree.
        depth = relative.replace("\\", "/").strip("/").count("/")
        if depth > 1:
            check.problems.append(
                FolderProblem(
                    relative,
                    "it is inside a subfolder; choose a folder that holds the pages directly",
                )
            )
            continue

        suffix = _suffix(name)
        if suffix not in SUPPORTED_SUFFIXES:
            described = suffix or "no extension"
            check.problems.append(FolderProblem(name, f"not an image RevelAI reads ({described})"))
            continue

        key = name.lower()
        if key in seen:
            check.problems.append(FolderProblem(name, "two files in this folder share a name"))
            continue
        seen.add(key)

        try:
            size = int(raw.get("size", 0))
        except (TypeError, ValueError):
            check.problems.append(FolderProblem(name, "the size is not a number"))
            continue
        if size <= 0:
            check.problems.append(FolderProblem(name, "the file is empty"))
            continue
        if size > max_file_bytes:
            check.problems.append(
                FolderProblem(name, f"larger than the {_mb(max_file_bytes)} MB limit for one file")
            )
            continue

        check.total_bytes += size
        check.accepted.append({"name": name, "size": size})

    if not check.problems:
        if not check.accepted:
            check.problems.append(FolderProblem("", "the folder holds no images"))
        elif len(check.accepted) > max_files:
            check.problems.append(
                FolderProblem(
                    "",
                    f"{len(check.accepted)} images, which is over the limit of {max_files}; "
                    f"run it in batches, or raise --max-files",
                )
            )
        elif check.total_bytes > max_total_bytes:
            check.problems.append(
                FolderProblem(
                    "",
                    f"{_mb(check.total_bytes)} MB in total, which is over the "
                    f"limit of {_mb(max_total_bytes)} MB",
                )
            )

    return check


def _mb(value: int) -> int:
    return max(1, int(value // (1024 * 1024)))


def verify_image_bytes(data: bytes) -> str | None:
    """None when ``data`` decodes as an image, otherwise why it does not.

    This is the check that actually matters. The listing check reads names; a
    name is whatever the file was called. Decoding is the only thing that
    establishes that a folder contains images and nothing else, which is the
    promise this server makes before it runs anything.
    """
    if not data:
        return "the file is empty"

    import cv2
    import numpy as np

    try:
        pixels = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    except cv2.error as exc:  # pragma: no cover - malformed beyond decoding
        return f"could not be decoded as an image ({exc.err or 'unsupported data'})"
    if pixels is None:
        return "could not be decoded as an image; it is not one, or it is corrupt"
    if pixels.ndim == 3 and pixels.shape[2] not in (3, 4):
        return f"has {pixels.shape[2]} channels, which RevelAI does not support"
    if min(pixels.shape[:2]) < 2:
        return "is too small to be a photograph of a page"
    return None
