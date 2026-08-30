"""Turning the engine's reports into JSON the model can reason about.

The engine already returns structured reports, so nothing here invents a shape.
What it does insist on is that the *inconvenient* parts survive: flagged crops,
skipped operations and per-file errors are as much a part of the result as the
count of files written. A summary that mentions only the successes would let an
assistant tell somebody their album came out fine when four crops are wrong.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["split_report", "enhance_report", "verdict", "description", "provenance"]


def _rel(path: Path | str | None, root: Path | None) -> str | None:
    """Paths as the caller gave them, shortened against the output folder."""
    if path is None:
        return None
    path = Path(path)
    if root is not None:
        try:
            return str(path.relative_to(root))
        except ValueError:
            pass
    return str(path)


def split_report(report: Any) -> dict:
    """Serialise a ``SplitReport``."""
    flagged = [
        {"file": name, "reasons": list(reasons)}
        for page in report.pages
        for name, reasons in page.flagged_crops
    ]
    written = [name for page in report.pages for name in page.written]
    pages = [
        {
            "source": page.source.name,
            "photographs": page.detections,
            "written": list(page.written),
            "needs_review": page.needs_review,
            "notes": list(dict.fromkeys(page.notes)),
            "flagged_crops": [
                {"file": name, "reasons": list(reasons)} for name, reasons in page.flagged_crops
            ],
            "error": page.error,
        }
        for page in report.pages
    ]
    return {
        "output_folder": str(report.output) if report.output else None,
        "dry_run": report.dry_run,
        "pages_read": len(report.pages),
        "photographs_found": report.photographs,
        "numbering": {"first": written[0], "last": written[-1]} if written else None,
        "pages_needing_review": [p.source.name for p in report.needing_review],
        "flagged_crops": flagged,
        "failed_pages": [{"source": p.source.name, "error": p.error} for p in report.failed],
        "pages": pages,
    }


def _operation(op: Any) -> dict:
    return {
        "name": op.name,
        "engine": op.engine,
        "version": op.version,
        "generative": op.generative,
    }


def enhance_report(report: Any) -> dict:
    """Serialise an ``EnhanceReport``."""
    return {
        "output_folder": str(report.output) if report.output else None,
        "dry_run": report.dry_run,
        "backend": report.backend,
        "photographs_read": len(report.photos),
        "restored": report.restored,
        "skipped_operations": [
            {"operation": name, "reason": reason}
            for name, reason in report.skipped_operations.items()
        ],
        "failed": [{"source": p.source.name, "error": p.error} for p in report.failed],
        "photographs": [
            {
                "file": photo.written or photo.source.name,
                "operations": [_operation(op) for op in photo.operations],
                "skipped": [
                    {"operation": name, "reason": reason} for name, reason in photo.skipped
                ],
                "error": photo.error,
            }
            for photo in report.photos
        ],
    }


def verdict(value: Any) -> dict:
    """Serialise a ``Verdict``.

    ``checked`` is first on purpose. A crop the model could not look at is not a
    crop that passed, and the difference has to be impossible to miss.
    """
    if not value.checked:
        return {
            "checked": False,
            "reason": value.unavailable,
            "note": "This crop was not verified. Do not report it as having passed.",
        }
    return {
        "checked": True,
        "ok": value.ok,
        "complete": value.complete,
        "cut_off_edges": list(value.cut_off_edges),
        "contains_multiple": value.contains_multiple,
        "contains_page_background": value.contains_page_background,
        "orientation_degrees_clockwise": value.orientation,
        "confidence": value.confidence,
        "problems": value.problems(),
        "note": value.note or None,
    }


def description(value: Any) -> dict:
    """Serialise a ``Description``."""
    if not value.checked:
        return {"checked": False, "reason": value.unavailable}
    return {
        "checked": True,
        "caption": value.caption,
        "tags": list(value.tags),
        "people_count": value.people_count,
        "estimated_decade": value.estimated_decade or None,
        "decade_confidence": value.decade_confidence,
        "date_stamp": value.date_stamp or None,
        "handwriting": value.handwriting or None,
        "is_black_and_white": value.is_black_and_white,
        "note": (
            "people_count is how many people are visible. RevelAI does not "
            "identify anyone, and the decade is an estimate from the image."
        ),
    }


def provenance(text: dict[str, str], path: Path) -> dict:
    """Serialise the RevelAI metadata found in a PNG."""
    ours = {k[len("revelai:") :]: v for k, v in text.items() if k.startswith("revelai:")}
    return {
        "file": path.name,
        "has_revelai_metadata": bool(ours),
        "metadata": ours,
        "other_text_chunks": sorted(k for k in text if not k.startswith("revelai:")),
    }
