"""Crop verification (specification 8.6).

This is the most valuable thing in the project, and it is worth being clear
about why. Detection is good but not perfect, and a batch of fifty album pages
produces two hundred crops that nobody is going to inspect one by one. Without a
check, the honest summary of such a run is "I processed fifty pages and I do not
know whether it came out right".

Asking a vision-language model one structured question per crop turns that into
"I processed fifty pages and four crops are flagged for review". That is the
difference between a tool you can trust with an album and a tool you have to
audit by hand.

The model is asked to judge, never to change anything. It returns a small object
and nothing it says alters a single pixel.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from revelai.vlm.client import VLMClient, VLMUnavailable, get_vlm_client

__all__ = ["Verdict", "VERIFY_SCHEMA", "verify_crop", "build_verifier"]

#: The question, as a schema. Constraining the shape of the answer is what makes
#: this usable over a large batch.
VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "complete": {
            "type": "boolean",
            "description": "True if this is exactly one complete photograph.",
        },
        "cut_off_edges": {
            "type": "array",
            "items": {"type": "string", "enum": ["top", "bottom", "left", "right"]},
            "description": "Edges where the photograph appears cut off.",
        },
        "contains_multiple": {
            "type": "boolean",
            "description": "True if more than one photograph is visible.",
        },
        "contains_page_background": {
            "type": "boolean",
            "description": "True if album paper or part of another photo is visible.",
        },
        "orientation": {
            "type": "integer",
            "enum": [0, 90, 180, 270],
            "description": "Clockwise rotation needed to make the photograph upright.",
        },
        "confidence": {
            "type": "number",
            "description": "How sure you are, from 0 to 1.",
        },
        "note": {
            "type": "string",
            "description": "One short sentence, only if something is wrong.",
        },
    },
    "required": [
        "complete",
        "cut_off_edges",
        "contains_multiple",
        "contains_page_background",
        "orientation",
        "confidence",
    ],
    "additionalProperties": False,
}

_SYSTEM = (
    "You inspect crops taken from photographs of family album pages. Each image "
    "should be exactly one complete printed photograph with no album paper "
    "around it and no part of a neighbouring photograph. Judge only what you "
    "can see. Do not describe the people."
)

_PROMPT = """\
Look at this crop from an album page and answer these questions.

1. Is it exactly one complete photograph?
2. Is the photograph cut off at any edge? A photograph that runs off the edge of
   the crop is cut off; one that simply reaches the edge with its border intact
   is not.
3. Is more than one photograph visible, even partly?
4. Is any album paper, page background or sliver of a neighbouring photograph
   visible around the picture?
5. Which way up is it? Give the clockwise rotation in degrees that would make
   the photograph upright: 0, 90, 180 or 270.

A date stamp printed in the border, like "MAR 82", is part of the photograph.
It is not page background and it must not count against completeness.
"""


@dataclass
class Verdict:
    """What the model said about one crop."""

    complete: bool = True
    cut_off_edges: list[str] = field(default_factory=list)
    contains_multiple: bool = False
    contains_page_background: bool = False
    orientation: int = 0
    confidence: float = 0.0
    note: str = ""
    #: Set when the model could not be reached; the crop is not judged.
    unavailable: str = ""

    @property
    def checked(self) -> bool:
        return not self.unavailable

    @property
    def ok(self) -> bool:
        """Whether this crop can be accepted without a human looking at it."""
        if not self.checked:
            return True
        return (
            self.complete
            and not self.cut_off_edges
            and not self.contains_multiple
            and not self.contains_page_background
        )

    def problems(self) -> list[str]:
        """Why the crop was flagged, in words that fit a review list."""
        if not self.checked or self.ok:
            return []
        reasons = []
        if self.contains_multiple:
            reasons.append("more than one photograph in the crop")
        if self.cut_off_edges:
            reasons.append("cut off at the " + ", ".join(self.cut_off_edges))
        if self.contains_page_background:
            reasons.append("album paper or a neighbouring photo is visible")
        if not self.complete and not reasons:
            reasons.append("not a complete photograph")
        if self.note:
            reasons.append(self.note)
        return reasons

    @classmethod
    def from_payload(cls, payload: dict) -> Verdict:
        edges = payload.get("cut_off_edges") or []
        return cls(
            complete=bool(payload.get("complete", True)),
            cut_off_edges=[str(e) for e in edges],
            contains_multiple=bool(payload.get("contains_multiple", False)),
            contains_page_background=bool(payload.get("contains_page_background", False)),
            orientation=int(payload.get("orientation", 0) or 0),
            confidence=float(payload.get("confidence", 0.0) or 0.0),
            note=str(payload.get("note", "") or ""),
        )


def verify_crop(image: np.ndarray, client: VLMClient | None = None) -> Verdict:
    """Ask whether one crop is a single complete photograph.

    An unreachable model is reported, never guessed around: a crop that could
    not be checked is marked as unchecked rather than quietly assumed good.
    """
    client = client or get_vlm_client()
    try:
        payload = client.ask_json(image, _PROMPT, VERIFY_SCHEMA, system=_SYSTEM)
    except VLMUnavailable as exc:
        return Verdict(unavailable=str(exc))
    return Verdict.from_payload(payload)


def build_verifier(client: VLMClient | None = None):
    """A callable for the split pipeline: crop in, verdict out."""
    client = client or get_vlm_client()

    def verify(image: np.ndarray) -> Verdict:
        return verify_crop(image, client)

    return verify
