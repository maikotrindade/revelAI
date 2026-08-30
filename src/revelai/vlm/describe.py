"""Captions, tags, dating and orientation (specification 10.6).

The model describes what is in the photograph: how many people are present, what
the scene is, what decade the clothing and the paper suggest, and any date stamp
or handwritten caption it can read. It counts people; it does not identify them.
Face recognition is explicitly out of scope for this project.

Everything here is a guess made from the picture, and it is recorded as a guess:
an estimated decade goes into the metadata as an estimate, not as a date.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from revelai.vlm.client import VLMClient, VLMUnavailable, get_vlm_client

__all__ = ["Description", "DESCRIBE_SCHEMA", "describe_photo", "build_describer"]

DESCRIBE_SCHEMA = {
    "type": "object",
    "properties": {
        "caption": {
            "type": "string",
            "description": "One sentence describing the scene.",
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "A few short keywords: setting, occasion, objects.",
        },
        "people_count": {
            "type": "integer",
            "description": "How many people are visible. Not who they are.",
        },
        "estimated_decade": {
            "type": "string",
            "description": "Best guess from clothing, print style and paper, e.g. '1970s'.",
        },
        "decade_confidence": {"type": "number"},
        "date_stamp": {
            "type": "string",
            "description": (
                "Any date printed in the border, verbatim, e.g. 'MAR 82'. Empty if none."
            ),
        },
        "handwriting": {
            "type": "string",
            "description": "Any handwritten caption visible, transcribed. Empty if none.",
        },
        "is_black_and_white": {"type": "boolean"},
    },
    "required": [
        "caption",
        "tags",
        "people_count",
        "estimated_decade",
        "decade_confidence",
        "date_stamp",
        "handwriting",
        "is_black_and_white",
    ],
    "additionalProperties": False,
}

_SYSTEM = (
    "You describe scanned family photographs so they can be found again later. "
    "Count the people present; never guess who they are, and never name anyone. "
    "Describe only what is visible."
)

_PROMPT = """\
Describe this photograph for a family archive.

Give a one sentence caption, a few keyword tags, and how many people are
visible. Estimate the decade it was taken from the clothing, the print style and
the state of the paper, and say how confident you are.

Read out any date stamp printed in the border, exactly as it appears - prints
from the 1970s and 1980s often carry one, and it is the best dating evidence in
the picture. Transcribe any handwritten caption you can see.
"""


@dataclass
class Description:
    caption: str = ""
    tags: list[str] = field(default_factory=list)
    people_count: int = 0
    estimated_decade: str = ""
    decade_confidence: float = 0.0
    date_stamp: str = ""
    handwriting: str = ""
    is_black_and_white: bool = False
    unavailable: str = ""

    @property
    def checked(self) -> bool:
        return not self.unavailable

    @classmethod
    def from_payload(cls, payload: dict) -> Description:
        return cls(
            caption=str(payload.get("caption", "") or ""),
            tags=[str(t) for t in (payload.get("tags") or [])],
            people_count=int(payload.get("people_count", 0) or 0),
            estimated_decade=str(payload.get("estimated_decade", "") or ""),
            decade_confidence=float(payload.get("decade_confidence", 0.0) or 0.0),
            date_stamp=str(payload.get("date_stamp", "") or ""),
            handwriting=str(payload.get("handwriting", "") or ""),
            is_black_and_white=bool(payload.get("is_black_and_white", False)),
        )

    def as_metadata(self) -> dict[str, str]:
        """PNG text chunks. Estimates are labelled as estimates."""
        if not self.checked:
            return {}
        out = {
            "revelai:caption": self.caption,
            "revelai:tags": ", ".join(self.tags),
            "revelai:people-count": str(self.people_count),
        }
        if self.estimated_decade:
            out["revelai:estimated-decade"] = (
                f"{self.estimated_decade} (estimated from the image, "
                f"confidence {self.decade_confidence:.2f})"
            )
        if self.date_stamp:
            out["revelai:date-stamp"] = self.date_stamp
        if self.handwriting:
            out["revelai:handwriting"] = self.handwriting
        return {key: value for key, value in out.items() if value}


def describe_photo(image: np.ndarray, client: VLMClient | None = None) -> Description:
    """Caption, tag and date one photograph."""
    client = client or get_vlm_client()
    try:
        payload = client.ask_json(image, _PROMPT, DESCRIBE_SCHEMA, system=_SYSTEM)
    except VLMUnavailable as exc:
        return Description(unavailable=str(exc))
    return Description.from_payload(payload)


def build_describer(client: VLMClient | None = None):
    """A callable for the enhance pipeline: photograph in, metadata out."""
    client = client or get_vlm_client()

    def describe(image: np.ndarray, source=None) -> dict[str, str]:
        return describe_photo(image, client).as_metadata()

    return describe
