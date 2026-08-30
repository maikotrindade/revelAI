"""The bridge between the split pipeline and the vision-language model.

``--verify`` asks whether each crop is one complete photograph. ``--auto-orient``
acts on the rotation the same answer already contains, which is why the two
share one call: asking twice would double the cost of a batch for nothing.
"""

from __future__ import annotations

import numpy as np

from revelai.vlm.client import VLMClient, get_vlm_client
from revelai.vlm.verify import verify_crop

__all__ = ["Inspector", "build_inspector", "rotate_upright"]


def rotate_upright(image: np.ndarray, clockwise_degrees: int) -> np.ndarray:
    """Rotate by a multiple of 90 degrees, losslessly.

    A quarter turn only moves pixels around; it does not resample them. That is
    the only reason ``--auto-orient`` is allowed to happen inside ``split`` at
    all, where the rule is one resampling per photograph and no more.
    """
    turns = int(clockwise_degrees or 0) % 360
    if turns == 0:
        return image
    if turns not in (90, 180, 270):
        return image
    # np.rot90 turns anticlockwise, so a clockwise quarter turn is three of them.
    return np.ascontiguousarray(np.rot90(image, k=(4 - turns // 90) % 4))


class Inspector:
    """The callback ``run_split`` uses to look at each finished crop.

    It also keeps count of how many crops were actually checked, which the run
    summary needs in order to be honest. Reporting "11 of 11 passed" when the
    model was never reachable would be worse than not offering the feature.
    """

    def __init__(
        self,
        *,
        verify: bool = False,
        auto_orient: bool = False,
        client: VLMClient | None = None,
        on_unavailable=None,
    ) -> None:
        self.verify = verify
        self.auto_orient = auto_orient
        self.client = client or get_vlm_client()
        self.on_unavailable = on_unavailable
        #: Crops the model actually looked at.
        self.checked = 0
        #: Crops it took issue with.
        self.flagged = 0
        #: Crops it rotated.
        self.reoriented = 0
        #: Why it could not be reached, if it could not be.
        self.unavailable_reason = ""

    @property
    def ran(self) -> bool:
        return self.checked > 0

    def __call__(self, pixels, detection, source, name):
        metadata: dict[str, str] = {}
        problems: list[str] = []
        if pixels is None or not (self.verify or self.auto_orient):
            return pixels, metadata, problems

        verdict = verify_crop(pixels, self.client)

        if not verdict.checked:
            if not self.unavailable_reason and self.on_unavailable:
                self.on_unavailable(verdict.unavailable)
            self.unavailable_reason = verdict.unavailable
            return pixels, metadata, problems

        self.checked += 1

        if self.verify:
            metadata["revelai:verified"] = "yes" if verdict.ok else "flagged"
            metadata["revelai:verify-confidence"] = f"{verdict.confidence:.2f}"
            problems.extend(verdict.problems())
            if problems:
                self.flagged += 1

        if self.auto_orient and verdict.orientation:
            pixels = rotate_upright(pixels, verdict.orientation)
            metadata["revelai:auto-orient"] = f"rotated {verdict.orientation} degrees clockwise"
            self.reoriented += 1

        return pixels, metadata, problems


def build_inspector(
    *,
    verify: bool = False,
    auto_orient: bool = False,
    client: VLMClient | None = None,
    on_unavailable=None,
) -> Inspector:
    """Build the crop inspector for ``--verify`` and ``--auto-orient``."""
    return Inspector(
        verify=verify, auto_orient=auto_orient, client=client, on_unavailable=on_unavailable
    )
