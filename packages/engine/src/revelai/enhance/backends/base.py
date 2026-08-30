"""The backend interface every restoration provider implements.

A backend is a bag of optional capabilities. Anything it cannot do it says so,
and the pipeline reports that rather than failing: a user who asks for face
restoration on a machine without the model should be told, not given a
traceback, and certainly not given the photograph back unchanged and unlabelled.

Each operation returns the pixels *and* a record of what did the work, because
a restored photograph has to be able to say what was done to it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from revelai import RevelAIError

__all__ = [
    "Operation",
    "BackendUnavailable",
    "EnhancerBackend",
    "OPERATIONS",
    "MODEL_OPERATIONS",
]

#: Everything a backend may be asked to do, in no particular order.
OPERATIONS = ("color", "denoise", "dust", "faces", "upscale", "colorize")

#: The subset that needs a model rather than a deterministic algorithm. These
#: are the ones that invent detail, and the ones a user has to opt into.
MODEL_OPERATIONS = ("faces", "colorize")


class BackendUnavailable(RevelAIError):
    """A backend cannot perform an operation that was asked of it."""


@dataclass(frozen=True)
class Operation:
    """A record of one thing that was done to a photograph."""

    name: str
    #: What performed it: an algorithm name, or a model name.
    engine: str
    version: str = ""
    #: True when a model produced detail that was not in the original.
    generative: bool = False

    def __str__(self) -> str:
        engine = f"{self.engine} {self.version}".strip()
        return f"{self.name}({engine})" if engine else self.name


class EnhancerBackend:
    """Base class. Every operation is unavailable until a subclass provides it.

    Subclasses override :meth:`capabilities` and whichever operations they
    implement. The default implementations raise, so a backend can never
    silently do nothing and report success.
    """

    #: Short name, as used by ``--backend``.
    name = "base"
    #: Whether using this backend sends photographs to somebody else's computer.
    sends_images_away = False
    #: Where the images go, for the confirmation prompt. Empty when local.
    destination = ""

    def capabilities(self) -> set[str]:
        """The operations this backend can actually perform right now."""
        return set()

    def supports(self, operation: str) -> bool:
        return operation in self.capabilities()

    def unavailable_reason(self, operation: str) -> str:
        """Why an operation is not available, in words a user can act on."""
        if operation not in OPERATIONS:
            return f"{operation!r} is not an operation RevelAI knows about"
        return f"the {self.name} backend cannot do {operation} on this machine"

    # -- operations --------------------------------------------------------

    def _refuse(self, operation: str):
        raise BackendUnavailable(self.unavailable_reason(operation))

    def denoise(self, image: np.ndarray) -> tuple[np.ndarray, Operation]:
        """Remove film grain and scanning noise."""
        self._refuse("denoise")

    def dust(self, image: np.ndarray) -> tuple[np.ndarray, Operation]:
        """Remove dust, scratches and creases."""
        self._refuse("dust")

    def faces(self, image: np.ndarray) -> tuple[np.ndarray, Operation]:
        """Reconstruct faces. Generative: see the warning in the README."""
        self._refuse("faces")

    def upscale(self, image: np.ndarray, factor: int) -> tuple[np.ndarray, Operation]:
        """Increase resolution."""
        self._refuse("upscale")

    def colorize(self, image: np.ndarray) -> tuple[np.ndarray, Operation]:
        """Invent colour for a black and white photograph."""
        self._refuse("colorize")
