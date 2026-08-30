"""RevelAI - split album page scans into individual photographs, then restore them.

The tool has two independent stages. ``split`` takes photographs of album pages
and returns each individual print, cropped and deskewed, with no quality loss
whatsoever. ``enhance`` takes separated photographs and returns restored
versions, never touching the originals. Neither stage depends on the other.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__", "RevelAIError", "PRODUCT_NAME"]

#: The product name as it appears in prose, the README and the CLI banner. The
#: repository is ``revelAI`` in mixed case; that casing stops at the repository
#: name and never reaches a Python identifier or a module path.
PRODUCT_NAME = "RevelAI"


class RevelAIError(Exception):
    """Base class for every error RevelAI raises on purpose.

    The CLI catches this and prints a readable message. Anything that escapes as
    a raw traceback is a bug.
    """
