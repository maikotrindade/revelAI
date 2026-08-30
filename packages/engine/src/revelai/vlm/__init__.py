"""Vision-language models, used to judge and describe. Never to change a pixel.

This package answers questions about photographs: is this crop one complete
photograph, which way up is it, what is in it, roughly when was it taken. The
models that actually restore an image are in ``revelai.enhance.backends`` and
are a different kind of thing entirely.
"""

from __future__ import annotations

from revelai.vlm.client import VLMClient, VLMUnavailable, get_vlm_client
from revelai.vlm.describe import Description, describe_photo
from revelai.vlm.verify import Verdict, verify_crop

__all__ = [
    "VLMClient",
    "VLMUnavailable",
    "get_vlm_client",
    "Verdict",
    "verify_crop",
    "Description",
    "describe_photo",
]
