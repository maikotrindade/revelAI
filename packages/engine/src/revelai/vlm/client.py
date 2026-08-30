"""Provider abstraction for the vision-language model.

A vision-language model is used here to *judge and describe*, never to change a
pixel. It answers questions like "is this one complete photograph?" and "which
way up is it?" - questions that geometry cannot settle and that a language model
is genuinely good at. The restoration models live somewhere else entirely, in
``enhance/backends``; the two are not the same kind of thing and are not
interchangeable.

Nothing here is required to use RevelAI. With no API key, ``--verify`` reports
that it is unavailable and the run continues without it.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from revelai import RevelAIError

__all__ = [
    "VLMUnavailable",
    "VLMClient",
    "AnthropicVLM",
    "NullVLM",
    "get_vlm_client",
    "encode_for_vlm",
    "DEFAULT_MODEL",
    "MAX_EDGE",
]

#: Sent to the model unless REVELAI_VLM_MODEL says otherwise.
DEFAULT_MODEL = "claude-opus-5"

#: Longest edge of the copy sent for analysis. Beyond about this size a
#: vision model gains nothing and the request just costs more.
MAX_EDGE = 1400

_ENV_KEY = "ANTHROPIC_API_KEY"
_ENV_MODEL = "REVELAI_VLM_MODEL"


class VLMUnavailable(RevelAIError):
    """No vision-language model is configured, or it could not be reached."""


def encode_for_vlm(image: np.ndarray, max_edge: int = MAX_EDGE) -> tuple[str, str]:
    """Encode a photograph for analysis. Returns ``(media_type, base64)``.

    The copy that is sent is downscaled and JPEG encoded. That is a copy made in
    memory for the question being asked; the photograph on disk is never
    touched, and nothing this module returns is written back into an image.
    """
    array = np.asarray(image)
    if array.dtype == np.uint16:
        array = (array / 257.0).round().astype(np.uint8)
    if array.ndim == 2:
        array = cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    elif array.shape[2] == 4:
        array = array[..., :3]

    longest = max(array.shape[:2])
    if longest > max_edge:
        factor = max_edge / float(longest)
        array = cv2.resize(
            array,
            (max(1, round(array.shape[1] * factor)), max(1, round(array.shape[0] * factor))),
            interpolation=cv2.INTER_AREA,
        )

    ok, buffer = cv2.imencode(".jpg", array, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:  # pragma: no cover - encoder failure
        raise VLMUnavailable("could not encode the photograph for analysis")
    return "image/jpeg", base64.b64encode(buffer.tobytes()).decode()


@dataclass
class VLMClient:
    """Base class. Subclasses answer a question about an image with JSON."""

    name: str = "none"
    model: str = ""

    def available(self) -> bool:
        return False

    def unavailable_reason(self) -> str:
        return (
            f"no vision-language model is configured: set {_ENV_KEY} and "
            f"install the extra with pip install 'revelai[vlm]'"
        )

    def ask_json(self, image: np.ndarray, prompt: str, schema: dict, *, system: str = "") -> dict:
        raise VLMUnavailable(self.unavailable_reason())


@dataclass
class NullVLM(VLMClient):
    """Stands in when nothing is configured, so callers need no special case."""

    name: str = "none"


@dataclass
class AnthropicVLM(VLMClient):
    """Claude, through the official Anthropic SDK."""

    name: str = "anthropic"
    model: str = field(default_factory=lambda: os.environ.get(_ENV_MODEL, DEFAULT_MODEL))
    api_key: str = field(default_factory=lambda: os.environ.get(_ENV_KEY, ""))
    max_tokens: int = 1024
    _client: object | None = field(default=None, repr=False)

    def available(self) -> bool:
        if not self.api_key:
            return False
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        return True

    def unavailable_reason(self) -> str:
        if not self.api_key:
            return f"{_ENV_KEY} is not set, so crop verification cannot run"
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return "the Anthropic SDK is not installed: pip install 'revelai[vlm]'"
        return "the vision-language model is unavailable"

    def client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise VLMUnavailable(self.unavailable_reason()) from exc
            self._client = anthropic.Anthropic(api_key=self.api_key or None)
        return self._client

    def ask_json(self, image: np.ndarray, prompt: str, schema: dict, *, system: str = "") -> dict:
        """Ask one question about one image and get a validated object back.

        The answer is constrained by a JSON schema rather than parsed out of
        prose. Over a fifty page batch, a single unparseable reply is a crop
        silently unchecked, which defeats the point of checking at all.
        """
        if not self.available():
            raise VLMUnavailable(self.unavailable_reason())

        import anthropic

        media_type, data = encode_for_vlm(image)
        request = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": data,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "output_config": {"format": {"type": "json_schema", "schema": schema}},
        }
        if system:
            request["system"] = system

        try:
            response = self.client().messages.create(**request)
        except anthropic.APIStatusError as exc:
            raise VLMUnavailable(f"the vision-language model refused the request: {exc}") from exc
        except anthropic.APIConnectionError as exc:
            raise VLMUnavailable(f"could not reach the vision-language model: {exc}") from exc

        text = next((block.text for block in response.content if block.type == "text"), "")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise VLMUnavailable(
                f"the vision-language model did not return usable JSON: {text[:200]}"
            ) from exc


def get_vlm_client(name: str | None = None) -> VLMClient:
    """Build a client. Falls back to one that politely refuses."""
    key = (name or "anthropic").strip().lower()
    if key in ("none", "null", "off"):
        return NullVLM()
    if key == "anthropic":
        client = AnthropicVLM()
        return client if client.available() else client
    raise RevelAIError(f"unknown vision-language provider {name!r}")
