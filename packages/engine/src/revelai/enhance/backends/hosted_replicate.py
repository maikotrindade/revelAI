"""A hosted backend, so that restoration works without a capable machine.

Read the privacy section of the README before using this. These are family
photographs, and a hosted backend uploads them to somebody else's computer.
RevelAI's default is local for that reason, and the CLI asks for confirmation
the first time a hosted backend is used in a run.

Replicate was chosen because it hosts the models this problem actually calls for
under stable, versioned identifiers, and because it needs nothing but an HTTP
client. The model identifiers live in :data:`MODELS` and are meant to be edited;
nothing else in RevelAI knows what they are.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np

from revelai.enhance.backends.base import BackendUnavailable, EnhancerBackend, Operation

__all__ = ["ReplicateBackend", "MODELS", "HostedModel"]

_API = "https://api.replicate.com/v1/predictions"
_ENV_TOKEN = "REPLICATE_API_TOKEN"
_POLL_SECONDS = 2.0
_TIMEOUT_SECONDS = 600.0


@dataclass(frozen=True)
class HostedModel:
    """One hosted model and what it costs, roughly."""

    version: str
    #: Human readable name, recorded in the output metadata.
    name: str
    #: Approximate US cents per photograph, for the README and the CLI.
    cents: float
    #: Extra input fields the model wants.
    extra: dict | None = None
    generative: bool = False


#: The models each operation is sent to. Replace freely; nothing depends on
#: these beyond this file.
MODELS: dict[str, HostedModel] = {
    "denoise": HostedModel(
        version="9283608cc6b7be6b65a8e44983db012355fde4132009bf99d976b2f0896856a3",
        name="real-esrgan (denoise pass)",
        cents=0.4,
        extra={"scale": 1, "face_enhance": False},
    ),
    "upscale": HostedModel(
        version="9283608cc6b7be6b65a8e44983db012355fde4132009bf99d976b2f0896856a3",
        name="real-esrgan",
        cents=0.4,
        extra={"face_enhance": False},
    ),
    "faces": HostedModel(
        version="7de2ea26c616d5bf2245ad0d5e24f0ff9a6204578a5c876db53142edd9d2cd56",
        name="gfpgan",
        cents=0.6,
        generative=True,
    ),
    "colorize": HostedModel(
        version="0da600fab0c45a66211339f1c16b71345d22f26ef5fea3dca1bb90bb5711e950",
        name="deoldify",
        cents=1.2,
        generative=True,
    ),
}


class ReplicateBackend(EnhancerBackend):
    """Restoration through Replicate's hosted models."""

    name = "replicate"
    sends_images_away = True
    destination = "Replicate (replicate.com), in the United States"

    def __init__(self, token: str | None = None, *, timeout: float = _TIMEOUT_SECONDS) -> None:
        self.token = token or os.environ.get(_ENV_TOKEN, "")
        self.timeout = timeout

    def capabilities(self) -> set[str]:
        return set(MODELS) if self.token else set()

    def unavailable_reason(self, operation: str) -> str:
        if not self.token:
            return (
                f"the replicate backend needs an API token: set {_ENV_TOKEN}. "
                f"The default local backend needs no key and sends nothing anywhere."
            )
        if operation not in MODELS:
            return f"the replicate backend has no model registered for {operation}"
        return super().unavailable_reason(operation)

    def estimated_cents(self, operations: list[str]) -> float:
        """Roughly what one photograph will cost, in US cents."""
        return sum(MODELS[op].cents for op in operations if op in MODELS)

    # -- operations --------------------------------------------------------

    def denoise(self, image):
        return self._run("denoise", image)

    def dust(self, image):
        # No hosted model is registered for dust removal: the classical method
        # in the local backend is better at it than a general restoration model,
        # which tends to repaint the whole photograph.
        self._refuse("dust")

    def faces(self, image):
        return self._run("faces", image)

    def colorize(self, image):
        return self._run("colorize", image)

    def upscale(self, image, factor: int):
        return self._run("upscale", image, extra={"scale": int(factor)})

    # -- transport ---------------------------------------------------------

    def _run(self, operation: str, image: np.ndarray, extra: dict | None = None):
        if not self.token:
            raise BackendUnavailable(self.unavailable_reason(operation))
        model = MODELS.get(operation)
        if model is None:
            raise BackendUnavailable(self.unavailable_reason(operation))

        payload = {"image": _to_data_uri(image)}
        payload.update(model.extra or {})
        payload.update(extra or {})

        url = self._create(model.version, payload)
        result = _fetch(url)
        pixels = _decode(result, image.dtype)
        return pixels, Operation(
            name=operation,
            engine=f"replicate/{model.name}",
            version=model.version[:12],
            generative=model.generative,
        )

    def _create(self, version: str, payload: dict) -> str:
        """Start a prediction and wait for its output URL."""
        request = urllib.request.Request(
            _API,
            data=json.dumps({"version": version, "input": payload}).encode(),
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                prediction = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            raise BackendUnavailable(
                f"replicate refused the request ({exc.code}). "
                f"Check {_ENV_TOKEN}, or run with the default local backend."
            ) from exc
        except urllib.error.URLError as exc:
            raise BackendUnavailable(f"could not reach replicate: {exc.reason}") from exc

        deadline = time.monotonic() + self.timeout
        poll_url = prediction.get("urls", {}).get("get")
        while True:
            status = prediction.get("status")
            if status == "succeeded":
                output = prediction.get("output")
                if isinstance(output, list):
                    output = output[0] if output else None
                if not output:
                    raise BackendUnavailable("replicate returned no image")
                return output
            if status in ("failed", "canceled"):
                raise BackendUnavailable(
                    f"replicate prediction {status}: {prediction.get('error') or 'no reason given'}"
                )
            if time.monotonic() > deadline or not poll_url:
                raise BackendUnavailable("replicate did not finish in time")
            time.sleep(_POLL_SECONDS)
            poll_request = urllib.request.Request(
                poll_url, headers={"Authorization": f"Bearer {self.token}"}
            )
            with urllib.request.urlopen(poll_request, timeout=60) as response:
                prediction = json.loads(response.read())


def _to_data_uri(image: np.ndarray) -> str:
    """Encode losslessly. A hosted model should not be fed a JPEG of a scan."""
    array = np.asarray(image)
    ok, buffer = cv2.imencode(".png", array)
    if not ok:  # pragma: no cover - encoder failure
        raise BackendUnavailable("could not encode the photograph for upload")
    return "data:image/png;base64," + base64.b64encode(buffer.tobytes()).decode()


def _fetch(url: str) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            return response.read()
    except (urllib.error.URLError, OSError) as exc:
        raise BackendUnavailable(f"could not download the restored image: {exc}") from exc


def _decode(data: bytes, dtype) -> np.ndarray:
    pixels = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if pixels is None:
        raise BackendUnavailable("the restored image could not be decoded")
    if dtype == np.uint16 and pixels.dtype == np.uint8:
        return (pixels.astype(np.float32) * 257.0).round().clip(0, 65535).astype(np.uint16)
    return pixels
