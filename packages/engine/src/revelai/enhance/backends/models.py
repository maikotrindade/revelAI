"""On-demand model download, caching and inference for the local backend.

No model is hardwired into the pipeline. Each one is a :class:`ModelSpec` in the
registry below, paired with a runner that knows how to feed it. Swapping a model
for a better one is an edit to the registry, not to the restoration code, which
is the point: the models worth using change faster than this tool will.

Models are fetched only when the user asks, verified against a checksum, and
cached. Nothing is downloaded during a normal run, and nothing is downloaded
during tests: an operation with no cached model reports itself unavailable.

Inference goes through ONNX Runtime, which is an optional extra. It is the
mature way to run a restoration model on a user's own machine without dragging
in a deep learning framework, and it keeps the base install small enough that
the classical path stays instant.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from revelai import RevelAIError
from revelai.enhance.backends.base import Operation

__all__ = ["ModelSpec", "ModelCache", "LoadedModel", "REGISTRY", "cache_directory"]

_ENV_CACHE = "REVELAI_MODEL_CACHE"
_CHUNK = 1 << 16


class ModelError(RevelAIError):
    """A model could not be fetched, verified or run."""


def cache_directory() -> Path:
    """Where downloaded models live.

    Overridable with ``REVELAI_MODEL_CACHE`` so that a user can keep several
    gigabytes of weights somewhere other than their home directory, and so that
    tests can point it at an empty folder.
    """
    override = os.environ.get(_ENV_CACHE)
    if override:
        return Path(override)
    base = os.environ.get("XDG_CACHE_HOME")
    if base:
        return Path(base) / "revelai" / "models"
    return Path.home() / ".cache" / "revelai" / "models"


@dataclass(frozen=True)
class ModelSpec:
    """One model: where it comes from, and how to feed it."""

    #: Operation this model performs.
    operation: str
    #: Short identifier, also the cached filename stem.
    name: str
    url: str
    #: Human readable version, recorded in the output metadata.
    version: str
    #: Expected SHA-256 of the download. Empty means "do not verify", which is
    #: only acceptable for a model the user pointed at themselves.
    sha256: str = ""
    #: Side of the square tile the model expects, or 0 when it takes any size.
    tile: int = 0
    #: How much larger the output is than the input.
    scale: int = 1
    #: Which channels the model consumes: "y" for luminance, "bgr" for colour.
    channels: str = "y"
    #: True when the model produces detail that was not in the original.
    generative: bool = False
    notes: str = ""

    @property
    def filename(self) -> str:
        return f"{self.name}.onnx"


#: The models RevelAI knows how to fetch.
#:
#: These are starting points, not commitments. Anything with a compatible
#: runner can be dropped in here, and a user can point at their own file with
#: REVELAI_MODEL_<OPERATION>.
REGISTRY: dict[str, list[ModelSpec]] = {
    "upscale": [
        ModelSpec(
            operation="upscale",
            name="subpixel-cnn-x3",
            url=(
                "https://github.com/onnx/models/raw/main/validated/vision/"
                "super_resolution/sub_pixel_cnn_2016/model/super-resolution-10.onnx"
            ),
            version="onnx-zoo-10",
            tile=224,
            scale=3,
            channels="y",
            notes="Small and quick. Sharpens luminance only; colour is interpolated.",
        ),
    ],
    "faces": [],
    "colorize": [],
}


def _user_supplied(operation: str) -> Path | None:
    """A model file the user pointed at directly, if any."""
    value = os.environ.get(f"REVELAI_MODEL_{operation.upper()}")
    if not value:
        return None
    path = Path(value)
    return path if path.is_file() else None


@dataclass
class LoadedModel:
    """A model that is on disk and ready to run."""

    spec: ModelSpec
    path: Path
    _session: object | None = field(default=None, repr=False)

    def session(self):
        if self._session is None:
            try:
                import onnxruntime as ort
            except ImportError as exc:  # pragma: no cover - depends on the install
                raise ModelError(
                    "running a local model needs ONNX Runtime: pip install 'revelai[local-models]'"
                ) from exc
            options = ort.SessionOptions()
            options.log_severity_level = 3
            self._session = ort.InferenceSession(
                str(self.path), options, providers=["CPUExecutionProvider"]
            )
        return self._session

    def operation(self, name: str, generative: bool | None = None) -> Operation:
        return Operation(
            name=name,
            engine=self.spec.name,
            version=self.spec.version,
            generative=self.spec.generative if generative is None else generative,
        )

    # -- inference ---------------------------------------------------------

    def apply(self, image: np.ndarray, **kwargs) -> np.ndarray:
        if self.spec.channels == "y":
            return self._apply_luminance(image)
        return self._apply_colour(image)

    def _run(self, batch: np.ndarray) -> np.ndarray:
        session = self.session()
        name = session.get_inputs()[0].name
        return session.run(None, {name: batch.astype(np.float32)})[0]

    def _apply_luminance(self, image: np.ndarray) -> np.ndarray:
        """Run the model on luminance and carry the colour up by interpolation.

        Chrominance carries very little of the detail the eye uses, so this is
        the standard arrangement for super-resolution and it keeps the model
        small. The colour is resized with Lanczos, which is the best thing
        available that does not invent anything.
        """
        array = np.asarray(image)
        eight = (array / 257.0).round().astype(np.uint8) if array.dtype == np.uint16 else array
        if eight.ndim == 2:
            eight = cv2.cvtColor(eight, cv2.COLOR_GRAY2BGR)

        ycrcb = cv2.cvtColor(eight, cv2.COLOR_BGR2YCrCb)
        luma = ycrcb[..., 0].astype(np.float32) / 255.0
        upscaled = self._tile(luma)

        height, width = eight.shape[:2]
        target = (width * self.spec.scale, height * self.spec.scale)
        chroma = cv2.resize(ycrcb[..., 1:], target, interpolation=cv2.INTER_LANCZOS4)
        merged = np.dstack([np.clip(upscaled * 255.0, 0, 255).astype(np.uint8), chroma])
        result = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
        if array.dtype == np.uint16:
            return (result.astype(np.float32) * 257.0).round().clip(0, 65535).astype(np.uint16)
        return result

    def _apply_colour(self, image: np.ndarray) -> np.ndarray:
        array = np.asarray(image)
        eight = (array / 257.0).round().astype(np.uint8) if array.dtype == np.uint16 else array
        blob = eight[..., ::-1].astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        out = self._run(blob)[0].transpose(1, 2, 0)
        result = np.clip(out * 255.0, 0, 255).astype(np.uint8)[..., ::-1]
        if array.dtype == np.uint16:
            return (result.astype(np.float32) * 257.0).round().clip(0, 65535).astype(np.uint16)
        return result

    def _tile(self, plane: np.ndarray) -> np.ndarray:
        """Run a fixed-tile model over an image of any size.

        Models exported at one input size are the norm rather than the
        exception. Tiles overlap and are blended with a raised-cosine weight,
        because a hard join between tiles shows up as a grid of seams across
        the sky of every photograph that gets upscaled.
        """
        size = self.spec.tile
        scale = self.spec.scale
        if size <= 0:
            return self._run(plane[None, None])[0, 0]

        height, width = plane.shape
        overlap = max(8, size // 8)
        step = size - overlap

        # Pad so that the tiles cover the image and the model always gets a
        # full tile, even for an image smaller than one tile.
        padded_h = max(size, -(-max(height - size, 0) // step) * step + size)
        padded_w = max(size, -(-max(width - size, 0) // step) * step + size)
        padded = cv2.copyMakeBorder(
            plane, 0, padded_h - height, 0, padded_w - width, cv2.BORDER_REFLECT_101
        )

        window = np.hanning(size * scale + 2)[1:-1].astype(np.float32)
        weights = np.outer(window, window)

        out_h, out_w = padded_h * scale, padded_w * scale
        accumulator = np.zeros((out_h, out_w), np.float32)
        divisor = np.zeros((out_h, out_w), np.float32)

        for top in range(0, padded_h - size + 1, step):
            for left in range(0, padded_w - size + 1, step):
                tile = padded[top : top + size, left : left + size]
                result = self._run(tile[None, None])[0, 0]
                oy, ox = top * scale, left * scale
                accumulator[oy : oy + size * scale, ox : ox + size * scale] += result * weights
                divisor[oy : oy + size * scale, ox : ox + size * scale] += weights

        divisor[divisor < 1e-6] = 1.0
        return (accumulator / divisor)[: height * scale, : width * scale]


class ModelCache:
    """Finds, fetches and verifies the models the local backend can use."""

    def __init__(self, directory: Path | None = None) -> None:
        self.directory = Path(directory) if directory is not None else cache_directory()

    def specs(self, operation: str) -> list[ModelSpec]:
        return REGISTRY.get(operation, [])

    def cached_path(self, spec: ModelSpec) -> Path:
        return self.directory / spec.filename

    def available_operations(self) -> set[str]:
        """Operations that have a usable model on this machine right now."""
        found = set()
        for operation in REGISTRY:
            if _user_supplied(operation) is not None:
                found.add(operation)
                continue
            if any(self.cached_path(spec).is_file() for spec in self.specs(operation)):
                found.add(operation)
        return found

    def load(self, operation: str, **kwargs) -> LoadedModel | None:
        """The model for an operation, or None if there is not one here.

        Returning None rather than raising is deliberate: the backend turns it
        into a message that names the operation and says how to get the model.
        """
        override = _user_supplied(operation)
        if override is not None:
            specs = self.specs(operation)
            spec = specs[0] if specs else ModelSpec(operation, override.stem, "", "user-supplied")
            return LoadedModel(spec=spec, path=override)

        for spec in self.specs(operation):
            path = self.cached_path(spec)
            if path.is_file():
                return LoadedModel(spec=spec, path=path)
        return None

    def fetch(self, operation: str, *, progress=None) -> Path:
        """Download and verify the model for an operation."""
        specs = self.specs(operation)
        if not specs:
            raise ModelError(
                f"RevelAI does not have a registered local model for {operation!r}. "
                f"Point at your own with REVELAI_MODEL_{operation.upper()}, "
                f"or use --backend with a hosted provider."
            )
        spec = specs[0]
        destination = self.cached_path(spec)
        if destination.is_file():
            return destination
        return self.download(spec, progress=progress)

    def download(self, spec: ModelSpec, *, progress=None) -> Path:
        """Fetch one model into the cache, verifying its checksum."""
        import urllib.error
        import urllib.request

        self.directory.mkdir(parents=True, exist_ok=True)
        destination = self.cached_path(spec)
        partial = destination.with_suffix(destination.suffix + ".partial")

        digest = hashlib.sha256()
        try:
            with urllib.request.urlopen(spec.url) as response, partial.open("wb") as handle:
                total = int(response.headers.get("Content-Length") or 0)
                seen = 0
                while True:
                    chunk = response.read(_CHUNK)
                    if not chunk:
                        break
                    handle.write(chunk)
                    digest.update(chunk)
                    seen += len(chunk)
                    if progress:
                        progress(seen, total)
        except (urllib.error.URLError, OSError) as exc:
            partial.unlink(missing_ok=True)
            raise ModelError(f"could not download {spec.name} from {spec.url}: {exc}") from exc

        if spec.sha256 and digest.hexdigest() != spec.sha256:
            partial.unlink(missing_ok=True)
            raise ModelError(
                f"{spec.name} downloaded but its checksum does not match; "
                f"expected {spec.sha256}, got {digest.hexdigest()}"
            )
        partial.replace(destination)
        return destination
