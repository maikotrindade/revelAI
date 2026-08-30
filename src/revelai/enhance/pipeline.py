"""The enhance stage: restore photographs that have already been separated.

Operation order is fixed (specification 10.1) and the order is the substance:

    geometry (done in split)
      -> classical colour correction
      -> denoise
      -> dust and scratch removal
      -> colourisation, if asked for
      -> face restoration, if asked for
      -> super-resolution, last

Super-resolution goes last because it is a magnifier. Run it first and every
defect still in the photograph gets magnified along with the picture, and the
cleaning that follows then has four times as many pixels of amplified grain to
work through. Colourisation goes before face restoration so that faces are
reconstructed in the image that will actually be kept.

Classical before model, always. A per-channel white balance fixes most yellowed
prints outright, costs nothing, and invents nothing; a model is called only for
what is left.

This stage never writes into its input folder. That is checked before any work
starts, and there is a test that the input tree is byte-for-byte identical
afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from revelai import RevelAIError
from revelai.enhance.backends.base import BackendUnavailable, EnhancerBackend, Operation
from revelai.enhance.color import correct_colour_cast

__all__ = [
    "EnhanceOptions",
    "EnhanceResult",
    "PhotoReport",
    "EnhanceReport",
    "enhance_image",
    "run_enhance",
    "write_comparison",
]


@dataclass
class EnhanceOptions:
    """What to do to each photograph."""

    #: Classical colour cast and fading correction. On by default, as specified.
    color: bool = True
    color_strength: float = 1.0
    denoise: bool = False
    dust: bool = False
    #: 0 means no super-resolution; otherwise 2 or 4.
    upscale: int = 0
    #: Off by default. Reconstructs faces; see the warning in the README.
    faces: bool = False
    #: Off by default. The colour is invented, not recovered.
    colorize: bool = False

    def requested(self) -> list[str]:
        """The operations asked for, in the order they must be applied."""
        order = []
        if self.color:
            order.append("color")
        if self.denoise:
            order.append("denoise")
        if self.dust:
            order.append("dust")
        if self.colorize:
            order.append("colorize")
        if self.faces:
            order.append("faces")
        if self.upscale:
            order.append("upscale")
        return order


@dataclass
class EnhanceResult:
    """One restored photograph, and the record of what was done to it."""

    pixels: np.ndarray
    operations: list[Operation] = field(default_factory=list)
    #: Operations that were asked for but could not be performed, with reasons.
    skipped: list[tuple[str, str]] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.operations)

    def summary(self) -> str:
        return ", ".join(str(op) for op in self.operations) or "nothing"


def enhance_image(
    image: np.ndarray, options: EnhanceOptions, backend: EnhancerBackend
) -> EnhanceResult:
    """Apply the requested operations to one photograph, in the fixed order."""
    pixels = np.asarray(image)
    applied: list[Operation] = []
    skipped: list[tuple[str, str]] = []

    for name in options.requested():
        if name == "color":
            # Classical, offline, deterministic. Never a model.
            pixels = correct_colour_cast(pixels, strength=options.color_strength)
            applied.append(Operation("color", "percentile-white-balance", "classical"))
            continue

        if not backend.supports(name):
            skipped.append((name, backend.unavailable_reason(name)))
            continue

        try:
            if name == "upscale":
                pixels, operation = backend.upscale(pixels, options.upscale)
            else:
                pixels, operation = getattr(backend, name)(pixels)
        except BackendUnavailable as exc:
            skipped.append((name, str(exc)))
            continue
        applied.append(operation)

    return EnhanceResult(pixels=pixels, operations=applied, skipped=skipped)


# --------------------------------------------------------------------------
# Comparisons
# --------------------------------------------------------------------------


def write_comparison(path: Path, before: np.ndarray, after: np.ndarray, label: str) -> Path:
    """A before and after pair, side by side, captioned with what was done.

    This is how the owner decides whether the restoration helped or hurt, which
    is not a question the tool can answer for them.
    """
    import cv2

    from revelai.io import write_png

    def prepare(array: np.ndarray) -> np.ndarray:
        out = np.asarray(array)
        if out.dtype == np.uint16:
            out = (out / 257.0).round().astype(np.uint8)
        if out.ndim == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        return out

    left, right = prepare(before), prepare(after)
    height = max(left.shape[0], right.shape[0])

    def fit(array: np.ndarray) -> np.ndarray:
        if array.shape[0] == height:
            return array
        width = max(1, round(array.shape[1] * height / array.shape[0]))
        return cv2.resize(array, (width, height), interpolation=cv2.INTER_AREA)

    left, right = fit(left), fit(right)
    gap = 12
    banner = 34
    canvas = np.full((height + banner, left.shape[1] + gap + right.shape[1], 3), 24, np.uint8)
    canvas[banner : banner + height, : left.shape[1]] = left
    canvas[banner : banner + height, left.shape[1] + gap :] = right
    cv2.putText(
        canvas,
        f"before  |  after: {label}",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    return write_png(path, canvas, overwrite=True)


# --------------------------------------------------------------------------
# Running the stage over a folder
# --------------------------------------------------------------------------


@dataclass
class PhotoReport:
    source: Path
    written: str | None = None
    operations: list[Operation] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class EnhanceReport:
    photos: list[PhotoReport] = field(default_factory=list)
    output: Path | None = None
    dry_run: bool = False
    backend: str = ""

    @property
    def restored(self) -> int:
        return sum(1 for p in self.photos if p.ok and p.written)

    @property
    def failed(self) -> list[PhotoReport]:
        return [p for p in self.photos if not p.ok]

    @property
    def skipped_operations(self) -> dict[str, str]:
        """Operations that could not be performed anywhere in the run."""
        out: dict[str, str] = {}
        for photo in self.photos:
            for name, reason in photo.skipped:
                out.setdefault(name, reason)
        return out


def run_enhance(
    inputs: list[Path],
    output: Path,
    options: EnhanceOptions | None = None,
    backend: EnhancerBackend | None = None,
    *,
    dry_run: bool = False,
    compare_dir: Path | None = None,
    write_metadata: bool = True,
    jobs: int = 1,
    describe=None,
    on_photo=None,
) -> EnhanceReport:
    """Restore every photograph in ``inputs``, writing to ``output``.

    Filenames are preserved: a restored ``photo_00000042.png`` is still
    ``photo_00000042.png``, in a different folder, so the original and the
    restored version always correspond.
    """
    from concurrent.futures import ThreadPoolExecutor

    from revelai import __version__
    from revelai.enhance.backends.local import LocalBackend
    from revelai.io import (
        ImageReadError,
        OutputCollisionError,
        read_image,
        read_png_text,
        write_png,
    )

    options = options or EnhanceOptions()
    backend = backend or LocalBackend()
    output = Path(output)

    _refuse_to_write_into_the_input(inputs, output)
    if compare_dir is not None:
        compare_dir = Path(compare_dir)
        if _same_folder(compare_dir, output):
            raise OutputCollisionError(
                "--compare-dir must not point at the output folder; "
                "the output folder may contain nothing but photo_XXXXXXXX.png files"
            )

    report = EnhanceReport(output=output, dry_run=dry_run, backend=backend.name)

    def process(source: Path) -> PhotoReport:
        photo = PhotoReport(source=source)
        try:
            loaded = read_image(source)
        except ImageReadError as exc:
            photo.error = str(exc)
            return photo

        try:
            result = enhance_image(loaded.pixels, options, backend)
        except RevelAIError as exc:
            photo.error = str(exc)
            return photo
        except Exception as exc:  # noqa: BLE001 - one bad photo must not stop the run
            photo.error = f"could not be restored: {exc}"
            return photo

        photo.operations = result.operations
        photo.skipped = result.skipped
        photo.written = source.name

        if not dry_run:
            text = None
            if write_metadata:
                # Carry the split provenance forward, then add ours: a restored
                # photograph has to be able to say everything done to it.
                text = (
                    {
                        key: value
                        for key, value in read_png_text(source).items()
                        if key.startswith("revelai:") and key != "revelai:stage"
                    }
                    if source.suffix.lower() == ".png"
                    else {}
                )
                text.update(
                    {
                        "revelai:version": __version__,
                        "revelai:stage": "enhance",
                        "revelai:backend": backend.name,
                        "revelai:operations": ", ".join(str(op) for op in result.operations)
                        or "none",
                    }
                )
                generative = [op.name for op in result.operations if op.generative]
                if generative:
                    text["revelai:generative"] = (
                        ", ".join(generative)
                        + " - these invented detail that was not in the original"
                    )
                if describe is not None:
                    text.update(describe(result.pixels, source) or {})
            write_png(
                output / source.name,
                result.pixels,
                icc_profile=loaded.icc_profile,
                text=text,
                overwrite=True,
            )
            if compare_dir is not None:
                write_comparison(
                    compare_dir / f"{source.stem}_compare.png",
                    loaded.pixels,
                    result.pixels,
                    result.summary(),
                )
        return photo

    if jobs > 1:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            processed = list(pool.map(process, inputs))
    else:
        processed = [process(source) for source in inputs]

    for photo in processed:
        report.photos.append(photo)
        if on_photo:
            on_photo(photo)
    return report


def _same_folder(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:  # pragma: no cover - unresolvable paths
        return Path(a).absolute() == Path(b).absolute()


def _refuse_to_write_into_the_input(inputs: list[Path], output: Path) -> None:
    """enhance never touches its originals. Checked before any work is done."""
    from revelai.io import OutputCollisionError

    folders = {Path(source).parent for source in inputs}
    for folder in folders:
        if _same_folder(folder, output):
            raise OutputCollisionError(
                f"--output points at the input folder ({output}). "
                f"enhance never writes into its input: the originals are the "
                f"only copy of the photograph that has not been altered. "
                f"Choose a different output folder."
            )
