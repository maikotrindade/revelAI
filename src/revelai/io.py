"""Reading images, writing lossless PNGs, ICC profiles and provenance metadata.

Everything RevelAI writes is a PNG with no quantisation and no bit depth
reduction. Pixels are handled in OpenCV's native BGR order throughout, because
that is what the detection and warping code speaks; the conversion to and from
disk happens here and nowhere else.

PNG metadata is written by assembling the chunks directly rather than going
through a library, for three reasons: it keeps 8-bit and 16-bit output on one
code path, it lets a restored photograph carry the exact list of operations
applied to it, and it makes the byte layout of our output predictable.
"""

from __future__ import annotations

import struct
import zlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from revelai import RevelAIError
from revelai.naming import natural_key, sort_pages

__all__ = [
    "SUPPORTED_SUFFIXES",
    "ImageReadError",
    "OutputCollisionError",
    "LoadedImage",
    "read_image",
    "write_png",
    "read_png_text",
    "read_png_icc",
    "iter_input_images",
]

#: What a phone or a scanner is likely to hand us.
SUPPORTED_SUFFIXES: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp", ".jpe", ".jp2"}
)

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class ImageReadError(RevelAIError):
    """An input file could not be read as an image."""


class OutputCollisionError(RevelAIError):
    """Writing would destroy a file that already exists."""


@dataclass
class LoadedImage:
    """Pixels plus the colour information we are obliged to carry forward."""

    pixels: np.ndarray
    icc_profile: bytes | None
    path: Path

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])


# --------------------------------------------------------------------------
# PNG chunk plumbing
# --------------------------------------------------------------------------


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _iter_chunks(data: bytes) -> Iterable[tuple[bytes, bytes, int, int]]:
    """Yield ``(kind, payload, start, end)`` for every chunk in a PNG stream."""
    if not data.startswith(_PNG_SIGNATURE):
        raise ImageReadError("not a PNG stream")
    offset = len(_PNG_SIGNATURE)
    total = len(data)
    while offset + 8 <= total:
        (length,) = struct.unpack(">I", data[offset : offset + 4])
        kind = data[offset + 4 : offset + 8]
        start = offset + 8
        end = start + length
        if end + 4 > total:
            raise ImageReadError("truncated PNG chunk")
        yield kind, data[start:end], offset, end + 4
        offset = end + 4
        if kind == b"IEND":
            break


def _encode_text_chunk(keyword: str, value: str) -> bytes:
    """A ``tEXt`` chunk when latin-1 suffices, otherwise ``iTXt`` in UTF-8.

    The specification asks for tEXt. A VLM caption is not necessarily latin-1,
    and silently mangling an accented caption would be worse than using the
    chunk type PNG defines for exactly this case.
    """
    key_bytes = keyword.encode("latin-1")
    if not 1 <= len(key_bytes) <= 79:
        raise ValueError(f"PNG keyword must be 1..79 bytes, got {len(key_bytes)}: {keyword!r}")
    try:
        return _chunk(b"tEXt", key_bytes + b"\x00" + value.encode("latin-1"))
    except UnicodeEncodeError:
        # keyword \0 compression_flag compression_method language \0 translated \0 text
        return _chunk(
            b"iTXt",
            key_bytes + b"\x00" + b"\x00\x00" + b"\x00" + b"\x00" + value.encode("utf-8"),
        )


def _encode_icc_chunk(profile: bytes, name: str = "ICC profile") -> bytes:
    return _chunk(b"iCCP", name.encode("latin-1") + b"\x00\x00" + zlib.compress(profile, 9))


def _decode_text_chunks(data: bytes) -> dict[str, str]:
    out: dict[str, str] = {}
    for kind, payload, _, _ in _iter_chunks(data):
        if kind == b"tEXt":
            key, _, value = payload.partition(b"\x00")
            out[key.decode("latin-1")] = value.decode("latin-1")
        elif kind == b"zTXt":
            key, _, rest = payload.partition(b"\x00")
            if rest[:1] == b"\x00":
                out[key.decode("latin-1")] = zlib.decompress(rest[1:]).decode("latin-1")
        elif kind == b"iTXt":
            key, _, rest = payload.partition(b"\x00")
            if len(rest) < 2:
                continue
            compressed, method = rest[0], rest[1]
            _language, _, rest = rest[2:].partition(b"\x00")
            _translated, _, text = rest.partition(b"\x00")
            if compressed:
                if method != 0:
                    continue
                text = zlib.decompress(text)
            out[key.decode("latin-1")] = text.decode("utf-8")
        elif kind == b"IDAT":
            break
    return out


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------

#: EXIF orientation tag value -> the transform that puts the pixels upright.
_EXIF_ORIENTATION = {
    2: lambda a: a[:, ::-1],
    3: lambda a: a[::-1, ::-1],
    4: lambda a: a[::-1],
    5: lambda a: a.swapaxes(0, 1),
    6: lambda a: a.swapaxes(0, 1)[:, ::-1],
    7: lambda a: a.swapaxes(0, 1)[::-1, ::-1],
    8: lambda a: a.swapaxes(0, 1)[::-1],
}


def _sidecar_metadata(path: Path) -> tuple[bytes | None, int]:
    """ICC profile and EXIF orientation, best effort and never fatal."""
    try:
        with Image.open(path) as img:
            icc = img.info.get("icc_profile")
            try:
                orientation = int(img.getexif().get(274, 1) or 1)
            except Exception:
                orientation = 1
            return (bytes(icc) if icc else None), orientation
    except (UnidentifiedImageError, OSError, ValueError):
        return None, 1


def read_image(path: Path | str) -> LoadedImage:
    """Read an image as BGR pixels, honouring EXIF orientation.

    Pixels come through OpenCV with ``IMREAD_UNCHANGED`` so that 16-bit files
    stay 16-bit. Colour management and orientation come from Pillow, which
    reads the metadata OpenCV discards.
    """
    path = Path(path)
    if not path.exists():
        raise ImageReadError(f"{path} does not exist")
    if not path.is_file():
        raise ImageReadError(f"{path} is not a file")

    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ImageReadError(f"{path} could not be read: {exc}") from exc
    if not data:
        raise ImageReadError(f"{path} is empty")

    pixels = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if pixels is None:
        raise ImageReadError(
            f"{path} could not be decoded as an image (unsupported format, or the file is corrupt)"
        )
    if pixels.ndim == 3 and pixels.shape[2] not in (3, 4):
        raise ImageReadError(f"{path} has {pixels.shape[2]} channels, which is not supported")

    icc, orientation = _sidecar_metadata(path)
    transform = _EXIF_ORIENTATION.get(orientation)
    if transform is not None:
        pixels = np.ascontiguousarray(transform(pixels))

    return LoadedImage(pixels=pixels, icc_profile=icc, path=path)


def read_png_text(path: Path | str) -> dict[str, str]:
    """Every text chunk in a PNG, as a plain mapping."""
    return _decode_text_chunks(Path(path).read_bytes())


def read_png_icc(path: Path | str) -> bytes | None:
    """The embedded ICC profile of a PNG, or None."""
    for kind, payload, _, _ in _iter_chunks(Path(path).read_bytes()):
        if kind == b"iCCP":
            _name, _, rest = payload.partition(b"\x00")
            if rest[:1] != b"\x00":
                return None
            return zlib.decompress(rest[1:])
        if kind == b"IDAT":
            break
    return None


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def write_png(
    path: Path | str,
    image: np.ndarray,
    *,
    icc_profile: bytes | None = None,
    text: Mapping[str, str] | None = None,
    overwrite: bool = False,
    compression: int = 6,
) -> Path:
    """Write ``image`` as a lossless PNG, with optional ICC and text chunks.

    ``image`` is BGR (or BGRA, or single channel), ``uint8`` or ``uint16``. The
    bit depth is preserved: a 16-bit array becomes a 16-bit PNG.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise OutputCollisionError(
            f"{path} already exists and would be overwritten. "
            f"Choose another output folder, or remove the existing file."
        )

    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise ValueError(f"expected a 2D or 3D array, got shape {array.shape}")
    if array.ndim == 3 and array.shape[2] not in (1, 3, 4):
        raise ValueError(f"expected 1, 3 or 4 channels, got {array.shape[2]}")
    if array.dtype not in (np.uint8, np.uint16):
        raise ValueError(
            f"expected uint8 or uint16 pixels, got {array.dtype}; "
            f"converting here would risk a silent loss of precision"
        )

    ok, buffer = cv2.imencode(".png", array, [int(cv2.IMWRITE_PNG_COMPRESSION), int(compression)])
    if not ok:
        raise RevelAIError(f"failed to encode {path} as PNG")
    data = buffer.tobytes()

    extra = b""
    if icc_profile:
        extra += _encode_icc_chunk(icc_profile)
    for key, value in (text or {}).items():
        extra += _encode_text_chunk(str(key), str(value))

    if extra:
        # Ancillary chunks go straight after IHDR, which is always first.
        header = next(_iter_chunks(data))
        insert_at = header[3]
        data = data[:insert_at] + extra + data[insert_at:]

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    try:
        tmp.write_bytes(data)
        tmp.replace(path)
    finally:
        if tmp.exists():  # pragma: no cover - only on a failed write
            tmp.unlink(missing_ok=True)
    return path


# --------------------------------------------------------------------------
# Input discovery
# --------------------------------------------------------------------------


def iter_input_images(root: Path | str, recursive: bool = False) -> list[Path]:
    """Input images under ``root``, in the deterministic page order.

    ``root`` may be a single file. Unsupported files are skipped silently here;
    a file the user named explicitly is always returned so that the caller can
    report a real error rather than an empty run.
    """
    root = Path(root)
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise ImageReadError(f"{root} does not exist")

    entries = root.rglob("*") if recursive else root.iterdir()
    found = [p for p in entries if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES]
    return sort_pages(found) if recursive else sorted(found, key=lambda p: natural_key(p.name))
