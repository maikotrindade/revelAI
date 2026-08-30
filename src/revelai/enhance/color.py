"""Classical colour correction (specification 10.2). No AI anywhere in here.

Most of what is wrong with a scanned family photograph is that its dye layers
have faded at different rates. The cyan layer usually goes first, which is why
old prints turn yellow-red. That is a per-channel gain and offset problem, and a
per-channel gain and offset fixes it.

Percentile clipping is what makes it safe. Taking the true minimum and maximum
of a channel would let one dust speck or one blown highlight define the whole
mapping; ignoring half a per cent at each end reads the black and white points
the photograph actually has.

This runs offline, is deterministic, invents nothing, and is instantaneous. It
is tried before any model is called, and on a plain yellowed print it is
usually the only thing needed.
"""

from __future__ import annotations

import numpy as np

__all__ = ["DEFAULT_CLIP", "correct_colour_cast", "channel_levels", "grey_deviation"]

#: Percentage ignored at each end of each channel when reading its levels.
DEFAULT_CLIP = 0.5


def _max_value(dtype: np.dtype) -> float:
    return 65535.0 if dtype == np.uint16 else 255.0


def channel_levels(image: np.ndarray, clip: float = DEFAULT_CLIP) -> tuple[np.ndarray, np.ndarray]:
    """The black and white point of each channel, ignoring ``clip`` per cent.

    Returns ``(low, high)``, one value per channel, in the image's own units.
    """
    array = np.asarray(image)
    if array.ndim == 2:
        array = array[..., None]
    flat = array.reshape(-1, array.shape[2]).astype(np.float64)
    low = np.percentile(flat, clip, axis=0)
    high = np.percentile(flat, 100.0 - clip, axis=0)
    return low, high


def correct_colour_cast(
    image: np.ndarray, strength: float = 1.0, clip: float = DEFAULT_CLIP
) -> np.ndarray:
    """Balance the channels of a faded print, per channel, by their own levels.

    Each channel is mapped from the range it actually occupies onto the full
    range. A channel that has faded to a narrow band is stretched back out; one
    that has not is left more or less alone. The result is that the three
    channels end up describing the same range, which is what removing a cast
    means.

    ``strength`` blends between the original (0) and the fully corrected
    image (1), because on a print that is only slightly warm the full
    correction can be more than the owner wants.
    """
    array = np.asarray(image)
    if array.size == 0:
        return array.copy()

    strength = float(np.clip(strength, 0.0, 1.0))
    if strength == 0.0:
        return array.copy()

    original_shape = array.shape
    working = array[..., None] if array.ndim == 2 else array
    ceiling = _max_value(array.dtype)

    low, high = channel_levels(working, clip)
    span = high - low
    # A flat channel has nothing to stretch; leave it exactly as it is rather
    # than dividing by nearly zero and inventing contrast that is not there.
    usable = span > 1e-6

    values = working.astype(np.float32)
    corrected = values.copy()
    scale = np.zeros_like(span)
    scale[usable] = ceiling / span[usable]
    corrected[..., usable] = (values[..., usable] - low[usable]) * scale[usable]
    corrected = np.clip(corrected, 0.0, ceiling)

    if strength < 1.0:
        corrected = values * (1.0 - strength) + corrected * strength

    result = np.clip(corrected, 0.0, ceiling).astype(array.dtype)
    return result.reshape(original_shape)


def grey_deviation(image: np.ndarray) -> float:
    """How far a photograph is from neutral, in 0..255 units.

    The mean absolute spread between the channels of a pixel. A neutral
    photograph scores near zero; a yellowed one scores high. Used to judge
    whether the correction did what it claims.
    """
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] < 3:
        return 0.0
    values = array[..., :3].astype(np.float64)
    if array.dtype == np.uint16:
        values = values / 257.0
    return float(np.abs(values.max(axis=2) - values.min(axis=2)).mean())
