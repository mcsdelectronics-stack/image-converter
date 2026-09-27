"""Pre-trace image repair.

A low quality source (JPEG mush, a phone photo of a print, a 48 px favicon) has to
be cleaned up *before* tracing, otherwise the vectoriser faithfully reproduces
every compression artefact.  The steps here all exist to remove artefacts while
protecting the real edges:

* denoise   - kills JPEG blocking and sensor noise without blurring edges
* upscale   - supersamples the image, so the traced curves have sub-pixel detail
* contrast  - local contrast (CLAHE) so faint shapes survive quantisation
* sharpen   - unsharp mask to re-crisp the edges after resampling
* despeckle - morphological open/close to delete isolated specks and pinholes
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class EnhanceSettings:
    denoise: int = 6          # 0..30  (0 = off)
    upscale: float = 1.5      # 1, 1.5, 2, 3, 4
    contrast: int = 0         # 0..10  (0 = off)
    sharpen: int = 40         # 0..200 (0 = off)
    despeckle: int = 0        # 0..10  (0 = off)
    max_dimension: int = 2600  # working-resolution cap, protects memory/time
    min_dimension: int = 220   # below this the trace is too coarse to be good


@dataclass
class EnhanceResult:
    image: np.ndarray          # uint8 RGB
    scale: float               # upscale factor actually applied


def _as_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    if image.shape[2] == 3:
        return image.copy()
    raise ValueError(f"unsupported channel count: {image.shape[2]}")


def _denoise(rgb: np.ndarray, strength: int) -> np.ndarray:
    if strength <= 0:
        return rgb
    h = float(strength)
    return cv2.fastNlMeansDenoisingColored(rgb, None, h, h, 7, 21)


def _clamp_scale(rgb: np.ndarray, wanted: float, max_dimension: int,
                 min_dimension: int = 0, max_floor: float = 4.0) -> float:
    """Decide the working-resolution factor.

    Two competing limits.  `max_dimension` is a memory/time cap.  `min_dimension`
    is a quality floor: a 60 px favicon has too few pixels to describe its own
    curves, so it gets scaled up past the user's setting - measured on the
    benchmark, small images trace markedly better at 3-4x while a 300 px image
    traces best at 1.5x, because extra upscaling there only magnifies JPEG
    artefacts.  The user's slider is therefore a *minimum*, not a fixed value.

    The floor is itself capped: there is nothing to gain from blowing a 24 px
    icon up ninefold, and the memory cost is not free.
    """
    longest = max(rgb.shape[0], rgb.shape[1])
    if longest <= 0:
        return 1.0
    target = float(wanted)
    if min_dimension > 0:
        target = max(target, min(max_floor, min_dimension / float(longest)))
    if target <= 1.001:
        return 1.0
    return max(1.0, min(target, max_dimension / float(longest)))


def _upscale(rgb: np.ndarray, scale: float) -> Tuple[np.ndarray, float]:
    if scale <= 1.001:
        return rgb, 1.0
    height, width = rgb.shape[:2]
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    out = cv2.resize(rgb, new_size, interpolation=cv2.INTER_LANCZOS4)
    # Report the factor that was really achieved, not the one that was asked
    # for: rounding means the two differ slightly, and every pixel measurement
    # downstream is scaled by this number.
    actual = 0.5 * (new_size[0] / float(width) + new_size[1] / float(height))
    return out, actual


def _local_contrast(rgb: np.ndarray, amount: int) -> np.ndarray:
    if amount <= 0:
        return rgb
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.0 + amount * 0.6, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    return cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2RGB)


def _unsharp(rgb: np.ndarray, amount: int) -> np.ndarray:
    if amount <= 0:
        return rgb
    blurred = cv2.GaussianBlur(rgb, (0, 0), sigmaX=1.1, sigmaY=1.1)
    return cv2.addWeighted(rgb, 1.0 + amount / 100.0, blurred, -(amount / 100.0), 0)


def _despeckle(rgb: np.ndarray, amount: int) -> np.ndarray:
    """Delete isolated specks and fill pinholes using morphology."""
    if amount <= 0:
        return rgb
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    out = rgb
    rounds = 1 if amount <= 4 else 2
    for _ in range(rounds):
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel, iterations=1)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=1)
    return out


def enhance(image: np.ndarray, settings: EnhanceSettings) -> EnhanceResult:
    """Run the full repair chain and report the scale that was applied."""
    rgb = _as_rgb(image)

    rgb = _denoise(rgb, settings.denoise)

    scale = _clamp_scale(rgb, settings.upscale, settings.max_dimension,
                         settings.min_dimension)
    rgb, scale = _upscale(rgb, scale)

    rgb = _local_contrast(rgb, settings.contrast)
    rgb = _unsharp(rgb, settings.sharpen)
    rgb = _despeckle(rgb, settings.despeckle)

    return EnhanceResult(image=np.ascontiguousarray(rgb), scale=scale)
