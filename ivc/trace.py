"""Mask to vector-path tracing.

Each quantised colour becomes a binary mask; OpenCV extracts every closed
outline (outer shapes *and* holes) and `geometry.fit_contour` smooths each one
into cubic Beziers.  All outlines of one colour are emitted as a single path
that is filled with the even-odd rule, which reproduces Illustrator's nesting
behaviour without any depth bookkeeping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

import cv2
import numpy as np

from . import geometry
from .geometry import Path, Point


@dataclass
class TraceSettings:
    tolerance: float = 0.9        # max curve deviation, in working pixels
    corner_threshold: float = 75.0  # degrees; above this a vertex stays a corner
    min_area: float = 6.0         # drop specks smaller than this (px^2)
    simplify: float = 0.0         # Douglas-Peucker epsilon; 0 = derive from tolerance
    smoothing: int = 3            # outline de-jitter passes
    pixel_scale: float = 1.0      # working-resolution upscale factor
    max_paths: int = 20000        # hard safety limit


def _contour_points(contour: np.ndarray) -> List[Point]:
    # OpenCV returns (N,1,2) or (N,2) depending on the build; flatten both.
    flat = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    return [(float(p[0]), float(p[1])) for p in flat]


def _valid(path: Sequence[geometry.Bezier]) -> bool:
    """Reject a fit that produced non-finite numbers (NaN would corrupt output)."""
    if not path:
        return False
    for bez in path:
        for x, y in bez:
            if not (math.isfinite(x) and math.isfinite(y)):
                return False
    return True


def trace_mask(mask: np.ndarray, settings: TraceSettings) -> List[Path]:
    """Trace one binary mask into a list of closed Bezier paths."""
    if mask.size == 0 or not mask.any():
        return []

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    min_area = max(0.0, settings.min_area)
    max_paths = max(1, settings.max_paths)
    scale = max(0.05, settings.pixel_scale)
    epsilon = settings.simplify if settings.simplify > 0.0 else max(0.45, settings.tolerance * 0.8)
    paths: List[Path] = []

    # Bigger shapes first keeps the number of tiny leftovers low when the
    # caller has to stop early, and gives nicer overlap for the preview.
    ordered = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in ordered:
        if len(paths) >= max_paths:
            break
        if abs(cv2.contourArea(contour)) < min_area:
            continue
        points = _contour_points(contour)
        path = geometry.fit_contour(points,
                                    tolerance=settings.tolerance,
                                    corner_threshold=settings.corner_threshold,
                                    simplify_epsilon=epsilon,
                                    smoothing=settings.smoothing,
                                    pixel_scale=scale)
        if path and _valid(path):
            paths.append(path)
    return paths


def masks_from_indices(indices: np.ndarray, palette_size: int) -> List[np.ndarray]:
    """One uint8 {0,1} mask per palette colour (index -1 stays excluded)."""
    masks: List[np.ndarray] = []
    for i in range(palette_size):
        masks.append(np.ascontiguousarray((indices == i).astype(np.uint8)))
    return masks
