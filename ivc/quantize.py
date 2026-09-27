"""Colour quantisation - the step that turns a photo into flat vector colours.

Adobe Illustrator's "Colours" slider is a k-means reduction of the image, so that
is exactly what happens here, only in CIE Lab space so that the palette matches
what the eye considers "the same colour" rather than what RGB numbers say.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

Rgb = Tuple[int, int, int]

_MAX_SAMPLE = 120_000
_MAP_CHUNK = 150_000


@dataclass
class QuantizeResult:
    indices: np.ndarray      # int32, HxW, -1 means "transparent / no colour"
    palette: List[Rgb]
    counts: List[int]


def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Convert an (N,3) Lab array back to RGB.

    The reshape is required because OpenCV reads a 2-D array of 3 columns as a
    single-channel image, not as three channels.
    """
    flat = np.ascontiguousarray(lab.reshape(1, -1, 3), dtype=np.uint8)
    return cv2.cvtColor(flat, cv2.COLOR_LAB2RGB).reshape(-1, 3)


def _nearest_labels(samples: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Assign every sample to its nearest centre (chunked to save memory)."""
    center_sq = np.einsum("ij,ij->i", centers, centers).reshape(1, -1)
    out = np.empty(samples.shape[0], dtype=np.int32)
    for start in range(0, samples.shape[0], _MAP_CHUNK):
        chunk = samples[start:start + _MAP_CHUNK]
        dists = center_sq - 2.0 * (chunk @ centers.T)
        out[start:start + chunk.shape[0]] = np.argmin(dists, axis=1)
    return out


def _border_color(rgb: np.ndarray) -> Optional[np.ndarray]:
    """Median colour of a thin frame around the image (background guess)."""
    height, width = rgb.shape[:2]
    if height < 8 or width < 8:
        return None
    strip = max(2, min(height, width) // 100)
    frame = np.concatenate([
        rgb[:strip].reshape(-1, 3),
        rgb[-strip:].reshape(-1, 3),
        rgb[:, :strip].reshape(-1, 3),
        rgb[:, -strip:].reshape(-1, 3),
    ])
    return np.median(frame.astype(np.float32), axis=0)


def _interior_mask(rgb: np.ndarray, colors: int) -> np.ndarray:
    """Pixels that sit in the flat middle of a region, not on a boundary.

    Feeding only these to k-means is what keeps a noisy scan from growing a
    palette of near-duplicate junk colours: JPEG speckle lives on the boundary
    between two real colours, so it never gets a palette entry of its own and
    ends up mapped onto whichever of the two real colours is nearer.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    smooth = cv2.GaussianBlur(gray, (0, 0), 1.0)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    flat = magnitude <= float(np.percentile(magnitude, 62.0))
    # A palette needs room to breathe; if the image is mostly boundary, sample
    # everything rather than starve k-means.
    if int(flat.sum()) < max(4096, colors * 64):
        return np.ones(magnitude.shape, dtype=bool)
    return flat


def _spatially_aware(lab: np.ndarray, shape: Tuple[int, int], keep_mask: np.ndarray,
                     weight: float = 0.75, sigma: float = 2.0) -> np.ndarray:
    """Append each pixel's locally averaged colour to its clustering feature.

    Colour alone cannot tell a dark stroke core from its lighter anti-aliased
    flank - they are the same drawing, but a plain k-means happily spends a dozen
    palette entries on that ramp and shreds one thin line into a dozen slivers.
    The local average is the same for every pixel of that stroke, so adding it
    (at `weight` less than 1, so the true colour still leads) teaches the
    clustering that those pixels belong together.

    Dropped background pixels are excluded from the average; otherwise a shape
    touching the border would be tinted towards the background it sits on.
    """
    if weight <= 0.0:
        return lab
    height, width = shape
    flat_keep = np.asarray(keep_mask, dtype=bool).reshape(-1)
    planes = lab.reshape(height, width, 3)
    if not flat_keep.all():
        planes = planes.copy()
        planes.reshape(-1, 3)[~flat_keep] = np.nan
    filled = np.nan_to_num(planes, nan=0.0)
    weight_map = (~np.isnan(planes[:, :, 0])).astype(np.float32)
    blurred = np.empty_like(filled)
    for channel in range(3):
        plane = cv2.GaussianBlur(filled[:, :, channel], (0, 0), sigma)
        mask_blur = cv2.GaussianBlur(weight_map, (0, 0), sigma)
        blurred[:, :, channel] = np.where(mask_blur > 1e-3,
                                          plane / np.maximum(mask_blur, 1e-3), 0.0)
    stacked = np.concatenate([planes.reshape(-1, 3), blurred.reshape(-1, 3)], axis=1)
    stacked[:, 3:] *= weight
    return np.ascontiguousarray(stacked, dtype=np.float32)


def _two_nearest(samples: np.ndarray, centers: np.ndarray) -> tuple:
    """Distance to the closest and the second closest centre for every sample.

    The gap between the two is the confidence of the assignment: a large gap
    means the pixel sits well inside a region, a small gap means it lies on the
    boundary between two colours and is genuinely undecided.
    """
    center_sq = np.einsum("ij,ij->i", centers, centers).reshape(1, -1)
    best = np.empty(samples.shape[0], dtype=np.float32)
    second = np.empty(samples.shape[0], dtype=np.float32)
    for start in range(0, samples.shape[0], _MAP_CHUNK):
        chunk = samples[start:start + _MAP_CHUNK]
        dists = center_sq - 2.0 * (chunk @ centers.T)
        order = np.argpartition(dists, 1, axis=1)[:, :2]
        picked = np.take_along_axis(dists, order, axis=1)
        picked = np.sqrt(np.maximum(picked, 0.0))
        best[start:start + chunk.shape[0]] = picked[:, 0]
        second[start:start + chunk.shape[0]] = picked[:, 1]
    return best, second


def _settle_boundaries(labels: np.ndarray, undecided: np.ndarray,
                       colors: int, votes: int) -> np.ndarray:
    """Let spatial context decide the fate of undecided boundary pixels.

    Pixels whose two best colours are nearly equally good are exactly the
    anti-aliased boundary pixels.  Assigning them by colour distance alone puts
    them on whichever side the noise happened to favour, which is what makes a
    traced edge wobble.  Instead each such pixel is handed to the colour that
    owns most of its neighbourhood, so the boundary follows the region rather
    than the grain.  Pixels that are clearly inside a region are never touched.
    """
    if colors < 2 or votes < 3 or not undecided.any():
        return labels

    kernel = (votes, votes)
    winner = np.zeros(labels.shape, np.int32)
    best_share = np.full(labels.shape, -1.0, np.float32)
    for k in range(colors):
        share = cv2.blur((labels == k).astype(np.float32), kernel)
        better = share > best_share
        winner[better] = k
        best_share = np.where(better, share, best_share)

    out = labels.copy()
    out[undecided] = winner[undecided]
    return out


def quantize(rgb: np.ndarray, colors: int,
             drop_background: bool = False,
             background_tolerance: float = 26.0,
             seed: int = 7) -> QuantizeResult:
    """Reduce `rgb` to at most `colors` flat colours.

    When `drop_background` is set, pixels matching the estimated background are
    marked -1 so the EPS/SVG export keeps them transparent (like "remove white"
    in Illustrator).
    """
    height, width = rgb.shape[:2]
    flat = rgb.reshape(-1, 3)

    # Everything is drawn unless a background is explicitly removed.
    indices = np.zeros(flat.shape[0], dtype=np.int32)
    if drop_background:
        bg = _border_color(rgb)
        if bg is not None:
            limit = float(background_tolerance)
            diff = np.abs(flat.astype(np.float32) - bg.reshape(1, 3)).max(axis=1)
            indices[diff <= limit] = -1

    keep_mask = indices >= 0
    flat_kept = flat[keep_mask]

    if flat_kept.shape[0] == 0:
        return QuantizeResult(indices=indices.reshape(height, width),
                              palette=[(255, 255, 255)], counts=[0])

    colors = max(1, min(int(colors), 256))
    lab_all = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    # Cluster on colour *and* local context (see _spatially_aware); the palette
    # itself is still read off the colour half of each centre.
    feature_all = _spatially_aware(lab_all, (height, width), keep_mask)
    lab_kept = feature_all[keep_mask]

    # Train mostly on flat interiors (see _interior_mask) but reserve a slice of
    # the budget for a uniform sample, so a small but important region still
    # earns its own palette entry.
    interior = _interior_mask(rgb, colors).reshape(-1)[keep_mask]
    pool = np.flatnonzero(interior)
    if pool.shape[0] < max(2048, colors * 64):
        pool = np.arange(flat_kept.shape[0], dtype=np.int64)
    interior_stride = max(1, pool.shape[0] // int(_MAX_SAMPLE * 0.75))
    uniform_stride = max(1, flat_kept.shape[0] // int(_MAX_SAMPLE * 0.25))
    picked = np.unique(np.concatenate([
        pool[::interior_stride],
        np.arange(0, flat_kept.shape[0], uniform_stride, dtype=np.int64),
    ]))
    samples = np.ascontiguousarray(lab_kept[picked], dtype=np.float32)
    if samples.shape[0] < colors:
        samples = np.ascontiguousarray(lab_kept, dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    attempts = 3 if colors <= 32 else 2
    cv2.setRNGSeed(int(seed) % (2 ** 31 - 1))  # reproducible palettes
    feature_width = samples.shape[1]
    _, _, centers = cv2.kmeans(
        samples, colors, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    centers = np.ascontiguousarray(centers.reshape(-1, feature_width), dtype=np.float32)

    kept_labels = _nearest_labels(lab_kept, centers)
    closest, second = _two_nearest(lab_kept, centers)

    # Boundary pixels sit within a hair of the decision.  The gate is quoted in
    # colour-only units, so it is widened when the feature space carries the
    # extra context dimensions.
    gate = 2.5 * math.sqrt(feature_width / 3.0)
    undecided = np.zeros(height * width, dtype=bool)
    undecided[keep_mask] = (second - closest) < gate
    full_labels = np.full(height * width, -1, dtype=np.int32)
    full_labels[keep_mask] = kept_labels
    settled = _settle_boundaries(full_labels.reshape(height, width),
                                 undecided.reshape(height, width),
                                 colors=len(centers), votes=5)
    kept_labels = settled.reshape(-1)[keep_mask]

    palette_rgb = _lab_to_rgb(np.ascontiguousarray(centers[:, :3]))
    palette: List[Rgb] = [(int(r), int(g), int(b)) for r, g, b in palette_rgb]

    indices[keep_mask] = kept_labels

    counts = np.bincount(indices[indices >= 0], minlength=colors)
    # A centre can lose every pixel once the boundaries settle.  Such an entry
    # only adds an empty layer to the output, so fold it away and close the gaps
    # in the numbering.
    used = np.flatnonzero(counts > 0)
    if used.shape[0] and used.shape[0] < len(palette):
        remap = np.full(len(palette), -1, dtype=np.int32)
        remap[used] = np.arange(used.shape[0], dtype=np.int32)
        safe = np.where(indices >= 0, remap[np.maximum(indices, 0)], -1)
        new_counts = np.bincount(safe[safe >= 0], minlength=used.shape[0])
        return QuantizeResult(indices=safe.reshape(height, width),
                              palette=[palette[i] for i in used],
                              counts=[int(c) for c in new_counts])
    return QuantizeResult(indices=indices.reshape(height, width),
                          palette=palette,
                          counts=[int(c) for c in counts])


def order_by_area(result: QuantizeResult) -> List[int]:
    """Colour indices sorted from most to least used (paint order)."""
    return sorted(range(len(result.palette)),
                  key=lambda i: (-result.counts[i], i))


def merge_duplicates(result: QuantizeResult, min_distance: float = 12.0) -> QuantizeResult:
    """Fold near-identical palette entries together to cut the path count."""
    palette = result.palette
    if len(palette) < 2:
        return result
    mapping = list(range(len(palette)))
    for i in range(len(palette)):
        if mapping[i] != i:
            continue
        for j in range(i + 1, len(palette)):
            if mapping[j] != j:
                continue
            a = np.array(palette[i], dtype=np.float32)
            b = np.array(palette[j], dtype=np.float32)
            if float(np.linalg.norm(a - b)) < min_distance:
                mapping[j] = i

    keep = sorted({m for m in mapping})
    remap = np.full(len(mapping), -1, dtype=np.int32)
    for new_index, old_index in enumerate(keep):
        remap[old_index] = new_index

    indices = result.indices
    safe = np.where(indices >= 0, remap[np.maximum(indices, 0)], -1)
    new_palette = [palette[i] for i in keep]
    new_counts = np.bincount(safe[safe >= 0], minlength=len(keep)).tolist()
    return QuantizeResult(indices=safe, palette=new_palette,
                          counts=[int(c) for c in new_counts])
