"""Conversion pipeline: bitmap in, layered vector document out."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from . import quantize as quant_mod
from . import trace as trace_mod
from .enhance import EnhanceSettings, enhance
from .geometry import flatten_path
from .document import VectorDocument, VectorLayer

ProgressFn = Callable[[int, str], None]
CancelFn = Callable[[], bool]


@dataclass
class ConverterSettings:
    """Everything the GUI sliders control."""

    # vector side
    colors: int = 8
    tolerance: float = 0.6          # lower = smoother, more accurate
    corner_threshold: float = 75.0  # degrees
    min_area: float = 6.0           # specks below this (px^2) are dropped
    smoothing: int = 2              # outline de-jitter passes
    drop_background: bool = True
    background_tolerance: float = 28.0
    max_paths: int = 20000

    # image repair side
    denoise: int = 6
    upscale: float = 1.5
    contrast: int = 2
    sharpen: int = 40
    despeckle: int = 1
    max_dimension: int = 2600

    def enhance_settings(self) -> EnhanceSettings:
        return EnhanceSettings(
            denoise=int(self.denoise),
            upscale=float(self.upscale),
            contrast=int(self.contrast),
            sharpen=int(self.sharpen),
            despeckle=int(self.despeckle),
            max_dimension=int(self.max_dimension),
        )

    def trace_settings(self) -> trace_mod.TraceSettings:
        return trace_mod.TraceSettings(
            tolerance=max(0.15, float(self.tolerance)),
            corner_threshold=float(self.corner_threshold),
            min_area=float(self.min_area),
            smoothing=int(self.smoothing),
            max_paths=int(self.max_paths),
        )


class Cancelled(Exception):
    pass


def _noop_progress(percent: int, message: str) -> None:
    pass


def _no_cancel() -> bool:
    return False


def load_image(path: str) -> Image.Image:
    image = Image.open(path)
    if image.mode in ("RGBA", "LA", "PA"):
        background = Image.new("RGB", image.size, (255, 255, 255))
        image = image.convert("RGBA")
        background.paste(image, mask=image.split()[-1])
        image = background
    else:
        image = image.convert("RGB")
    return image


def to_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image, dtype=np.uint8)


def from_array(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.ascontiguousarray(array.astype(np.uint8)), "RGB")


def convert(image: Image.Image, settings: ConverterSettings,
            progress: Optional[ProgressFn] = None,
            cancel: Optional[CancelFn] = None,
            source_name: str = "image") -> VectorDocument:
    """Run the full pipeline and return the layered vector document."""
    report = progress or _noop_progress
    is_cancelled = cancel or _no_cancel
    started = time.perf_counter()

    def checkpoint(percent: int, message: str) -> None:
        if is_cancelled():
            raise Cancelled()
        report(percent, message)

    source_size = image.size
    rgb = to_array(image)

    checkpoint(4, "Cleaning up the image...")
    repaired = enhance(rgb, settings.enhance_settings())
    clean = repaired.image
    height, width = clean.shape[:2]

    checkpoint(28, f"Reducing to {int(settings.colors)} colours...")
    palette_result = quant_mod.quantize(
        clean,
        colors=int(settings.colors),
        drop_background=bool(settings.drop_background),
        background_tolerance=float(settings.background_tolerance))
    palette_result = quant_mod.merge_duplicates(palette_result)
    if len(palette_result.palette) == 0:
        raise ValueError("nothing left to trace - try fewer clean-up steps")

    order = quant_mod.order_by_area(palette_result)
    masks = trace_mod.masks_from_indices(palette_result.indices, len(palette_result.palette))
    trace_cfg = settings.trace_settings()
    # Every pixel measurement below is converted into working-image units, so
    # the sliders keep the same meaning whatever the upscale setting is, and the
    # output size stays stable while the user drags "Upscale".
    scale = max(0.05, repaired.scale)
    trace_cfg.pixel_scale = scale
    trace_cfg.tolerance = max(0.15, float(settings.tolerance)) * scale
    trace_cfg.min_area = max(0.0, float(settings.min_area) * scale * scale)

    layers: List[VectorLayer] = []
    span = max(1, len(order) - 1)
    for position, color_index in enumerate(order):
        if palette_result.counts[color_index] < 2:
            continue
        message = f"Tracing colour {position + 1} of {len(order)}..."
        checkpoint(34 + int(58 * position / span), message)
        paths = trace_mod.trace_mask(masks[color_index], trace_cfg)
        if paths:
            layers.append(VectorLayer(color=palette_result.palette[color_index],
                                      paths=paths))

    background: Optional[Tuple[int, int, int]] = None
    if not settings.drop_background and order:
        background = palette_result.palette[order[0]]

    checkpoint(94, "Building the vector document...")
    result = VectorDocument(
        width=width,
        height=height,
        layers=layers,
        background=background,
        source_name=source_name,
        source_size=source_size,
        trace_scale=repaired.scale,
    )
    checkpoint(100, f"Done in {time.perf_counter() - started:.1f}s")
    return result


# --------------------------------------------------------------------------- #
# preview rendering (rasterises the real vector paths)
# --------------------------------------------------------------------------- #
def render(document: VectorDocument, max_size: Tuple[int, int] = (720, 720),
           background: Optional[Tuple[int, int, int]] = (255, 255, 255),
           size: Optional[Tuple[int, int]] = None) -> Image.Image:
    """Flatten the vector paths back into a bitmap so the user sees the truth.

    `size` renders at exactly that pixel size, which is how the quality benchmark
    puts the traced outline and the source photo on the same pixel grid.
    """
    if size is not None and size[0] > 0 and size[1] > 0:
        out_w, out_h = int(size[0]), int(size[1])
        scale = out_w / float(document.width) if document.width else 1.0
    else:
        scale = 1.0
        if document.width > 0 and document.height > 0:
            scale = min(max_size[0] / document.width, max_size[1] / document.height, 1.0)
            if scale <= 0:
                scale = 1.0
        out_w = max(1, int(round(document.width * scale)))
        out_h = max(1, int(round(document.height * scale)))

    if background is None:
        canvas = np.zeros((out_h, out_w, 4), dtype=np.uint8)
    else:
        canvas = np.zeros((out_h, out_w, 4), dtype=np.uint8)
        canvas[:, :] = (*background, 255)

    doc_background = document.background
    if doc_background is not None:
        canvas[:, :] = (*doc_background, 255)

    for layer in document.layers:
        rings = []
        for path in layer.paths:
            ring = flatten_path(path, target_step=0.6)
            if len(ring) >= 3:
                rings.append(np.round(np.asarray(ring, dtype=np.float32) * scale).astype(np.int32))
        if not rings:
            continue
        colour = (int(layer.color[0]), int(layer.color[1]), int(layer.color[2]), 255)
        cv2.fillPoly(canvas, rings, color=colour, lineType=cv2.LINE_8)

    return Image.fromarray(canvas, "RGBA")
