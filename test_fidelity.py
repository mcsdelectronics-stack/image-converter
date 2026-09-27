"""Fidelity check: does the vector output reproduce the quantised bitmap?

The vector result is rendered and compared, pixel by pixel, with the flat colour
image the tracer was asked to reproduce.  Comparing raw pixels would be unfair -
a curve fitted to a 1 px tolerance is allowed to sit half a pixel off the
staircase - so mismatches are only counted when they are further than 1.5 px away
from a colour boundary.  Those would be real structural errors: wrong nesting,
missing shapes, holes filled in, paint order problems.
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ivc.enhance import enhance as run_enhance
from ivc.quantize import merge_duplicates, quantize
from ivc.engine import ConverterSettings, convert, render

STRUCTURAL_LIMIT_PCT = 0.5


def colorize(indices: np.ndarray, palette, background=None) -> np.ndarray:
    out = np.zeros(indices.shape + (3,), dtype=np.uint8)
    if background is not None:
        out[:, :] = background
    for i, color in enumerate(palette):
        out[indices == i] = color
    return out


def boundary_distance(indices: np.ndarray) -> np.ndarray:
    edges = np.zeros(indices.shape, dtype=bool)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        edges |= (np.roll(indices, dy, 0) != indices) | (np.roll(indices, dx, 1) != indices)
    return cv2.distanceTransform((~edges).astype(np.uint8), cv2.DIST_L2, 3)


def check(source: Image.Image, settings: ConverterSettings, label: str) -> float:
    doc = convert(source, settings)

    rgb = np.asarray(source, dtype=np.uint8)
    repaired = run_enhance(rgb, settings.enhance_settings()).image
    q = quantize(repaired, colors=int(settings.colors),
                 drop_background=settings.drop_background,
                 background_tolerance=settings.background_tolerance)
    q = merge_duplicates(q)

    target = colorize(q.indices, q.palette, background=doc.background)
    # Render at exactly the quantized size: both sides of the comparison have to
    # sit on the same pixel grid or the diff means nothing.
    rendered = np.asarray(render(doc, background=doc.background,
                                 size=(target.shape[1], target.shape[0])
                                 ).convert("RGB"), dtype=np.uint8)
    assert rendered.shape == target.shape

    diff = np.abs(rendered.astype(np.int16) - target.astype(np.int16)).max(axis=2)
    bad = diff > 12
    distance = boundary_distance(q.indices)
    # a curve fitted to `tolerance` is allowed to sit that far from the staircase,
    # and that distance grows with the working upscale factor
    margin = 1.0 + settings.tolerance * doc.trace_scale
    structural = bad & (distance > margin)
    pct = float(structural.mean() * 100.0)

    for layer in doc.layers:
        for path in layer.paths:
            for bez in path:
                for x, y in bez:
                    assert np.isfinite(x) and np.isfinite(y), "NaN in a path"
                    assert -1 <= x <= doc.width + 1, f"x outside canvas: {x}"
                    assert -1 <= y <= doc.height + 1, f"y outside canvas: {y}"

    print(f"{label:22s} {doc.width:4d}x{doc.height:<4d} colours {len(doc.layers):2d} "
          f"paths {doc.path_count:5d} anchors {doc.anchor_count:6d} | "
          f"edge-diff {bad.mean() * 100:5.2f}%  structural {pct:5.3f}% "
          f"(margin {margin:.1f}px)")
    assert pct < STRUCTURAL_LIMIT_PCT, f"structural mismatch too high: {pct:.3f}%"
    return pct


def main() -> None:
    from test_pipeline import ensure_source
    source = Image.open(ensure_source()).convert("RGB")
    base = ConverterSettings(colors=6, tolerance=0.5, min_area=6.0, smoothing=2,
                             drop_background=True, denoise=8, upscale=1.0,
                             contrast=2, sharpen=50, despeckle=1)
    check(source, base, "jpeg logo (1x)")

    two_x = ConverterSettings(**{**base.__dict__, "upscale": 2.0, "tolerance": 0.4,
                                 "colors": 8, "min_area": 8.0})
    check(source, two_x, "jpeg logo (2x, 8c)")

    keep_bg = ConverterSettings(**{**base.__dict__, "drop_background": False})
    check(source, keep_bg, "keep background")

    # a flat synthetic image: exact shapes, no noise - the geometry must be tight
    flat = Image.new("RGB", (400, 300), "white")
    from PIL import ImageDraw
    d = ImageDraw.Draw(flat)
    d.ellipse([60, 40, 340, 260], fill=(200, 30, 30))
    d.ellipse([110, 75, 290, 225], fill="white")
    d.rectangle([180, 120, 230, 180], fill=(20, 60, 200))
    synthetic = ConverterSettings(**{**base.__dict__, "colors": 4, "tolerance": 0.35,
                                     "smoothing": 2, "despeckle": 0, "upscale": 2.0})
    check(flat, synthetic, "synthetic rings")

    print("OK - vector output matches the bitmap structurally")


if __name__ == "__main__":
    main()
