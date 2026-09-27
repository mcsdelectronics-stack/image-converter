"""Quality benchmark for the tracer.

Numbers, not vibes.  Every image is converted with the shipped defaults and
scored on three independent axes:

  edge   - mean distance from the traced outline to the edge the *original*
           image really has.  This is the honest measure of trace accuracy: it
           does not care about colours, only about where the curves land.
  soft   - mean absolute difference from the original after a small blur, which
           is what the eye judges ("does it look like the source").
  tidy   - anchors per 1000 px of outline; lower is a cleaner, lighter file.

`edge` and `soft` are lower-is-better, `tidy` is lower-is-better as well.
"""
from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ivc.document import to_eps
from ivc.engine import ConverterSettings, convert
from ivc.geometry import flatten_path

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "testdata")


# --------------------------------------------------------------------------- #
# test images
# --------------------------------------------------------------------------- #
def _jitter(image: Image.Image, size: int, quality: int, noise: float, seed: int):
    rng = np.random.default_rng(seed)
    small = image.resize((size, int(size * image.size[1] / image.size[0])), Image.LANCZOS)
    array = np.asarray(small).astype(np.float32)
    if noise:
        array += rng.normal(0, noise, array.shape)
    path = os.path.join(DATA, f"bench_{size}_{seed}.jpg")
    Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).save(path, quality=quality)
    return Image.open(path).convert("RGB")


def image_logo() -> Image.Image:
    img = Image.new("RGB", (900, 600), "white")
    d = ImageDraw.Draw(img)
    d.ellipse([120, 60, 780, 540], fill=(214, 40, 40))
    d.ellipse([200, 100, 700, 500], fill=(255, 255, 255))
    d.ellipse([300, 150, 600, 450], fill=(30, 90, 200))
    d.ellipse([360, 190, 540, 410], fill=(255, 255, 255))
    d.rectangle([150, 300, 300, 360], fill=(20, 160, 90))
    d.rectangle([600, 300, 750, 360], fill=(20, 160, 90))
    d.polygon([(340, 120), (560, 120), (450, 250)], fill=(240, 180, 20))
    return _jitter(img, 190, 25, 8.0, 1)


def image_text() -> Image.Image:
    """Sharp typography: the hardest case for corner accuracy."""
    img = Image.new("RGB", (760, 260), "white")
    d = ImageDraw.Draw(img)
    try:
        from PIL import ImageFont
        font = ImageFont.truetype("arialbd.ttf", 150)
    except OSError:
        font = ImageFont.load_default()
    d.text((20, 40), "Aa 24", font=font, fill=(25, 28, 35))
    d.rectangle([60, 200, 300, 235], fill=(200, 40, 60))
    d.polygon([(420, 235), (520, 150), (620, 235)], fill=(30, 90, 200))
    d.ellipse([650, 140, 740, 230], fill=(20, 160, 90))
    return _jitter(img, 200, 30, 6.0, 2)


def image_photo() -> Image.Image:
    rng = np.random.default_rng(5)
    img = Image.new("RGB", (760, 560))
    d = ImageDraw.Draw(img)
    for i in range(0, 760, 40):
        d.line([(i, 0), (i - 200, 560)], fill=(40 + i % 180, 90, 190 - i % 140), width=6)
    for _ in range(10):
        x, y = int(rng.integers(0, 640)), int(rng.integers(0, 460))
        d.ellipse([x, y, x + int(rng.integers(60, 150)), y + int(rng.integers(60, 150))],
                  fill=tuple(int(v) for v in rng.integers(0, 255, 3)))
    return _jitter(img, 260, 35, 5.0, 3)


def image_lines() -> Image.Image:
    """Thin strokes: tests whether hairlines survive tracing."""
    img = Image.new("RGB", (600, 400), (250, 250, 248))
    d = ImageDraw.Draw(img)
    for i in range(6):
        d.arc([40 + i * 18, 60, 520 - i * 18, 340], 200, 340,
              fill=(40, 40, 60), width=4)
    d.line([(60, 380), (540, 380)], fill=(200, 60, 60), width=3)
    for i in range(12):
        d.line([(80 + i * 40, 30), (80 + i * 40, 120)], fill=(60, 90, 160), width=2)
    return _jitter(img, 150, 20, 7.0, 4)


IMAGES = {
    "logo": image_logo,
    "text": image_text,
    "photo": image_photo,
    "lines": image_lines,
}


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _edges(gray: np.ndarray) -> np.ndarray:
    """Strong edges only.

    A light blur first: the metric is about *where* an edge is, not about
    pixel noise, and single-pixel JPEG speckle must not be mistaken for an
    edge.  Otsu on the gradient magnitude keeps the threshold meaningful for
    both clean logos and noisy scans.
    """
    smooth = cv2.GaussianBlur(gray, (0, 0), 1.0)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    peak = float(magnitude.max())
    if peak <= 0:
        return np.zeros(gray.shape, bool)
    scaled = np.clip(magnitude * (255.0 / peak), 0, 255).astype(np.uint8)
    level, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return magnitude > max(0.094 * peak, 0.45 * float(level))


def gray_size(gray: np.ndarray) -> float:
    return float(gray.shape[0] * gray.shape[1])


def edge_error(source: Image.Image, doc) -> float:
    """Mean distance between the traced outlines and the source's real edges.

    Measured at the source resolution: the document is rendered at the source
    size, so both edge sets live on the same pixel grid.
    """
    from ivc.engine import render

    preview = render(doc, background=doc.background, size=source.size)
    on_white = Image.new("RGB", preview.size, (255, 255, 255))
    if preview.mode == "RGBA":
        on_white.paste(preview, (0, 0), preview)
    else:
        on_white.paste(preview)

    src_gray = np.asarray(source.convert("L"), np.float32)
    vec_gray = np.asarray(on_white.convert("L"), np.float32)

    src_edges = _edges(src_gray)
    vec_edges = _edges(vec_gray)
    # If either edge map is nearly empty the measurement is meaningless: a thin
    # stroke can render too faint for the gradient test to find, and the average
    # would then be dominated by whatever few pixels survived.
    floor = 0.002 * gray_size(src_gray)
    if src_edges.sum() < floor or vec_edges.sum() < floor:
        return float("nan")

    dist_to_vec = cv2.distanceTransform((~vec_edges).astype(np.uint8), cv2.DIST_L2, 3)
    dist_to_src = cv2.distanceTransform((~src_edges).astype(np.uint8), cv2.DIST_L2, 3)
    forward = float(dist_to_vec[src_edges].mean())   # source edge -> traced edge
    backward = float(dist_to_src[vec_edges].mean())  # traced edge -> source edge
    return 0.5 * (forward + backward)


def soft_error(source: Image.Image, doc) -> float:
    from ivc.engine import render
    preview = render(doc, background=doc.background, size=source.size)
    on_white = Image.new("RGB", preview.size, (255, 255, 255))
    if preview.mode == "RGBA":
        on_white.paste(preview, (0, 0), preview)
    else:
        on_white.paste(preview)
    a = np.asarray(source.convert("RGB"), np.float32)
    b = np.asarray(on_white.convert("RGB"), np.float32)
    k = (0, 0), 1.2
    a = cv2.GaussianBlur(a, *k)
    b = cv2.GaussianBlur(b, *k)
    return float(np.abs(a - b).mean())


def outline_length(doc) -> float:
    total = 0.0
    for layer in doc.layers:
        for path in layer.paths:
            ring = flatten_path(path, target_step=0.5)
            for i in range(1, len(ring)):
                total += math.dist(ring[i - 1], ring[i])
            total += math.dist(ring[-1], ring[0]) if len(ring) > 2 else 0.0
    return total / max(doc.trace_scale, 1e-6)  # in source pixels


def tidiness(doc) -> float:
    length = outline_length(doc)
    if length <= 0:
        return float("inf")
    return doc.anchor_count / (length / 1000.0)


# --------------------------------------------------------------------------- #
@dataclass
class Row:
    name: str
    edge: float
    soft: float
    tidy: float
    anchors: int
    paths: int
    kb: float
    seconds: float


def run(settings: ConverterSettings = None, verbose: bool = True) -> List[Row]:
    settings = settings or ConverterSettings()
    rows: List[Row] = []
    for colors in (6, 16):
        tuned = ConverterSettings(**{**vars(settings), "colors": colors})
        print(f"-- colors = {colors} " + "-" * 52)
        print(f"{'image':7s} {'edge px':>8s} {'soft':>7s} {'tidy':>7s} "
              f"{'anchors':>8s} {'paths':>6s} {'eps KB':>8s} {'sec':>5s}")
        for name, factory in IMAGES.items():
            source = factory()
            started = time.perf_counter()
            doc = convert(source, tuned)
            seconds = time.perf_counter() - started
            row = Row(name=name,
                      edge=edge_error(source, doc),
                      soft=soft_error(source, doc),
                      tidy=tidiness(doc),
                      anchors=doc.anchor_count,
                      paths=doc.path_count,
                      kb=len(to_eps(doc).encode()) / 1024,
                      seconds=seconds)
            rows.append(row)
            print(f"{row.name:7s} {row.edge:8.3f} {row.soft:7.2f} {row.tidy:7.1f} "
                  f"{row.anchors:8d} {row.paths:6d} {row.kb:8.1f} {row.seconds:5.1f}")
    flat = [r for r in rows if r.name in ("logo", "text", "lines")]
    print(f"{'MEAN*':7s} {float(np.mean([r.edge for r in flat])):8.3f} "
          f"{float(np.mean([r.soft for r in flat])):7.2f} "
          f"{float(np.mean([r.tidy for r in flat])):7.1f}   "
          f"(* logo/text/lines only; photo is a stress case)")
    return rows


def main() -> None:
    os.makedirs(DATA, exist_ok=True)
    print("== shipped defaults ==")
    run()


if __name__ == "__main__":
    main()

