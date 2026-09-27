"""Stress test: a large, photo-like input, plus grayscale and alpha inputs."""
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ivc.document import to_eps
from ivc.engine import ConverterSettings, convert, load_image

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "testdata")


def make_photo_like() -> Image.Image:
    rng = np.random.default_rng(3)
    image = Image.new("RGB", (1200, 900))
    draw = ImageDraw.Draw(image)
    for i in range(0, 1200, 60):
        draw.line([(i, 0), (i - 300, 900)], fill=(30 + i % 200, 80, 200 - i % 150), width=3)
    for _ in range(12):
        x, y = int(rng.integers(0, 1100)), int(rng.integers(0, 800))
        box = [x, y, x + int(rng.integers(40, 180)), y + int(rng.integers(40, 180))]
        draw.ellipse(box, fill=tuple(int(v) for v in rng.integers(0, 255, 3)))
    base = np.asarray(image).astype(np.float32)
    base += rng.normal(0, 10, base.shape)
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def main() -> None:
    source = make_photo_like()
    source.save(os.path.join(DATA, "photo_like.png"))

    for colors in (4, 12, 32):
        started = time.perf_counter()
        doc = convert(source, ConverterSettings(colors=colors, upscale=2.0))
        size = len(to_eps(doc).encode()) / 1024 / 1024
        print(f"colors={colors:2d}: {time.perf_counter() - started:5.1f}s "
              f"paths={doc.path_count:6d} anchors={doc.anchor_count:7d} eps={size:5.2f} MB")

    gray_path = os.path.join(DATA, "gray.png")
    Image.fromarray(np.asarray(source).mean(axis=2).astype(np.uint8), "L").save(gray_path)
    started = time.perf_counter()
    doc = convert(load_image(gray_path), ConverterSettings(colors=6))
    print(f"grayscale : {time.perf_counter() - started:5.1f}s paths={doc.path_count} "
          f"colours={len(doc.layers)}")

    alpha_path = os.path.join(DATA, "alpha.png")
    rgba = source.convert("RGBA")
    rgba.putalpha(200)
    rgba.save(alpha_path)
    started = time.perf_counter()
    doc = convert(load_image(alpha_path), ConverterSettings(colors=6))
    print(f"alpha png : {time.perf_counter() - started:5.1f}s paths={doc.path_count}")

    # a 1-bit style image and a tiny image, the two extremes
    tiny = source.resize((24, 18), Image.LANCZOS)
    tiny.save(os.path.join(DATA, "tiny.png"))
    doc = convert(tiny, ConverterSettings(colors=4, upscale=4.0))
    print(f"24x18 icon: {doc.width}x{doc.height} paths={doc.path_count} "
          f"colours={len(doc.layers)}")

    print("OK - large and awkward inputs all convert")


if __name__ == "__main__":
    main()
