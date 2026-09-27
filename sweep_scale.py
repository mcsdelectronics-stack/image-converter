"""Is the 1.5x default safe everywhere, or does it only suit the benchmark?

Checks the two cases the benchmark cannot speak for: tiny images (where there
is barely any pixel data to trace) and busy photographs.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bench_quality as B
from ivc.engine import ConverterSettings, convert


def small_cases():
    """The same logo at sizes a real user actually uploads."""
    base = B.image_logo()
    out = []
    for width in (48, 96, 150, 300, 600):
        height = int(width * base.size[1] / base.size[0])
        out.append((f"{width}px", base.resize((width, height), Image.LANCZOS)))
    return out


def main() -> None:
    head = (f"{'case':<10s} {'up':>4s} {'edge px':>8s} {'soft':>7s} {'anchors':>8s} "
            f"{'paths':>6s} {'eps KB':>8s} {'sec':>5s}")

    print("== tiny images: which upscale gives a usable trace? ==")
    print(head)
    for label, image in small_cases():
        for up in (1.5, 2.0, 3.0, 4.0):
            settings = ConverterSettings(colors=6, upscale=up, tolerance=0.6)
            started = time.perf_counter()
            doc = convert(image, settings)
            seconds = time.perf_counter() - started
            print(f"{label:<10s} {up:4.1f} {B.edge_error(image, doc):8.3f} "
                  f"{B.soft_error(image, doc):7.2f} {doc.anchor_count:8d} "
                  f"{doc.path_count:6d} "
                  f"{len(B.to_eps(doc).encode()) / 1024:8.1f} {seconds:5.1f}")

    print("\n== busy photo, more colours: does 1.5 still win? ==")
    print(head)
    photo = B.image_photo()
    for colors in (8, 16, 32):
        for up in (1.5, 2.0, 3.0):
            settings = ConverterSettings(colors=colors, upscale=up, tolerance=0.6)
            started = time.perf_counter()
            doc = convert(photo, settings)
            seconds = time.perf_counter() - started
            print(f"c={colors:<8d} {up:4.1f} {B.edge_error(photo, doc):8.3f} "
                  f"{B.soft_error(photo, doc):7.2f} {doc.anchor_count:8d} "
                  f"{doc.path_count:6d} "
                  f"{len(B.to_eps(doc).encode()) / 1024:8.1f} {seconds:5.1f}")

    print("\n== line art: 1.5 vs 2.0 at 6 and 16 colours ==")
    print(head)
    for label in ("lines", "text"):
        source = B.IMAGES[label]()
        for colors in (6, 16):
            for up in (1.5, 2.0):
                settings = ConverterSettings(colors=colors, upscale=up, tolerance=0.6)
                doc = convert(source, settings)
                print(f"{label + '/c' + str(colors):<10s} {up:4.1f} "
                      f"{B.edge_error(source, doc):8.3f} {B.soft_error(source, doc):7.2f} "
                      f"{doc.anchor_count:8d} {doc.path_count:6d} "
                      f"{len(B.to_eps(doc).encode()) / 1024:8.1f}")


if __name__ == "__main__":
    main()
