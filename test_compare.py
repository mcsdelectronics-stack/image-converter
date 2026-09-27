"""Render side-by-side comparisons so the vector quality can be eyeballed."""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ivc.engine import ConverterSettings, convert, render
from test_pipeline import ensure_source, make_hi_res_logo

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "testdata")
ZOOM = 3


def main() -> None:
    source = Image.open(ensure_source()).convert("RGB")
    ref = make_hi_res_logo()
    ref_small = ref.resize(source.size, Image.LANCZOS)

    settings = ConverterSettings(colors=8, tolerance=1.0, min_area=8.0, smoothing=3,
                                 drop_background=True, denoise=8, upscale=4.0,
                                 contrast=2, sharpen=50, despeckle=1)
    doc = convert(source, settings)
    vector = render(doc, max_size=source.size, background=(255, 255, 255)).convert("RGB")

    panels = [("SOURCE (150x100 jpeg)", source),
              ("ORIGINAL (downscaled)", ref_small),
              ("VECTOR (rendered)", vector)]
    w, h = source.size
    total = Image.new("RGB", (w * ZOOM * 3 + 40, h * ZOOM + 30), "white")
    draw = ImageDraw.Draw(total)
    for i, (title, panel) in enumerate(panels):
        big = panel.resize((w * ZOOM, h * ZOOM), Image.NEAREST)
        x = 10 + i * (w * ZOOM + 10)
        total.paste(big, (x, 25))
        draw.text((x, 8), title, fill="black")
    path = os.path.join(OUT, "compare.png")
    total.save(path)
    print(path)

    # zoomed crop of the ring to inspect curve quality
    box = (20, 20, 70, 60)
    crops = [p.crop(box).resize((50 * 10, 40 * 10), Image.NEAREST) for _, p in panels]
    strip = Image.new("RGB", (50 * 10 * 3 + 20, 40 * 10), "white")
    for i, crop in enumerate(crops):
        strip.paste(crop, (i * (50 * 10 + 10), 0))
    strip_path = os.path.join(OUT, "compare_zoom.png")
    strip.save(strip_path)
    print(strip_path)

    for name, panel in panels:
        a = np.asarray(panel, np.float32)
        b = np.asarray(vector, np.float32)
        print(f"{name:24s} mae vs vector = {np.abs(a - b).mean():5.1f}")


if __name__ == "__main__":
    main()
