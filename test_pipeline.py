"""Build a deliberately low quality test logo and run the converter on it."""
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ivc.document import to_eps, to_svg
from ivc.engine import ConverterSettings, convert, render

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testdata")
os.makedirs(OUT, exist_ok=True)


def make_hi_res_logo() -> Image.Image:
    img = Image.new("RGB", (900, 600), "white")
    d = ImageDraw.Draw(img)
    # outer ring
    d.ellipse([120, 60, 780, 540], fill=(214, 40, 40))
    d.ellipse([200, 100, 700, 500], fill=(255, 255, 255))
    # inner star-ish shape
    d.ellipse([300, 150, 600, 450], fill=(30, 90, 200))
    d.ellipse([360, 190, 540, 410], fill=(255, 255, 255))
    # bars
    d.rectangle([150, 300, 300, 360], fill=(20, 160, 90))
    d.rectangle([600, 300, 750, 360], fill=(20, 160, 90))
    try:
        font = ImageFont.truetype("arialbd.ttf", 72)
    except OSError:
        font = ImageFont.load_default()
    d.text((330, 520), "VECTOR", font=font, fill=(60, 60, 70))
    return img


def degrade(img: Image.Image) -> Image.Image:
    small = img.resize((150, 100), Image.LANCZOS)      # tiny
    arr = np.asarray(small).astype(np.float32)
    arr += np.random.normal(0, 9, arr.shape)            # noise
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    noisy = Image.fromarray(arr)
    buf = os.path.join(OUT, "low_quality_logo.jpg")
    noisy.save(buf, quality=22)                          # heavy jpeg
    return Image.open(buf).convert("RGB")


def ensure_source() -> str:
    """Path of the shared test fixture, generated on first use."""
    path = os.path.join(OUT, "low_quality_logo.jpg")
    if not os.path.exists(path):
        degrade(make_hi_res_logo())
    return path


def main() -> None:
    t0 = time.perf_counter()
    hi = make_hi_res_logo()
    low = degrade(hi)
    src = os.path.join(OUT, "low_quality_logo.jpg")
    print(f"source: {low.size[0]}x{low.size[1]}  ({os.path.getsize(src)/1024:.1f} KB)")

    settings = ConverterSettings(colors=8, tolerance=0.7, min_area=6.0,
                                 drop_background=True, denoise=8, upscale=4.0,
                                 contrast=2, sharpen=50, despeckle=2)
    doc = convert(low, settings, progress=lambda p, m: print(f"  [{p:3d}%] {m}"))

    print(f"\ncolours   : {len(doc.layers)}")
    print(f"paths     : {doc.path_count}")
    print(f"anchors   : {doc.anchor_count}")
    print(f"canvas    : {doc.width}x{doc.height} (scale {doc.trace_scale:g})")
    print(f"background: {doc.background}")
    for layer in doc.layers:
        print(f"   {layer.color}  paths={layer.path_count}")

    eps = os.path.join(OUT, "vector_logo.eps")
    svg = os.path.join(OUT, "vector_logo.svg")
    from ivc.document import save_eps, save_svg
    save_eps(doc, eps)
    save_svg(doc, svg)
    print(f"\nEPS: {os.path.getsize(eps)/1024:.1f} KB")
    print(f"SVG: {os.path.getsize(svg)/1024:.1f} KB")

    # validity checks
    text = to_eps(doc)
    assert text.startswith("%!PS-Adobe-3.0 EPSF-3.0"), "missing EPS header"
    assert "%%BoundingBox:" in text and "%%EOF" in text, "missing EPS structure"
    assert text.count("newpath") == text.count("\nf\n") == doc.path_count, \
        f"path/fill mismatch {text.count('newpath')} {text.count(chr(10)+'f'+chr(10))} {doc.path_count}"
    for layer in doc.layers:
        for path in layer.paths:
            for bez in path:
                for x, y in bez:
                    assert np.isfinite(x) and np.isfinite(y), "NaN in path"
                    assert -2 <= x <= doc.width + 2 and -2 <= y <= doc.height + 2, \
                        f"point outside canvas: {x},{y}"

    svg_text = to_svg(doc)
    assert svg_text.count("<path") == doc.path_count

    prev = render(doc, max_size=(900, 900), background=(255, 255, 255))
    prev.save(os.path.join(OUT, "vector_preview.png"))
    low.save(os.path.join(OUT, "source.png"))

    # compare preview against the source for a sanity score
    src_small = low.resize(prev.size, Image.LANCZOS)
    a = np.asarray(src_small.convert("RGB"), dtype=np.float32)
    b = np.asarray(prev.convert("RGB"), dtype=np.float32)
    mae = float(np.abs(a - b).mean())
    print(f"mean abs error vs source: {mae:.1f} / 255")
    print(f"total {time.perf_counter()-t0:.1f}s  -> OK")


if __name__ == "__main__":
    main()
