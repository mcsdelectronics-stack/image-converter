"""Parameter sweep: what each knob actually buys, measured.

Run `python sweep.py` to print the table.  Used to pick the shipped defaults from
evidence instead of guesswork.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bench_quality as B
from ivc.engine import ConverterSettings

# The three cases that behave like real logo work: a badge, typography, and
# thin strokes.  A busy photo is excluded on purpose - with a small palette it is
# dominated by colour error, not tracing error.
CORE = ("logo", "text", "lines")


def measure(sources, settings):
    edge, soft, anchors, kb, paths, seconds = [], [], [], [], [], 0.0
    for name in CORE:
        started = time.perf_counter()
        doc = B.convert(sources[name], settings)
        seconds += time.perf_counter() - started
        edge.append(B.edge_error(sources[name], doc))
        soft.append(B.soft_error(sources[name], doc))
        anchors.append(doc.anchor_count)
        paths.append(doc.path_count)
        kb.append(len(B.to_eps(doc).encode()) / 1024)
    return (float(np.mean(edge)), float(np.mean(soft)), float(np.mean(anchors)),
            float(np.mean(kb)), float(np.mean(paths)), seconds)


def table(rows, header, key_width=22):
    print(header)
    for label, values in rows:
        print(f"{label:<{key_width}} {values[0]:7.3f} {values[1]:7.2f} "
              f"{values[2]:8.0f} {values[3]:7.1f} {values[4]:7.0f} {values[5]:6.1f}")


def main() -> None:
    os.makedirs(B.DATA, exist_ok=True)
    sources = {name: factory() for name, factory in B.IMAGES.items()}
    head = (f"{'setting':<22} {'edge px':>7s} {'soft':>7s} {'anchors':>8s} "
            f"{'eps KB':>7s} {'paths':>7s} {'sec':>6s}")

    def row(label, **overrides):
        settings = ConverterSettings(**{"colors": 6, **overrides})
        return label, measure(sources, settings)

    print("== detail tolerance (finer fit, more anchors) ==")
    table([row(f"tol {d}", tolerance=d) for d in (0.6, 0.45, 0.35, 0.3, 0.25, 0.2)], head)

    print("\n== upscale factor ==")
    table([row(f"upscale {u}", tolerance=0.35, upscale=u) for u in (1.0, 1.5, 2.0, 3.0, 4.0)], head)

    print("\n== smoothing passes ==")
    table([row(f"smoothing {s}", tolerance=0.35, smoothing=s) for s in (0, 1, 2, 3, 4)], head)

    print("\n== denoise / despeckle / sharpen ==")
    table([
        row("denoise 0", tolerance=0.35, denoise=0.0),
        row("denoise 4", tolerance=0.35, denoise=4.0),
        row("denoise 6 (default)", tolerance=0.35, denoise=6.0),
        row("denoise 9", tolerance=0.35, denoise=9.0),
        row("despeckle 0", tolerance=0.35, despeckle=0.0),
        row("despeckle 2", tolerance=0.35, despeckle=2.0),
        row("sharpen 0", tolerance=0.35, sharpen=0.0),
        row("sharpen 70", tolerance=0.35, sharpen=70.0),
        row("contrast 0", tolerance=0.35, contrast=0.0),
    ], head)

    print("\n== corner threshold ==")
    table([row(f"corner {c}", tolerance=0.35, corner_threshold=c)
           for c in (40.0, 55.0, 75.0, 95.0)], head)

    print("\n== colour count ==")
    table([row(f"colors {c}", tolerance=0.35, colors=c)
           for c in (2, 4, 6, 8, 12, 16, 24, 32)], head)


if __name__ == "__main__":
    main()


