"""Image Vector Converter - entry point.

    python main.py                        -> open the GUI
    python main.py -i logo.jpg -o logo.eps -> convert from the command line
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from ivc import __version__
from ivc.document import save_eps, save_svg
from ivc.engine import ConverterSettings, convert, load_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="image-vector-converter",
        description="Convert a low quality bitmap into an Illustrator EPS / SVG vector file.")
    parser.add_argument("-i", "--input", help="source image (png/jpg/webp/bmp)")
    parser.add_argument("-o", "--output", help="target file (.eps or .svg)")
    parser.add_argument("-c", "--colors", type=int, default=8,
                        help="number of flat colours, 2-64 (default 8)")
    parser.add_argument("-d", "--detail", type=float, default=0.6,
                        help="curve accuracy in source pixels, lower = more accurate "
                             "(default 0.6)")
    parser.add_argument("--corner", type=float, default=75.0,
                        help="corner angle threshold in degrees (default 75)")
    parser.add_argument("--min-area", type=float, default=6.0,
                        help="drop shapes smaller than this area in source px^2 (default 6)")
    parser.add_argument("--denoise", type=int, default=6, help="JPEG denoise 0-30 (default 6)")
    parser.add_argument("--upscale", type=float, default=2.0,
                        help="upscale factor 1-4 (default 2)")
    parser.add_argument("--contrast", type=int, default=2,
                        help="local contrast boost 0-10 (default 2)")
    parser.add_argument("--sharpen", type=int, default=40, help="unsharp amount 0-200 (default 40)")
    parser.add_argument("--despeckle", type=int, default=1,
                        help="remove specks 0-10 (default 1)")
    parser.add_argument("--keep-background", action="store_true",
                        help="keep the background colour instead of making it transparent")
    parser.add_argument("-q", "--quiet", action="store_true", help="no progress output")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def run_cli(options: argparse.Namespace) -> int:
    if not options.input or not options.output:
        print("error: both --input and --output are required in CLI mode", file=sys.stderr)
        return 2

    extension = os.path.splitext(options.output)[1].lower()
    if extension not in (".eps", ".svg"):
        print("error: output must end with .eps or .svg", file=sys.stderr)
        return 2

    def report(percent: int, message: str) -> None:
        if not options.quiet:
            print(f"\r[{percent:3d}%] {message:<46}", end="", flush=True)

    image = load_image(options.input)
    settings = ConverterSettings(
        colors=options.colors,
        tolerance=options.detail,
        corner_threshold=options.corner,
        min_area=options.min_area,
        drop_background=not options.keep_background,
        denoise=options.denoise,
        upscale=options.upscale,
        contrast=options.contrast,
        sharpen=options.sharpen,
        despeckle=options.despeckle,
    )

    started = time.perf_counter()
    document = convert(image, settings, progress=report,
                       source_name=os.path.basename(options.input))
    if not options.quiet:
        print()

    if extension == ".eps":
        save_eps(document, options.output)
    else:
        save_svg(document, options.output)

    size_kb = os.path.getsize(options.output) / 1024.0
    if not options.quiet:
        print(f"colours      : {len(document.layers)}")
        print(f"paths        : {document.path_count}")
        print(f"anchors      : {document.anchor_count}")
        print(f"canvas       : {document.width} x {document.height} px "
              f"({document.trace_scale:g}x working scale)")
        print(f"output       : {options.output}  ({size_kb:.0f} KB, "
              f"{time.perf_counter() - started:.1f}s)")
    return 0


def main() -> int:
    parser = build_parser()
    options = parser.parse_args()

    if options.input or options.output:
        return run_cli(options)

    try:
        from ivc.gui import main as gui_main
    except ImportError as exc:
        print(f"error: cannot start the GUI ({exc}).\n"
              "Use CLI mode: python main.py -i input.png -o output.eps", file=sys.stderr)
        return 1
    gui_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
