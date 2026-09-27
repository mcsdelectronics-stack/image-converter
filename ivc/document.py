"""Vector document model plus the EPS and SVG writers.

The EPS output is plain, Illustrator-friendly PostScript: every colour is a
filled path using the even-odd rule, so holes (an "o" in a logo) come out
correct.  Nothing exotic is emitted, which means the file opens in Illustrator,
CorelDRAW, Inkscape, Affinity Designer and previews in any web browser.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from .geometry import Bezier, Path

Rgb = Tuple[int, int, int]


@dataclass
class VectorLayer:
    color: Rgb
    paths: List[Path] = field(default_factory=list)

    @property
    def path_count(self) -> int:
        return len(self.paths)


@dataclass
class VectorDocument:
    width: int
    height: int
    layers: List[VectorLayer] = field(default_factory=list)
    background: Optional[Rgb] = None
    source_name: str = "image"
    source_size: Tuple[int, int] = (0, 0)
    trace_scale: float = 1.0

    @property
    def path_count(self) -> int:
        return sum(layer.path_count for layer in self.layers)

    @property
    def anchor_count(self) -> int:
        return sum(len(path) for layer in self.layers for path in layer.paths)

    @property
    def is_transparent(self) -> bool:
        return self.background is None


# --------------------------------------------------------------------------- #
# number formatting
# --------------------------------------------------------------------------- #
def _num(value: float, digits: int = 2) -> str:
    text = f"{value:.{digits}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in ("", "-", "-0"):
        text = "0"
    return text


def _hex_color(color: Rgb) -> str:
    r, g, b = (max(0, min(255, int(c))) for c in color)
    return f"#{r:02x}{g:02x}{b:02x}"


# --------------------------------------------------------------------------- #
# SVG
# --------------------------------------------------------------------------- #
def _svg_path(path: Sequence[Bezier]) -> str:
    """Path data with 0.1 px precision - far finer than the trace tolerance."""
    chunks: List[str] = []
    first = path[0]
    chunks.append(f"M{_num(first[0][0], 1)} {_num(first[0][1], 1)}")
    for bez in path:
        chunks.append(
            "C{0} {1} {2} {3} {4} {5}".format(
                _num(bez[1][0], 1), _num(bez[1][1], 1),
                _num(bez[2][0], 1), _num(bez[2][1], 1),
                _num(bez[3][0], 1), _num(bez[3][1], 1)))
    chunks.append("Z")
    return "".join(chunks)


def to_svg(doc: VectorDocument) -> str:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
        f'width="{doc.width}" height="{doc.height}" '
        f'viewBox="0 0 {doc.width} {doc.height}">',
        f"<title>{_escape(doc.source_name)}</title>",
    ]
    if doc.background is not None:
        lines.append(f'<rect width="{doc.width}" height="{doc.height}" '
                     f'fill="{_hex_color(doc.background)}"/>')
    for layer in doc.layers:
        color = _hex_color(layer.color)
        for path in layer.paths:
            lines.append(f'<path fill="{color}" fill-rule="evenodd" '
                         f'shape-rendering="geometricPrecision" '
                         f'd="{_svg_path(path)}"/>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _escape(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _dsc_safe(text: str) -> str:
    """Make a filename safe for a DSC comment (ASCII, no stray parentheses)."""
    cleaned = "".join(ch if 32 <= ord(ch) < 127 else "_" for ch in text)
    return cleaned.replace("(", "[").replace(")", "]") or "image"


# --------------------------------------------------------------------------- #
# EPS / PostScript
# --------------------------------------------------------------------------- #
def _eps_path(path: Sequence[Bezier], height: int) -> str:
    """Emit a path, flipping the Y axis into PostScript's bottom-left origin."""
    out: List[str] = ["newpath"]
    start = path[0]
    out.append(f"{_num(start[0][0], 1)} {_num(height - start[0][1], 1)} moveto")
    for bez in path:
        out.append("{0} {1} {2} {3} {4} {5} curveto".format(
            _num(bez[1][0], 1), _num(height - bez[1][1], 1),
            _num(bez[2][0], 1), _num(height - bez[2][1], 1),
            _num(bez[3][0], 1), _num(height - bez[3][1], 1)))
    out.append("closepath")
    return "\n".join(out)


def to_eps(doc: VectorDocument) -> str:
    width, height = doc.width, doc.height
    created = _dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    title = _dsc_safe(doc.source_name)

    header = [
        "%!PS-Adobe-3.0 EPSF-3.0",
        f"%%BoundingBox: 0 0 {int(width)} {int(height)}",
        f"%%HiResBoundingBox: 0 0 {_num(float(width))} {_num(float(height))}",
        "%%LanguageLevel: 2",
        "%%Creator: (Image Vector Converter 1.0)",
        f"%%Title: {title}",
        f"%%CreationDate: {created}",
        "%%DocumentData: Clean7Bit",
        "%%EndComments",
        "%%BeginProlog",
        "/m { moveto } bind def",
        "/c { curveto } bind def",
        "/h { closepath } bind def",
        "/f { eofill } bind def",
        "/rgb { setrgbcolor } bind def",
        "%%EndProlog",
        "%%Page: 1 1",
        "gsave",
        "1 setlinejoin 1 setlinecap",
    ]

    body: List[str] = []
    if doc.background is not None:
        r, g, b = (c / 255.0 for c in doc.background)
        body.append(f"{_num(r, 4)} {_num(g, 4)} {_num(b, 4)} rgb")
        body.append("newpath 0 0 moveto "
                    f"{_num(float(width))} 0 lineto "
                    f"{_num(float(width))} {_num(float(height))} lineto "
                    f"0 {_num(float(height))} lineto closepath f")

    for layer in doc.layers:
        if not layer.paths:
            continue
        r, g, b = (c / 255.0 for c in layer.color)
        body.append(f"{_num(r, 4)} {_num(g, 4)} {_num(b, 4)} rgb")
        for path in layer.paths:
            body.append(_eps_path(path, height))
            body.append("f")

    footer = ["grestore", "showpage", "%%Trailer", "%%EOF"]
    return "\n".join(header + body + footer) + "\n"


# --------------------------------------------------------------------------- #
# convenience writers
# --------------------------------------------------------------------------- #
def save_svg(doc: VectorDocument, filename: str, encoding: str = "utf-8") -> None:
    with open(filename, "w", encoding=encoding, newline="\n") as handle:
        handle.write(to_svg(doc))


def save_eps(doc: VectorDocument, filename: str, encoding: str = "ascii") -> None:
    with open(filename, "w", encoding=encoding, newline="\n") as handle:
        handle.write(to_eps(doc))
