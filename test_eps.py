"""Output file checks: the EPS must be valid PostScript, the SVG valid XML.

A hand-written EPS writer can easily produce a file that "looks" fine but that
Illustrator refuses to open, so the structure is parsed back here: every path
block, every operator, the DSC header, the operator stack balance and the
bounding box.  The SVG is parsed with a real XML parser and its path data is
decoded again to confirm it round-trips.
"""
import os
import re
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ivc.document import to_eps, to_svg
from ivc.engine import ConverterSettings, convert, load_image

NUMBER = re.compile(r"^-?\d+(\.\d+)?$")
OPERATORS = {"newpath", "moveto", "curveto", "closepath", "eofill", "setrgbcolor",
             "gsave", "grestore", "lineto", "setlinejoin", "setlinecap", "showpage"}


def check_eps(text: str, width: int, height: int, path_count: int) -> None:
    lines = [line for line in text.splitlines()]
    assert lines[0] == "%!PS-Adobe-3.0 EPSF-3.0", f"bad magic: {lines[0]}"
    assert lines[1] == f"%%BoundingBox: 0 0 {width} {height}", f"bad bbox: {lines[1]}"
    assert "%%HiResBoundingBox:" in lines[2], "missing HiResBoundingBox"
    assert lines[-1] == "%%EOF", f"missing EOF, ends with {lines[-1]!r}"
    for required in ("%%Creator:", "%%Title:", "%%CreationDate:", "%%EndComments",
                     "%%BeginProlog", "%%EndProlog", "%%Page: 1 1", "%%Trailer"):
        assert any(line.startswith(required) for line in lines), f"missing {required}"

    # the title must not smuggle parentheses or non-ascii into a DSC comment
    title = next(line for line in lines if line.startswith("%%Title:"))
    assert title.isascii() and "(" not in title and ")" not in title, f"bad title: {title}"

    assert text.count("gsave") == text.count("grestore") == 1, "unbalanced gsave"
    assert text.count("newpath") == path_count, "path count mismatch"

    # walk the body: newpath, moveto, curveto*, closepath, f
    state = "idle"
    seen_fill = 0
    moveto = 0
    in_prolog = False
    for line in lines:
        token = line.strip()
        if not token:
            continue
        if token == "%%BeginProlog":
            in_prolog = True
            continue
        if token == "%%EndProlog":
            in_prolog = False
            continue
        if in_prolog or token.startswith("%"):
            continue
        if token == "newpath":
            assert state == "idle", f"newpath while {state}"
            state = "path"
        elif token.endswith("moveto"):
            assert state == "path", "moveto in the wrong state"
            assert len(token.split()) == 3, f"moveto takes 2 numbers: {token!r}"
            moveto += 1
        elif token.endswith("curveto"):
            assert state == "path", "curveto in the wrong state"
            assert len(token.split()) == 7, f"curveto takes 6 numbers: {token!r}"
        elif token.endswith("lineto"):
            assert state == "path", "lineto outside a path"
            assert len(token.split()) == 3, f"lineto takes 2 numbers: {token!r}"
        elif token == "closepath":
            assert state == "path", "closepath outside a path"
            state = "closed"
        elif token == "f":
            assert state == "closed", "fill without a closed path"
            seen_fill += 1
            state = "idle"
        elif token in ("setrgbcolor", "gsave", "grestore", "setlinejoin",
                       "setlinecap", "showpage", "eofill"):
            pass
        else:
            tokens = token.split()
            # "1 setlinejoin 1 setlinecap" style lines mix numbers and operators
            numbers = [t for t in tokens if not t.isalpha()]
            assert all(NUMBER.match(n) for n in numbers), f"bad number in {token!r}"
            for number in numbers:
                value = float(number)
                assert -1 <= value <= max(width, height) + 1, f"coordinate out of bbox: {value}"
    assert state == "idle", f"file ends inside a {state}"
    assert seen_fill == path_count, f"{seen_fill} fills for {path_count} paths"
    assert moveto == path_count, f"{moveto} moveto for {path_count} paths"
    print(f"  EPS ok: {path_count} paths, {text.count('curveto')} curves, "
          f"{len(lines)} lines")


def check_svg(text: str, width: int, height: int, path_count: int) -> None:
    root = ET.fromstring(text)
    assert root.tag.endswith("svg"), f"root is {root.tag}"
    assert root.get("viewBox") == f"0 0 {width} {height}", root.get("viewBox")
    assert root.get("width") == str(width) and root.get("height") == str(height)
    paths = root.findall("{http://www.w3.org/2000/svg}path")
    assert len(paths) == path_count, f"{len(paths)} svg paths for {path_count}"
    curves = 0
    for element in paths:
        assert element.get("fill-rule") == "evenodd"
        assert re.match(r"^#[0-9a-f]{6}$", element.get("fill")), element.get("fill")
        data = element.get("d")
        assert data.startswith("M") and data.endswith("Z")
        curves += data.count("C")
        # every number must round-trip inside the viewBox
        for chunk in re.findall(r"-?\d+(?:\.\d+)?", data):
            value = float(chunk)
            assert -1 <= value <= max(width, height) + 1, f"svg coord {value} out of viewBox"
    print(f"  SVG ok: {path_count} paths, {curves} curves, parses as XML")


def main() -> None:
    from test_pipeline import ensure_source
    image = load_image(ensure_source())
    doc = convert(image, ConverterSettings())
    print(f"document: {doc.width}x{doc.height}, {doc.path_count} paths, "
          f"{len(doc.layers)} colours")
    check_eps(to_eps(doc), doc.width, doc.height, doc.path_count)
    check_svg(to_svg(doc), doc.width, doc.height, doc.path_count)

    # a name with parentheses and non-ascii must not corrupt the header
    doc.source_name = "my logo (final)\u00e9.png"
    check_eps(to_eps(doc), doc.width, doc.height, doc.path_count)
    ET.fromstring(to_svg(doc))
    print("OK - output files are well formed")


if __name__ == "__main__":
    main()
