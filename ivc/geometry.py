"""Polygon to cubic Bezier conversion.

The tracer hands us staircase pixel outlines.  This module turns those outlines
into smooth cubic Bezier paths using the classic Schneider fitting pipeline:

    polygon -> Douglas-Peucker simplification -> corner detection
            -> least-squares cubic fit with Newton-Raphson reparameterisation
            -> recursive subdivision whenever a span misses the tolerance

Result: vector output that stays smooth at any zoom level, which is exactly what
a raster -> vector converter needs to beat the "blocky staircase" look.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

Point = Tuple[float, float]
Bezier = Tuple[Point, Point, Point, Point]  # start, control1, control2, end
Path = List[Bezier]

_EPS = 1e-12
_MAX_FIT_DEPTH = 12
_MAX_NR_ITER = 4
_NR_TOL = 1e-3


# --------------------------------------------------------------------------- #
# small vector helpers
# --------------------------------------------------------------------------- #
def _sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1])


def _add(a: Point, b: Point) -> Point:
    return (a[0] + b[0], a[1] + b[1])


def _scale(a: Point, k: float) -> Point:
    return (a[0] * k, a[1] * k)


def _dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _length(a: Point) -> float:
    return math.hypot(a[0], a[1])


def _length_sq(a: Point) -> float:
    return a[0] * a[0] + a[1] * a[1]


def _normalize(a: Point) -> Point:
    length = _length(a)
    if length < _EPS:
        return (0.0, 0.0)
    return (a[0] / length, a[1] / length)


def _unit_tangent(p0: Point, p1: Point, fallback: Point) -> Point:
    """Direction p0 -> p1, falling back when the two points coincide."""
    delta = _normalize(_sub(p1, p0))
    if delta == (0.0, 0.0):
        return fallback
    return delta


def _line_segment(p0: Point, p1: Point) -> Bezier:
    return (p0, _add(p0, _scale(_sub(p1, p0), 1.0 / 3.0)),
            _add(p0, _scale(_sub(p1, p0), 2.0 / 3.0)), p1)


# --------------------------------------------------------------------------- #
# polygon preparation
# --------------------------------------------------------------------------- #
def dedupe(points: Sequence[Point], tolerance: float = 1e-9) -> List[Point]:
    """Drop consecutive duplicate points."""
    out: List[Point] = []
    for p in points:
        if not out or _length(_sub(p, out[-1])) > tolerance:
            out.append((float(p[0]), float(p[1])))
    while len(out) > 1 and _length(_sub(out[0], out[-1])) <= tolerance:
        out.pop()
    return out


def polygon_area(points: Sequence[Point]) -> float:
    """Signed shoelace area."""
    total = 0.0
    n = len(points)
    for i in range(n):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n]
        total += x0 * y1 - x1 * y0
    return total * 0.5


def simplify(points: Sequence[Point], epsilon: float) -> List[Point]:
    """Douglas-Peucker simplification (iterative, no recursion limit issues)."""
    n = len(points)
    if n < 3 or epsilon <= 0.0:
        return list(points)

    keep = [False] * n
    keep[0] = keep[n - 1] = True
    stack = [(0, n - 1)]

    while stack:
        first, last = stack.pop()
        if last <= first + 1:
            continue
        ax, ay = points[first]
        bx, by = points[last]
        dx, dy = bx - ax, by - ay
        seg_len2 = dx * dx + dy * dy
        best_index = -1
        best_dist = -1.0
        for i in range(first + 1, last):
            px, py = points[i]
            if seg_len2 <= _EPS:
                dist = math.hypot(px - ax, py - ay)
            else:
                t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len2))
                dist = math.hypot(px - (ax + t * dx), py - (ay + t * dy))
            if dist > best_dist:
                best_dist = dist
                best_index = i
        if best_dist > epsilon:
            keep[best_index] = True
            stack.append((first, best_index))
            stack.append((best_index, last))

    return [points[i] for i in range(n) if keep[i]]


def find_corners(points: Sequence[Point], threshold_deg: float) -> List[int]:
    """Indices where the outline turns sharply (real corners, not curve noise)."""
    n = len(points)
    if n < 3:
        return [0]
    limit = math.radians(max(1.0, min(180.0, threshold_deg)))
    corners = [0]
    for i in range(1, n - 1):
        a1 = math.atan2(points[i][1] - points[i - 1][1], points[i][0] - points[i - 1][0])
        a2 = math.atan2(points[i + 1][1] - points[i][1], points[i + 1][0] - points[i][0])
        turn = abs((a2 - a1 + math.pi) % (2.0 * math.pi) - math.pi)
        if turn > limit:
            corners.append(i)
    return corners


def _fit_line(points: Sequence[Point]) -> Tuple[Point, Point]:
    """Centroid and principal direction of a point run (total least squares).

    The corner point itself is deliberately excluded by the caller, so this
    describes the straight edge arriving at the corner rather than the corner
    neighbourhood as a whole.
    """
    n = len(points)
    cx = sum(p[0] for p in points) / n
    cy = sum(p[1] for p in points) / n
    sxx = sum((p[0] - cx) ** 2 for p in points)
    syy = sum((p[1] - cy) ** 2 for p in points)
    sxy = sum((p[0] - cx) * (p[1] - cy) for p in points)
    theta = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    return (cx, cy), (math.cos(theta), math.sin(theta))


def _line_intersection(a: Tuple[Point, Point], b: Tuple[Point, Point]) -> Optional[Point]:
    (ax, ay), da = a
    (bx, by), db = b
    det = da[0] * (-db[1]) - da[1] * (-db[0])
    if abs(det) < 1e-9:
        return None
    diff = (bx - ax, by - ay)
    t = (diff[0] * (-db[1]) - diff[1] * (-db[0])) / det
    return (ax + t * da[0], ay + t * da[1])


def refine_corners(points: Sequence[Point], corner_indices: Sequence[int],
                   span: float = 2.0, max_shift: float = 2.0,
                   min_angle_deg: float = 15.0) -> Tuple[List[Point], List[Optional[Tuple[Point, Point]]]]:
    """Snap every corner onto the intersection of the two edges meeting there.

    A corner detected on a pixel outline is a pixel *centre*, so it can sit up to
    half a pixel away from the real corner - which is what makes traced logos
    look slightly blunt.  The two straight runs leading into and out of the
    corner describe the edges much better than the single corner pixel does, and
    where those two edges cross is the true corner.

    Returns the refined point list plus, per corner, the pair of edge direction
    vectors (incoming, outgoing) when refinement was possible, so the neighbouring
    arcs can leave and arrive along the real edges instead of along a diagonal.
    """
    n = len(points)
    refined = list(points)
    edges: List[Optional[Tuple[Point, Point]]] = [None] * len(corner_indices)
    if n < 8 or len(corner_indices) == 0:
        return refined, edges

    min_sin = math.sin(math.radians(max(1.0, min(80.0, min_angle_deg))))
    # A window that grows with the outline but stays small enough to describe a
    # straight run rather than the neighbouring curve.
    window = max(2, min(6, n // 8))

    for slot, index in enumerate(corner_indices):
        if index <= 0 or index >= n - 1:
            continue
        before = [points[(index - k) % n] for k in range(1, window + 1)]
        after = [points[(index + k) % n] for k in range(1, window + 1)]
        # only trust a run that really is straight and long enough to fit
        if (_length(_sub(before[0], before[-1])) < span * 0.5
                or _length(_sub(after[0], after[-1])) < span * 0.5):
            continue
        left = _fit_line(before)
        right = _fit_line(after)
        cross = left[1][0] * right[1][1] - left[1][1] * right[1][0]
        if abs(cross) < min_sin:          # parallel: no corner to find
            continue
        hit = _line_intersection(left, right)
        if hit is None:
            continue
        if _length_sq(_sub(hit, points[index])) > max_shift * max_shift:
            continue                      # the "corner" was really curve noise
        refined[index] = hit
        incoming = _normalize(_sub(hit, left[0]))
        outgoing = _normalize(_sub(right[0], hit))
        if incoming == (0.0, 0.0) or outgoing == (0.0, 0.0):
            refined[index] = points[index]
            continue
        edges[slot] = (incoming, outgoing)
    return refined, edges


# --------------------------------------------------------------------------- #
# Bezier evaluation
# --------------------------------------------------------------------------- #
def bezier_point(bez: Bezier, t: float) -> Point:
    mt = 1.0 - t
    a, b, c, d = mt * mt * mt, 3.0 * mt * mt * t, 3.0 * mt * t * t, t * t * t
    return (a * bez[0][0] + b * bez[1][0] + c * bez[2][0] + d * bez[3][0],
            a * bez[0][1] + b * bez[1][1] + c * bez[2][1] + d * bez[3][1])


def bezier_derivative(bez: Bezier, t: float) -> Point:
    mt = 1.0 - t
    a, b, c = 3.0 * mt * mt, 6.0 * mt * t, 3.0 * t * t
    return (a * (bez[1][0] - bez[0][0]) + b * (bez[2][0] - bez[1][0]) + c * (bez[3][0] - bez[2][0]),
            a * (bez[1][1] - bez[0][1]) + b * (bez[2][1] - bez[1][1]) + c * (bez[3][1] - bez[2][1]))


def bezier_second_derivative(bez: Bezier, t: float) -> Point:
    mt = 1.0 - t
    a, b = 6.0 * mt, 6.0 * t
    return (a * (bez[2][0] - 2.0 * bez[1][0] + bez[0][0]) + b * (bez[3][0] - 2.0 * bez[2][0] + bez[1][0]),
            a * (bez[2][1] - 2.0 * bez[1][1] + bez[0][1]) + b * (bez[3][1] - 2.0 * bez[2][1] + bez[1][1]))


# --------------------------------------------------------------------------- #
# curve fitting
# --------------------------------------------------------------------------- #
def _chord_parameterize(points: Sequence[Point]) -> List[float]:
    us = [0.0] * len(points)
    total = 0.0
    for i in range(1, len(points)):
        total += _length(_sub(points[i], points[i - 1]))
        us[i] = total
    if total < _EPS:
        return [i / max(1, len(points) - 1) for i in range(len(points))]
    return [u / total for u in us]


def _generate_bezier(points: Sequence[Point], us: Sequence[float],
                     t1: Point, t2: Point) -> Bezier:
    """Least-squares cubic through `points` with the given end tangents."""
    p0, p3 = points[0], points[-1]
    c00 = c01 = c11 = x0 = x1 = 0.0
    for i, p in enumerate(points):
        u = us[i]
        mu = 1.0 - u
        a1 = 3.0 * mu * mu * u
        a2 = 3.0 * mu * u * u
        c00 += a1 * a1
        c01 += a1 * a2
        c11 += a2 * a2
        base = _sub(p, _add(_scale(p0, u * u * u), _scale(p3, mu * mu * mu)))
        x0 += a1 * _dot(base, t1)
        x1 += a2 * _dot(base, t2)

    det = c00 * c11 - c01 * c01
    if abs(det) < _EPS:
        alpha1 = alpha2 = 0.0
    else:
        alpha1 = (x0 * c11 - x1 * c01) / det
        alpha2 = (c00 * x1 - c01 * x0) / det

    seg_len = _length(_sub(p3, p0))
    eps = 1e-6 * seg_len
    if alpha1 < eps or alpha2 < eps:
        # tangents point the wrong way (a cusp) -> fall back to a straight line
        return _line_segment(p0, p3)
    return (p0, _add(p0, _scale(t1, alpha1)), _add(p3, _scale(t2, alpha2)), p3)


def _max_error(points: Sequence[Point], bez: Bezier,
               us: Sequence[float]) -> Tuple[float, int]:
    """Largest squared distance from a data point to the curve, plus where it is.

    Squared distances are used throughout, so the fit tolerance stays a single
    squared number all the way down the recursion.
    """
    split = len(points) // 2
    max_sq = 0.0
    for i in range(1, len(points) - 1):
        dist_sq = _length_sq(_sub(points[i], bezier_point(bez, us[i])))
        if dist_sq > max_sq:
            max_sq = dist_sq
            split = i
    return max_sq, split


def _reparameterize(points: Sequence[Point], us: List[float], bez: Bezier) -> List[float]:
    """Newton-Raphson projection of data points onto the curve parameters."""
    out = [0.0] * len(points)
    for i, p in enumerate(points):
        t = us[i]
        for _ in range(_MAX_NR_ITER):
            q = bezier_point(bez, t)
            q1 = bezier_derivative(bez, t)
            q2 = bezier_second_derivative(bez, t)
            diff = _sub(p, q)
            num = _dot(diff, q1)
            den = _dot(q1, q1) + _dot(diff, q2)
            if abs(den) < _EPS:
                break
            step = num / den
            new_t = t - step
            if new_t < 0.0 or new_t > 1.0:
                # bisect on the monotone x component instead
                lo, hi = (0.0, t) if step > 0.0 else (t, 1.0)
                for _ in range(24):
                    mid = 0.5 * (lo + hi)
                    if bezier_point(bez, mid)[0] < p[0]:
                        lo = mid
                    else:
                        hi = mid
                new_t = 0.5 * (lo + hi)
            t = new_t
            if abs(step) < _NR_TOL:
                break
        out[i] = t
    if min(out) < 0.0:
        return _chord_parameterize(points)
    return out


def fit_cubic(points: Sequence[Point], t1: Point, t2: Point, error: float,
              depth: int = 0) -> Path:
    """Fit a chain of cubic Beziers through `points`.

    `error` is the squared tolerance (px^2), matching the squared distances that
    `_max_error` reports.
    """
    return [bez for bez, _, _ in fit_spans(points, t1, t2, error, depth)]


def fit_spans(points: Sequence[Point], t1: Point, t2: Point, error: float,
              depth: int = 0) -> List[Tuple[Bezier, int, int]]:
    """`fit_cubic` that also reports which input points each segment covers."""
    count = len(points) - 1
    if count < 1:
        return []

    if count == 1:
        deviation_sq = _length_sq(_sub(points[0], points[1])) / 9.0
        if deviation_sq < error or depth >= _MAX_FIT_DEPTH:
            return [(_line_segment(points[0], points[1]), 0, 1)]
        mid = (0.5 * (points[0][0] + points[1][0]), 0.5 * (points[0][1] + points[1][1]))
        return [(_line_segment(points[0], mid), 0, 0), (_line_segment(mid, points[1]), 1, 1)]

    us = _chord_parameterize(points)
    bez = _generate_bezier(points, us, t1, t2)
    max_error, split = _max_error(points, bez, us)

    if max_error < error:
        return [(bez, 0, len(points) - 1)]
    if max_error < error * error:  # close enough, worth reparameterising
        for _ in range(_MAX_NR_ITER):
            us = _reparameterize(points, us, bez)
            bez = _generate_bezier(points, us, t1, t2)
            max_error, split = _max_error(points, bez, us)
            if max_error < error:
                return [(bez, 0, len(points) - 1)]

    if depth >= _MAX_FIT_DEPTH:
        return [(bez, 0, len(points) - 1)]

    split = max(1, min(split, len(points) - 2))
    center = _unit_tangent(points[split - 1], points[split + 1], t2)
    left = fit_spans(points[:split + 1], t1, center, error, depth + 1)
    right = fit_spans(points[split:], _scale(center, -1.0), t2, error, depth + 1)
    out: List[Tuple[Bezier, int, int]] = [(bez, start, stop) for bez, start, stop in left]
    out.extend((bez, start + split, stop + split) for bez, start, stop in right)
    return out


# --------------------------------------------------------------------------- #
# control point refinement
# --------------------------------------------------------------------------- #
def _slide_controls(data: Sequence[Point], us: Sequence[float],
                    t1: Point, t2: Point, bez: Bezier) -> Optional[Bezier]:
    """One Gauss-Newton move of the two control points along their own tangents.

    The fit above already solves for the control points in closed form, but it
    does so for a parameterisation that is only an estimate.  This keeps the
    tangents and the parameterisation fixed and takes a single least-squares
    step on the two control-point lengths, which is the cheapest way to shave
    the last bit of error off a segment and to even out the tension where two
    segments meet.
    """
    p0, p3 = bez[0], bez[3]
    chord = _length(_sub(p3, p0))
    if chord < _EPS:
        return None

    c00 = c01 = c11 = g0 = g1 = 0.0
    for point, u in zip(data, us):
        mu = 1.0 - u
        a1 = 3.0 * mu * mu * u
        a2 = 3.0 * mu * u * u
        q = bezier_point(bez, u)
        rx = q[0] - point[0]
        ry = q[1] - point[1]
        j1x, j1y = a1 * t1[0], a1 * t1[1]
        j2x, j2y = a2 * t2[0], a2 * t2[1]
        c00 += j1x * j1x + j1y * j1y
        c01 += j1x * j2x + j1y * j2y
        c11 += j2x * j2x + j2y * j2y
        g0 += j1x * rx + j1y * ry
        g1 += j2x * rx + j2y * ry

    det = c00 * c11 - c01 * c01
    if abs(det) < _EPS:
        return None
    step1 = (g0 * c11 - g1 * c01) / det
    step2 = (c00 * g1 - c01 * g0) / det

    limit = 0.5 * chord
    step1 = max(-limit, min(limit, step1))
    step2 = max(-limit, min(limit, step2))
    len1 = _dot(_sub(bez[1], p0), t1) + step1
    len2 = _dot(_sub(bez[2], p3), t2) + step2
    if len1 <= 1e-6 * chord or len2 <= 1e-6 * chord:
        return None  # a zero-length handle would fold the curve back on itself
    return (p0, _add(p0, _scale(t1, len1)), _add(p3, _scale(t2, len2)), p3)


def fit_arc(points: Sequence[Point], t1: Point, t2: Point, error: float,
            passes: int = 2) -> Path:
    """Fit one arc between two anchors, then polish every segment's control points.

    A polished segment is only kept when it is genuinely closer to the outline,
    so this can improve a fit but never make one worse.
    """
    spans = fit_spans(points, t1, t2, error)
    if passes <= 0 or not spans:
        return [bez for bez, _, _ in spans]

    out: Path = []
    for bez, start, stop in spans:
        data = list(points[start:stop + 1])
        if len(data) < 3:
            out.append(bez)
            continue
        dir_in = _normalize(_sub(bez[1], bez[0]))
        dir_out = _normalize(_sub(bez[2], bez[3]))
        if dir_in == (0.0, 0.0) or dir_out == (0.0, 0.0):
            out.append(bez)
            continue
        us = _chord_parameterize(data)
        best = bez
        best_error = _max_error(data, best, us)[0]
        for _ in range(int(passes)):
            candidate = _slide_controls(data, us, dir_in, dir_out, best)
            if candidate is None:
                break
            error_now = _max_error(data, candidate, us)[0]
            if error_now >= best_error:
                break
            best = candidate
            best_error = error_now
        out.append(best)
    return out


def smooth_polygon(points: Sequence[Point], passes: int = 3) -> List[Point]:
    """Binomial smoothing of a closed outline.

    Traced pixel boundaries wobble by a pixel here and there, and a fitter that
    honours every wobble has to emit a dozen anchors for what is really a
    straight edge.  Two or three passes of a [1 2 1]/4 filter remove that
    jitter while leaving corners and large curves essentially untouched.
    """
    pts = list(points)
    if passes <= 0 or len(pts) < 5:
        return pts
    n = len(pts)
    for _ in range(int(passes)):
        out: List[Point] = []
        for i in range(n):
            x_prev, y_prev = pts[i - 1]
            x_cur, y_cur = pts[i]
            x_next, y_next = pts[(i + 1) % n]
            out.append((0.25 * x_prev + 0.5 * x_cur + 0.25 * x_next,
                        0.25 * y_prev + 0.5 * y_cur + 0.25 * y_next))
        pts = out
    return pts


def _anchor_tangents(pts: Sequence[Point], index: int, smooth: bool) -> Tuple[Point, Point]:
    """Incoming and outgoing tangent directions at an anchor point.

    At a real corner the two directions differ, which is what keeps the corner
    crisp.  At a virtual (seam) anchor they are averaged into one direction so
    the two neighbouring arcs join without a visible kink.
    """
    n = len(pts)
    prev_point = pts[index - 1]
    point = pts[index]
    next_point = pts[(index + 1) % n]

    incoming = _unit_tangent(prev_point, point, (1.0, 0.0))
    outgoing = _unit_tangent(point, next_point, incoming)
    if not smooth:
        return incoming, outgoing

    averaged = _normalize((incoming[0] + outgoing[0], incoming[1] + outgoing[1]))
    if averaged == (0.0, 0.0):  # a full reversal: keep them apart rather than fail
        return incoming, outgoing
    return averaged, averaged


def is_dense_outline(points: Sequence[Point], max_edge: float = 2.5) -> bool:
    """True when the outline is a fine pixel staircase rather than coarse edges.

    Smoothing is only safe on a densely sampled outline.  Averaging over a
    four-point rectangle would drag every corner 25% of a side length inwards,
    so the density check below is what keeps flat shapes intact.
    """
    n = len(points)
    if n < 8:
        return False
    perimeter = 0.0
    for i in range(n):
        perimeter += _length(_sub(points[i], points[(i + 1) % n]))
    return perimeter / n <= max_edge


def fit_contour(points: Sequence[Point], tolerance: float,
                corner_threshold: float = 75.0,
                simplify_epsilon: float = 0.35,
                smoothing: int = 3,
                pixel_scale: float = 1.0) -> Path:
    """Convert one closed outline into a closed Bezier path.

    `pixel_scale` is the working-resolution upscale factor.  All pixel
    measurements inside are divided by it, so every caller can speak in source
    pixels and get the same curve regardless of the working resolution.

    Smoothing happens on the dense pixel outline *before* simplification: on a
    dense outline the smoothing barely moves anything, whereas smoothing the
    already-thinned points visibly shrinks every curve.

    The outline is then cut into arcs at its corners and each arc is fitted on
    its own.  A smooth closed outline still needs at least two cut points,
    otherwise a single fit would have to wrap all the way around back onto its
    own start, which is a degenerate problem - hence the extra seam anchor.
    """
    pts = dedupe(points)
    if len(pts) < 3:
        return []
    if smoothing > 0 and is_dense_outline(pts, max_edge=2.5 * pixel_scale):
        pts = smooth_polygon(pts, smoothing)
    if simplify_epsilon > 0.0:
        pts = simplify(pts, simplify_epsilon)
    if len(pts) < 3:
        return []
    if abs(polygon_area(pts)) < 0.35:
        return []

    corners = find_corners(pts, corner_threshold)
    if len(corners) < 2:
        # No real corner at all: cut the smooth loop in two with a seam anchor.
        opposite = max(range(len(pts)),
                       key=lambda i: _length_sq(_sub(pts[i], pts[0])))
        if _length_sq(_sub(pts[opposite], pts[0])) < 16.0:
            return []  # blob too small to be worth two anchors
        corners = sorted({corners[0], opposite})
        smooth_flags = [True] * len(corners)
    else:
        smooth_flags = [False] * len(corners)

    # Pull every corner onto the intersection of its two edges, and remember
    # those edge directions: the arcs must leave and arrive along the real edge
    # or the corner stays blunt even though its position is right.
    pts, corner_edges = refine_corners(pts, corners,
                                       span=2.0 * pixel_scale,
                                       max_shift=2.0 * pixel_scale)

    tangents: List[Tuple[Point, Point]] = []
    for slot, (index, smooth) in enumerate(zip(corners, smooth_flags)):
        edge = corner_edges[slot]
        if edge is None or smooth:
            tangents.append(_anchor_tangents(pts, index, smooth))
        else:
            # Follow the fitted edges in and out of the corner.
            tangents.append((edge[0], edge[1]))

    error = max(tolerance, 1e-3) ** 2
    path: Path = []
    count = len(corners)
    for i in range(count):
        start = corners[i]
        end = corners[(i + 1) % count]
        arc = pts[start:end + 1] if end > start else pts[start:] + pts[:end + 1]
        if len(arc) < 2:
            continue
        path.extend(fit_arc(arc, tangents[i][1], tangents[(i + 1) % count][0], error))
    return path


# --------------------------------------------------------------------------- #
# flattening (preview rendering + validation)
# --------------------------------------------------------------------------- #
def flatten_bezier(bez: Bezier, steps: int) -> List[Point]:
    points = [bez[0]]
    for i in range(1, steps + 1):
        points.append(bezier_point(bez, i / steps))
    return points


def flatten_path(path: Sequence[Bezier], target_step: float = 1.0) -> List[Point]:
    """Flatten a path into a point ring for raster previews."""
    out: List[Point] = []
    for bez in path:
        if not out:
            out.append(bez[0])
        span = _length(_sub(bez[3], bez[0]))
        steps = int(max(4, min(64, span / max(target_step, 0.25) + 0.5)))
        out.extend(flatten_bezier(bez, steps)[1:])
    if len(out) > 1 and _length(_sub(out[0], out[-1])) < target_step:
        out.pop()
    return out


def path_bounds(path: Sequence[Bezier]) -> Tuple[float, float, float, float]:
    xs: List[float] = []
    ys: List[float] = []
    for bez in path:
        for p in bez:
            xs.append(p[0])
            ys.append(p[1])
    if not xs:
        return (0.0, 0.0, 0.0, 0.0)
    return (min(xs), min(ys), max(xs), max(ys))
