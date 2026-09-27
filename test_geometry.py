"""Geometry unit checks: circles, squares, corners, tolerance behaviour."""
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ivc import geometry


def circle_contour(cx, cy, r, size=400):
    mask = np.zeros((size, size), np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return [(float(p[0]), float(p[1])) for p in contours[0].reshape(-1, 2)], r


def square_contour(x0, y0, x1, y1):
    mask = np.zeros((300, 300), np.uint8)
    cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return [(float(p[0]), float(p[1])) for p in contours[0].reshape(-1, 2)]


def max_deviation_from_circle(path, cx, cy, r):
    """Largest radial distance between the flattened path and the true circle."""
    ring = geometry.flatten_path(path, target_step=0.05)
    d = [abs(math.hypot(p[0] - cx, p[1] - cy) - r) for p in ring]
    return max(d)


def max_deviation_from_data(path, data):
    """Largest distance from an input outline point to the fitted curve."""
    ring = np.asarray(geometry.flatten_path(path, target_step=0.05))
    worst = 0.0
    for x, y in data:
        dist = np.hypot(ring[:, 0] - x, ring[:, 1] - y)
        worst = max(worst, float(dist.min()))
    return worst


def main() -> None:
    failures = []

    # --- circle ---------------------------------------------------------
    for radius in (40, 120):
        points, r = circle_contour(200, 200, radius)
        for tolerance in (0.3, 0.7, 1.5):
            path = geometry.fit_contour(points, tolerance=tolerance,
                                        corner_threshold=75.0, smoothing=2)
            dev = max_deviation_from_circle(path, 200, 200, r)
            to_data = max_deviation_from_data(path, points)
            print(f"circle r={radius:3d} tol={tolerance:.1f} -> {len(path):3d} beziers, "
                  f"radial deviation {dev:.2f} px, point fit {to_data:.2f} px")
            if len(path) < 4:
                failures.append(f"circle r={radius}: only {len(path)} beziers")
            # the outline itself sits on pixel centres, so ~0.5 px of the radial
            # deviation is unavoidable before the fitter even starts
            if dev > 1.0 + tolerance + 0.5:
                failures.append(f"circle r={radius} tol={tolerance}: deviation {dev:.2f}")
            if to_data > tolerance + 1.0:
                failures.append(f"circle r={radius} tol={tolerance}: misses data by {to_data:.2f}")

    # --- square: must keep four crisp corners ---------------------------
    points = square_contour(50, 60, 240, 210)
    path = geometry.fit_contour(points, tolerance=0.5, corner_threshold=75.0, smoothing=2)
    corners = [(50, 60), (240, 60), (240, 210), (50, 210)]
    for corner in corners:
        nearest = min(math.hypot(p[0] - corner[0], p[1] - corner[1])
                      for bez in path for p in bez)
        print(f"square corner {corner} off by {nearest:.3f} px")
        if nearest > 1.0:
            failures.append(f"square corner {corner} rounded away by {nearest:.2f}px")
    straight = [bez for bez in path]
    if len(straight) > 8:
        failures.append(f"square needs {len(straight)} beziers, expected ~4")

    # --- a tight tolerance must never be silently ignored ----------------
    points, r = circle_contour(200, 200, 100)
    tight = geometry.fit_contour(points, tolerance=0.2, corner_threshold=75.0, smoothing=2)
    loose = geometry.fit_contour(points, tolerance=2.0, corner_threshold=75.0, smoothing=2)
    print(f"tolerance monotonic: 0.2 -> {len(tight)} beziers, 2.0 -> {len(loose)} beziers")
    if not len(tight) >= len(loose):
        failures.append("tighter tolerance did not produce at least as many segments")

    # --- continuity: consecutive segments must share endpoints ----------
    for name, pts, tol in (("circle", circle_contour(200, 200, 90)[0], 0.5),
                           ("square", points, 0.5)):
        path = geometry.fit_contour(pts, tolerance=tol, corner_threshold=75.0, smoothing=2)
        for a, b in zip(path, path[1:]):
            if math.hypot(a[3][0] - b[0][0], a[3][1] - b[0][1]) > 1e-6:
                failures.append(f"{name}: broken chain between segments")
                break
        if math.hypot(path[-1][3][0] - path[0][0][0],
                      path[-1][3][1] - path[0][0][1]) > 1e-6:
            failures.append(f"{name}: path is not closed")
        for bez in path:
            for x, y in bez:
                if not (math.isfinite(x) and math.isfinite(y)):
                    failures.append(f"{name}: non-finite control point")
                    break

    if failures:
        print("\nFAILURES:")
        for item in failures:
            print("  -", item)
        raise SystemExit(1)
    print("\nOK - geometry behaves")


if __name__ == "__main__":
    main()
