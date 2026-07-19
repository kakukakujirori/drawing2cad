#!/usr/bin/env python
"""Classical-CV geometry detection baselines, scored against the renderer GT.

Lines:   OpenCV LSD (cv2.createLineSegmentDetector) + probabilistic Hough.
Circles: cv2.HoughCircles (multi-pass over radius bands, since one pass can't span
         a tiny bolt hole and a big bore).
Run on clean.png and scan.png; report recall/precision per method.
CPU only.  Usage:  python run_opencv.py --dir experiments/detector_bench
"""

import os
import json
import argparse
import cv2
import numpy as np
from score_primitives import score


def lsd_lines(gray):
    try:
        det = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        segs = det.detect(gray)[0]
        if segs is None:
            return []
        return [
            [float(x1), float(y1), float(x2), float(y2)] for [[x1, y1, x2, y2]] in segs
        ]
    except Exception as e:
        print("  (LSD unavailable:", e, ")")
        return []


def hough_lines(gray):
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=50, minLineLength=25, maxLineGap=5
    )
    if lines is None:
        return []
    return [
        [float(x1), float(y1), float(x2), float(y2)] for [[x1, y1, x2, y2]] in lines
    ]


def hough_circles(gray):
    blur = cv2.medianBlur(gray, 3)
    out = []
    for rmin, rmax in [(8, 22), (20, 45), (40, 90), (85, 200)]:  # radius bands
        c = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=18,
            param1=120,
            param2=42,
            minRadius=rmin,
            maxRadius=rmax,
        )
        if c is not None:
            out += [[float(x), float(y), float(r)] for x, y, r in c[0]]
    # dedup near-identical
    keep = []
    for x, y, r in sorted(out, key=lambda t: -t[2]):
        if all(np.hypot(x - kx, y - ky) > 12 or abs(r - kr) > 8 for kx, ky, kr in keep):
            keep.append([x, y, r])
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="experiments/detector_bench")
    a = ap.parse_args()
    graph = os.path.join(a.dir, "part.graph.json")
    rows = []
    for variant in ("clean", "scan"):
        img = cv2.imread(os.path.join(a.dir, f"{variant}.png"), cv2.IMREAD_GRAYSCALE)
        det_lsd = {"lines": lsd_lines(img), "circles": []}
        det_hough = {"lines": hough_lines(img), "circles": hough_circles(img)}
        s_lsd = score(graph, det_lsd)
        s_hough = score(graph, det_hough)
        rows.append(
            (variant, "LSD(lines)", len(det_lsd["lines"]), s_lsd["lines"], None)
        )
        rows.append(
            (
                variant,
                "HoughP(lines)+HoughCircles",
                len(det_hough["lines"]),
                s_hough["lines"],
                s_hough["circles"],
            )
        )
        json.dump(
            {
                "lsd": s_lsd,
                "hough": s_hough,
                "n_lsd_lines": len(det_lsd["lines"]),
                "n_hough_lines": len(det_hough["lines"]),
                "n_hough_circles": len(det_hough["circles"]),
            },
            open(os.path.join(a.dir, f"opencv_{variant}.json"), "w"),
            indent=2,
        )
    print(
        f"\n{'variant':6} {'method':28} {'#det':>5}  line R(vis/hid/all) "
        f"line P   circle R/P"
    )
    for variant, name, ndet, ln, ci in rows:
        lr = f"{ln['visible']}/{ln['hidden']}/{ln['all']}"
        cstr = (
            f"{ci['recall']}/{ci['precision']} ({ci['tp']}/{ci['gt']})" if ci else "-"
        )
        print(
            f"{variant:6} {name:28} {ndet:>5}  {lr:18} {str(ln['precision']):7} {cstr}"
        )


if __name__ == "__main__":
    main()
