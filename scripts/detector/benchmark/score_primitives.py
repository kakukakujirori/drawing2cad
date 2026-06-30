#!/usr/bin/env python
"""Score detected line/circle primitives against the renderer's GT graph.json.

GT = scripts/renderer output (primitives in PNG-pixel space, with visibility).
Detections = {"lines":[[x1,y1,x2,y2],...], "circles":[[cx,cy,r],...]} in the same px space.

Metrics (honest, fragmentation-robust):
  * circles: center+radius match (recall/precision), like poc/score.py.
  * lines: COVERAGE-based (a long edge fragmented into many detections still counts).
    recall = fraction of GT line length within PERP_TOL of some detection;
    precision = fraction of detected length (inside the GT bbox) lying on some GT line.
  Lines are split visible / hidden / all — visible is the headline a real drawing shows.
"""
import json, math
import numpy as np

C_TOL = 22.0     # circle center match (px)
R_TOL = 12.0     # circle radius match (px) or 25% of r
PERP_TOL = 7.0   # line proximity (px)
SAMPLE = 6.0     # sample step along a segment (px)
BBOX_MARGIN = 30.0

def load_gt(graph_path):
    g = json.load(open(graph_path))
    lines, circles = [], []
    for v in g["views"]:
        for p in v["primitives"]:
            vis = p.get("line_role", p.get("visibility", "visible"))
            if p["type"] == "line":
                lines.append((p["p1"], p["p2"], vis))
            elif p["type"] in ("circle", "arc"):
                circles.append((p["center"], p.get("r_px", 0), vis))
    return lines, circles

def _seg_pts(a, b, step=SAMPLE):
    a = np.asarray(a, float); b = np.asarray(b, float)
    L = np.hypot(*(b - a))
    n = max(2, int(L / step) + 1)
    return a + (b - a) * np.linspace(0, 1, n)[:, None], L

def _pt_seg_dist(p, a, b):
    a = np.asarray(a, float); b = np.asarray(b, float); p = np.asarray(p, float)
    ab = b - a; t = np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-9), 0, 1)
    return np.hypot(*(p - (a + t * ab)))

def _covered(pts, segs, tol=PERP_TOL):
    if not segs:
        return np.zeros(len(pts), bool)
    out = np.zeros(len(pts), bool)
    for i, p in enumerate(pts):
        for (a, b) in segs:
            if _pt_seg_dist(p, a, b) < tol:
                out[i] = True; break
    return out

def score_lines(gt_lines, det_lines, bbox):
    det = [((l[0], l[1]), (l[2], l[3])) for l in det_lines]
    res = {}
    for tag in ("visible", "hidden", "all"):
        G = [(a, b) for (a, b, v) in gt_lines if tag == "all" or v == tag]
        tot = cov = 0.0
        for a, b in G:
            pts, L = _seg_pts(a, b)
            c = _covered(pts, det)
            cov += c.mean() * L; tot += L
        res[tag] = round(cov / tot, 3) if tot else None       # length-weighted recall
    # precision: detected segments whose midpoint is in the GT bbox
    (x0, y0, x1, y1) = bbox
    Gall = [(a, b) for (a, b, v) in gt_lines]
    pin = ptot = 0.0
    for a, b in det:
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        if not (x0 <= mx <= x1 and y0 <= my <= y1):
            continue
        pts, L = _seg_pts(a, b)
        c = _covered(pts, Gall)
        pin += c.mean() * L; ptot += L
    res["precision"] = round(pin / ptot, 3) if ptot else None
    return res

def score_circles(gt_circles, det_circles, bbox):
    (x0, y0, x1, y1) = bbox
    det = [c for c in det_circles if x0 <= c[0] <= x1 and y0 <= c[1] <= y1]
    used = set(); tp = 0
    for (cen, r, vis) in gt_circles:
        best = None; bd = 1e9
        for i, (dx, dy, dr) in enumerate(det):
            if i in used:
                continue
            dc = math.hypot(cen[0] - dx, cen[1] - dy)
            if dc < C_TOL and abs(r - dr) < max(R_TOL, 0.25 * r) and dc < bd:
                bd = dc; best = i
        if best is not None:
            used.add(best); tp += 1
    ng = len(gt_circles); nd = len(det)
    return {"recall": round(tp / ng, 3) if ng else None,
            "precision": round(tp / nd, 3) if nd else None,
            "tp": tp, "gt": ng, "det": nd}

def gt_bbox(gt_lines, gt_circles, m=BBOX_MARGIN):
    xs, ys = [], []
    for a, b, _ in gt_lines:
        xs += [a[0], b[0]]; ys += [a[1], b[1]]
    for cen, r, _ in gt_circles:
        xs += [cen[0] - r, cen[0] + r]; ys += [cen[1] - r, cen[1] + r]
    return (min(xs) - m, min(ys) - m, max(xs) + m, max(ys) + m)

def score(graph_path, det):
    gl, gc = load_gt(graph_path)
    bb = gt_bbox(gl, gc)
    return {"lines": score_lines(gl, det.get("lines", []), bb),
            "circles": score_circles(gc, det.get("circles", []), bb)}

if __name__ == "__main__":
    import sys
    graph, detj = sys.argv[1], sys.argv[2]
    print(json.dumps(score(graph, json.load(open(detj))), indent=2))
