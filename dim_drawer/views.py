"""Split a multi-view drawing into its individual projections.

Orthographic views are always separated by a band of whitespace that spans the
full width (or height) of the sheet, so a recursive XY-cut finds them. Distance
clustering does not: two views 9.8mm apart merge under any threshold loose
enough to hold one view together, and the merged bounding box then yields an
overall dimension that spans both views.
"""

import bisect
import math

KINDS = ("line", "dash_line", "circle", "arc")


def _arc_bbox(cx, cy, r, start_angle, end_angle):
    """Extent of the arc itself; the full circle box would swallow the sheet."""
    start = start_angle % 360.0
    sweep = (end_angle - start_angle) % 360.0 or 360.0
    points = [
        (cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a)))
        for a in (start, start + sweep)
    ]
    points += [
        (cx + r * math.cos(math.radians(q)), cy + r * math.sin(math.radians(q)))
        for q in (0, 90, 180, 270)
        if (q - start) % 360.0 <= sweep
    ]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def bbox_of(record, kind):
    if kind in ("line", "dash_line"):
        return (
            min(record["start_x"], record["end_x"]),
            min(record["start_y"], record["end_y"]),
            max(record["start_x"], record["end_x"]),
            max(record["start_y"], record["end_y"]),
        )
    cx, cy, r = record["center_x"], record["center_y"], record["radius"]
    if kind == "arc":
        return _arc_bbox(cx, cy, r, record["start_angle"], record["end_angle"])
    return (cx - r, cy - r, cx + r, cy + r)


def _split_axis(items, axis, min_gap):
    """Cut at every empty band wider than min_gap along one axis."""
    lo, hi = (0, 2) if axis == 0 else (1, 3)
    spans = sorted((item[2][lo], item[2][hi]) for item in items)
    cuts, reach = [], spans[0][1]
    for start, end in spans[1:]:
        if start - reach > min_gap:
            cuts.append((reach + start) / 2.0)
        reach = max(reach, end)
    if not cuts:
        return None

    groups = [[] for _ in range(len(cuts) + 1)]
    for item in items:
        middle = (item[2][lo] + item[2][hi]) / 2.0
        groups[bisect.bisect_left(cuts, middle)].append(item)
    return [g for g in groups if g]


def _xycut(items, min_gap):
    for axis in (0, 1):
        parts = _split_axis(items, axis, min_gap)
        if parts and len(parts) > 1:
            return [v for part in parts for v in _xycut(part, min_gap)]
    return [items]


def split_views(data, min_gap_ratio=0.022):
    """Return one dict per projection, largest first, each carrying a bbox."""
    items = [
        (kind, record, bbox_of(record, kind))
        for kind in KINDS
        for record in data.get(kind, [])
    ]
    if not items:
        return []

    xs = [b[0] for _, _, b in items] + [b[2] for _, _, b in items]
    ys = [b[1] for _, _, b in items] + [b[3] for _, _, b in items]
    min_gap = math.hypot(max(xs) - min(xs), max(ys) - min(ys)) * min_gap_ratio

    views = []
    for group in _xycut(items, min_gap):
        view = {kind: [] for kind in KINDS}
        for kind, record, _ in group:
            view[kind].append(record)
        boxes = [b for _, _, b in group]
        bbox = (
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        )
        view["bbox"] = bbox
        view["area"] = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        views.append(view)

    views.sort(key=lambda v: -v["area"])
    return views
