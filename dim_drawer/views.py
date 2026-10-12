"""Split a third-angle drawing into its front, top and right projections.

One vertical whitespace band separates {front, top} from right, and one
horizontal band separates front from top. Band width cannot identify them:
views may sit 3mm apart while a multi-body part leaves 6mm gaps inside a view.
So every band pair is tried, and the views must share their aligned extents.
"""

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


def _gaps(items, axis):
    """Cut positions inside the empty bands along one axis, widest band first."""
    lo, hi = (0, 2) if axis == 0 else (1, 3)
    spans = sorted((item[2][lo], item[2][hi]) for item in items)
    bands, reach = [], spans[0][1]
    for start, end in spans[1:]:
        if start > reach:
            bands.append((start - reach, (reach + start) / 2.0))
        reach = max(reach, end)
    return [cut for _, cut in sorted(bands, reverse=True)]


def _quadrants(items, x, y):
    """Front, top and right groups, or None unless exactly those are used."""
    groups = {"front": [], "top": [], "right": []}
    for item in items:
        left, low = item[2][2] < x, item[2][3] < y
        if left:
            groups["front" if low else "top"].append(item)
        elif low:
            groups["right"].append(item)
        else:
            return None
    return groups if all(groups.values()) else None


def _bbox(group):
    return (
        min(b[0] for _, _, b in group),
        min(b[1] for _, _, b in group),
        max(b[2] for _, _, b in group),
        max(b[3] for _, _, b in group),
    )


def _misalignment(front, top, right):
    return max(
        abs(top[0] - front[0]),
        abs(top[2] - front[2]),
        abs(right[1] - front[1]),
        abs(right[3] - front[3]),
    )


def _views(groups):
    views = []
    for name, group in groups.items():
        view: dict = {kind: [] for kind in KINDS}
        for kind, record, _ in group:
            view[kind].append(record)
        bbox = _bbox(group)
        view["name"] = name
        view["bbox"] = bbox
        view["area"] = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        views.append(view)
    return sorted(views, key=lambda v: -v["area"])


def split_views(data, tolerance_ratio=0.005):
    """Return the front, top and right views, largest first, each with a bbox."""
    items = [
        (kind, record, bbox_of(record, kind))
        for kind in KINDS
        for record in data.get(kind, [])
    ]
    if not items:
        raise ValueError("no geometry")
    sheet = _bbox(items)
    tolerance = math.hypot(sheet[2] - sheet[0], sheet[3] - sheet[1]) * tolerance_ratio

    # Duplicate edges can leave noise-wide bands that also align; wider wins.
    for x in _gaps(items, 0):
        for y in _gaps(items, 1):
            groups = _quadrants(items, x, y)
            if not groups:
                continue
            boxes = {name: _bbox(group) for name, group in groups.items()}
            if _misalignment(**boxes) <= tolerance:
                return _views(groups)
    raise ValueError("no third-angle front/top/right split")
