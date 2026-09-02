"""Decide which dimensions a projection gets and where they go."""

import math


def _solid_segments(view):
    return [
        (r["start_x"], r["start_y"], r["end_x"], r["end_y"])
        for r in view["line"]
        if not r.get("curved")
    ]


def _significant_coords(segments, axis, tol, min_sep, limit):
    """Bucket axis-aligned edges by coordinate, longest total edge first."""
    totals = {}
    for x1, y1, x2, y2 in segments:
        if axis == "x" and abs(x1 - x2) <= tol:
            key = round(x1 / tol) * tol
            totals[key] = totals.get(key, 0.0) + abs(y2 - y1)
        elif axis == "y" and abs(y1 - y2) <= tol:
            key = round(y1 / tol) * tol
            totals[key] = totals.get(key, 0.0) + abs(x2 - x1)

    picked = []
    for coord, _ in sorted(totals.items(), key=lambda kv: -kv[1]):
        if all(abs(coord - c) >= min_sep for c in picked):
            picked.append(coord)
        if len(picked) >= limit:
            break
    return picked


def _clearance(bbox, others, sheet):
    """Free space on each side, so dimensions do not run into another view."""
    x0, y0, x1, y1 = bbox
    free = {
        "left": x0 - sheet[0],
        "right": sheet[2] - x1,
        "below": y0 - sheet[1],
        "above": sheet[3] - y1,
    }
    for ox0, oy0, ox1, oy1 in others:
        if oy1 > y0 and oy0 < y1:
            if ox1 <= x0:
                free["left"] = min(free["left"], x0 - ox1)
            if ox0 >= x1:
                free["right"] = min(free["right"], ox0 - x1)
        if ox1 > x0 and ox0 < x1:
            if oy1 <= y0:
                free["below"] = min(free["below"], y0 - oy1)
            if oy0 >= y1:
                free["above"] = min(free["above"], oy0 - y1)
    return free


def _outward_sides(bbox, others, free, need):
    """Prefer the side facing away from the sheet centroid; choosing purely by
    free space makes two neighbouring views stack into the same gap."""
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    boxes = list(others) + [bbox]
    gx = sum((b[0] + b[2]) / 2 for b in boxes) / len(boxes)
    gy = sum((b[1] + b[3]) / 2 for b in boxes) / len(boxes)

    h_side = "below" if cy <= gy else "above"
    v_side = "left" if cx <= gx else "right"
    h_flip = "above" if h_side == "below" else "below"
    v_flip = "right" if v_side == "left" else "left"
    if free[h_side] < need <= free[h_flip]:
        h_side = h_flip
    if free[v_side] < need <= free[v_flip]:
        v_side = v_flip
    return h_side, v_side


def _dedupe(entities, rel=0.03):
    """Radii that differ only by rounding get one dimension, not two."""
    kept = []
    for entity in sorted(entities, key=lambda e: -e["radius"]):
        if any(
            abs(entity["radius"] - k["radius"]) <= rel * max(k["radius"], 1e-6)
            for k in kept
        ):
            continue
        kept.append(entity)
    return kept


def dimension_view(
    writer, view, others, sheet, occupancy, max_linear=3, max_holes=3, max_radii=2
):
    """Dimension one projection. Returns how many dimensions were placed."""
    x0, y0, x1, y1 = view["bbox"]
    txt = writer.txt
    first, step = 3.2 * txt, 2.4 * txt

    free = _clearance(view["bbox"], others, sheet)
    h_side, v_side = _outward_sides(
        view["bbox"], others, free, first + max_linear * step
    )

    segments = _solid_segments(view)
    size = max(x1 - x0, y1 - y0)
    tol, min_sep = size * 0.004, size * 0.12

    # Extension lines start from the edge nearest the dimension line; anchoring
    # to the far edge draws them straight across the view.
    y_anchor = y0 if h_side == "below" else y1
    x_anchor = x0 if v_side == "left" else x1
    h_sign = -1 if h_side == "below" else 1
    v_sign = -1 if v_side == "left" else 1

    inner_x = [
        c
        for c in _significant_coords(segments, "x", tol, min_sep, max_linear + 2)
        if x0 + min_sep < c < x1 - min_sep
    ][: max_linear - 1]
    inner_y = [
        c
        for c in _significant_coords(segments, "y", tol, min_sep, max_linear + 2)
        if y0 + min_sep < c < y1 - min_sep
    ][: max_linear - 1]

    def place_linear(p1, p2, angle, start_row):
        """Step the dimension line outward until a row is free."""
        for row in range(start_row, start_row + 6):
            offset = first + row * step
            base = (
                (x1, y_anchor + h_sign * offset)
                if angle == 0
                else (x_anchor + v_sign * offset, y1)
            )
            if writer.commit(writer.linear(base, p1, p2, angle), occupancy):
                return True
        return False

    placed = 0
    # Inner dimensions sit closest to the view, the overall one outermost.
    for row, c in enumerate(sorted(inner_x)):
        placed += place_linear((x0, y_anchor), (c, y_anchor), 0, row)
    placed += place_linear((x0, y_anchor), (x1, y_anchor), 0, len(inner_x))

    for row, c in enumerate(sorted(inner_y)):
        placed += place_linear((x_anchor, y0), (x_anchor, c), 90, row)
    placed += place_linear((x_anchor, y0), (x_anchor, y1), 90, len(inner_y))

    away = math.degrees(
        math.atan2(1 if h_side == "below" else -1, 1 if v_side == "left" else -1)
    )
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

    def place_leader(build, center):
        """Try angle and reach combinations until the label lands clear."""
        outward = math.degrees(math.atan2(center[1] - cy, center[0] - cx))
        for reach in (3.0, 4.2, 5.5, 7.0, 9.0):
            for angle in (
                away,
                outward,
                away + 45,
                away - 45,
                outward + 45,
                outward - 45,
                away + 90,
                away - 90,
                away + 135,
                away - 135,
            ):
                if writer.commit(build(angle % 360, reach), occupancy):
                    return True
        return False

    for circle in _dedupe(view["circle"])[:max_holes]:
        center = (circle["center_x"], circle["center_y"])
        placed += place_leader(
            lambda angle, reach, c=circle, p=center: writer.diameter(
                p, c["radius"], angle, reach
            ),
            center,
        )

    for arc in _dedupe(view["arc"])[:max_radii]:
        if arc["radius"] <= tol * 5:
            continue
        sweep = (arc["end_angle"] - arc["start_angle"]) % 360
        mid = (arc["start_angle"] + sweep / 2.0) % 360
        center = (arc["center_x"], arc["center_y"])
        # The leader must point where the arc actually is, so only reach varies.
        for reach in (2.5, 3.6, 4.8, 6.2, 7.8):
            if writer.commit(
                writer.radius(center, arc["radius"], mid, reach), occupancy
            ):
                placed += 1
                break

    return placed
