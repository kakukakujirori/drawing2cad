"""SVG writer: typed sheet-mm primitives -> GT-identical A4 SVG.

Path coordinates are emitted in points (mm * MM_TO_PT) in a y-up frame; the
constant GT ``SVG_TRANSFORM`` matrix carries them to device space with the
Y-flip.  Stroke styles match the GT byte-for-byte (widths / dasharrays / caps).
"""

from __future__ import annotations

import math

from src.data.render.config import (
    MM_TO_PT,
    SHEET_H_PT,
    SHEET_W_PT,
    SVG_CENTER_DASH,
    SVG_STYLE_HIDDEN,
    SVG_STYLE_VISIBLE,
    SVG_TRANSFORM,
)
from src.data.render.centermarks import mark_svg_arms


def _num(v: float) -> str:
    """Compact number: integers without a decimal point, else full precision."""
    if v == int(v):
        return str(int(v))
    return repr(v)


_M = ",".join(_num(v) for v in SVG_TRANSFORM)
_TRANSFORM_ATTR = f'transform="matrix({_M})"'


def _fmt(v: float) -> str:
    return f"{v:.6f}".rstrip("0").rstrip(".")


def _pt(p):
    return (p[0] * MM_TO_PT, p[1] * MM_TO_PT)


# ---- style strings (match GT exactly) -------------------------------------


def _style_visible():
    return (
        "fill:none;stroke-width:%s;stroke-linecap:round;stroke-linejoin:round;"
        "stroke:rgb(0%%,0%%,0%%);stroke-opacity:1;stroke-miterlimit:10;"
        % _fmt(SVG_STYLE_VISIBLE["stroke-width"])
    )


def _style_hidden():
    da = ",".join(_fmt(x) for x in SVG_STYLE_HIDDEN["dasharray"])
    return (
        "fill:none;stroke-width:%s;stroke-linecap:butt;stroke-linejoin:miter;"
        "stroke:rgb(0%%,0%%,0%%);stroke-opacity:1;stroke-dasharray:%s;"
        "stroke-miterlimit:10;" % (_fmt(SVG_STYLE_HIDDEN["stroke-width"]), da)
    )


def _style_center():
    da = ",".join(_fmt(x) for x in SVG_CENTER_DASH)
    return (
        "fill:none;stroke-width:%s;stroke-linecap:butt;stroke-linejoin:miter;"
        "stroke:rgb(0%%,0%%,0%%);stroke-opacity:1;stroke-dasharray:%s;"
        "stroke-miterlimit:10;" % (_fmt(0.51024), da)
    )


# ---- path 'd' builders (point coords) -------------------------------------


def _d_seg(p0, p1):
    a = _pt(p0)
    b = _pt(p1)
    return f"M {_fmt(a[0])} {_fmt(a[1])} L {_fmt(b[0])} {_fmt(b[1])} "


def _arc_cmd(center, radius, a0, a1, ccw):
    r = radius * MM_TO_PT
    c = _pt(center)
    s = (c[0] + r * math.cos(a0), c[1] + r * math.sin(a0))
    e = (c[0] + r * math.cos(a1), c[1] + r * math.sin(a1))
    if ccw:
        span = (a1 - a0) % (2 * math.pi)
        sweep = 1
    else:
        span = (a0 - a1) % (2 * math.pi)
        sweep = 0
    large = 1 if span > math.pi else 0
    return (
        f"M {_fmt(s[0])} {_fmt(s[1])} "
        f"A {_fmt(r)} {_fmt(r)} 0 {large} {sweep} {_fmt(e[0])} {_fmt(e[1])} "
    )


def _d_circle(center, radius):
    # two half arcs
    r = radius * MM_TO_PT
    c = _pt(center)
    left = (c[0] - r, c[1])
    right = (c[0] + r, c[1])
    return (
        f"M {_fmt(right[0])} {_fmt(right[1])} "
        f"A {_fmt(r)} {_fmt(r)} 0 0 1 {_fmt(left[0])} {_fmt(left[1])} "
        f"A {_fmt(r)} {_fmt(r)} 0 0 1 {_fmt(right[0])} {_fmt(right[1])} "
    )


def _d_ellipse(e):
    rmaj = e.rmaj * MM_TO_PT
    rmin = e.rmin * MM_TO_PT
    c = _pt(e.center)
    rot = e.rot
    xrot_deg = math.degrees(rot)

    def pt_at(t):
        x = e.rmaj * math.cos(t) * MM_TO_PT
        y = e.rmin * math.sin(t) * MM_TO_PT
        xr = x * math.cos(rot) - y * math.sin(rot)
        yr = x * math.sin(rot) + y * math.cos(rot)
        return (c[0] + xr, c[1] + yr)

    full = abs((e.a1 - e.a0) - 2 * math.pi) < 1e-6
    if full:
        s = pt_at(0.0)
        m = pt_at(math.pi)
        return (
            f"M {_fmt(s[0])} {_fmt(s[1])} "
            f"A {_fmt(rmaj)} {_fmt(rmin)} {_fmt(xrot_deg)} 0 1 {_fmt(m[0])} {_fmt(m[1])} "
            f"A {_fmt(rmaj)} {_fmt(rmin)} {_fmt(xrot_deg)} 0 1 {_fmt(s[0])} {_fmt(s[1])} "
        )
    s = pt_at(e.a0)
    en = pt_at(e.a1)
    span = (e.a1 - e.a0) % (2 * math.pi)
    large = 1 if span > math.pi else 0
    return (
        f"M {_fmt(s[0])} {_fmt(s[1])} "
        f"A {_fmt(rmaj)} {_fmt(rmin)} {_fmt(xrot_deg)} {large} 1 {_fmt(en[0])} {_fmt(en[1])} "
    )


def _d_polyline(pts):
    if len(pts) < 2:
        return ""
    a = _pt(pts[0])
    d = f"M {_fmt(a[0])} {_fmt(a[1])} "
    for p in pts[1:]:
        q = _pt(p)
        d += f"L {_fmt(q[0])} {_fmt(q[1])} "
    return d


def _edges_to_paths(edges):
    d = ""
    for s in edges.segments:
        d += _d_seg(s.p0, s.p1)
    for a in edges.arcs:
        d += _arc_cmd(a.center, a.radius, a.a0, a.a1, a.ccw)
    for c in edges.circles:
        d += _d_circle(c.center, c.radius)
    for e in edges.ellipses:
        d += _d_ellipse(e)
    for pl in edges.polylines:
        d += _d_polyline(pl.pts)
    return d


def _path(style, d):
    return f'<path style="{style}" d="{d}" {_TRANSFORM_ATTR}/>'


def write_svg(path, views, marks):
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{_fmt(SHEET_W_PT)}pt" height="{_fmt(SHEET_H_PT)}pt" '
        f'viewBox="0 0 {_fmt(SHEET_W_PT)} {_fmt(SHEET_H_PT)}" version="1.2">',
        '<g id="surface1">',
    ]
    # hidden first (drawn under), then centre lines, then visible (like GT order
    # is mixed, but z-order among black strokes is irrelevant to raster output)
    hid = _style_hidden()
    vis = _style_visible()
    cen = _style_center()
    for v in views:
        hd = _edges_to_paths(v.hidden)
        if hd.strip():
            parts.append(_path(hid, hd))
    for m in marks:
        for p0, p1 in mark_svg_arms(m):
            parts.append(_path(cen, _d_seg(p0, p1)))
    for v in views:
        vd = _edges_to_paths(v.visible)
        if vd.strip():
            parts.append(_path(vis, vd))
    parts.append("</g>")
    parts.append("</svg>")
    text = "\n".join(parts) + "\n"
    path.write_text(text)
    return text
