"""View layout + scale selection (sheet-mm, y-up).

Third-angle L-arrangement (verified against GT by raster-IoU):
  - front  (main, bottom-left)  : view_dir +Z, up +Y   -> screen (X, Y)
  - top    (above, shares x)    : view_dir +Y, up -Z   -> screen (X, Z)
  - right  (right, shares y)    : view_dir +X, up +Y   -> screen (-Z, Y)

The three views are placed with fixed gaps, the top sharing the front's
x-centre and the right sharing the front's y-centre, then the whole cluster is
centred on the sheet.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.data.render.config import (
    ENVELOPE_H_MM,
    ENVELOPE_W_MM,
    SCALE_LADDER,
    SHEET_CENTER_MM,
    VIEW_GAP_H_MM,
    VIEW_GAP_V_MM,
)
from src.data.render.hlr import (
    Arc,
    Circle,
    Ellipse,
    Polyline,
    ProjectedEdges,
    Segment,
    ViewProjection,
)


# --------------------------------------------------------------------------
# Affine (uniform scale + translate; no rotation / reflection)
# --------------------------------------------------------------------------


def _t(p, s, tx, ty):
    return (p[0] * s + tx, p[1] * s + ty)


def transform_edges(
    e: ProjectedEdges, s: float, tx: float, ty: float
) -> ProjectedEdges:
    out = ProjectedEdges()
    out.segments = [
        Segment(_t(g.p0, s, tx, ty), _t(g.p1, s, tx, ty)) for g in e.segments
    ]
    out.arcs = [
        Arc(_t(g.center, s, tx, ty), g.radius * s, g.a0, g.a1, g.ccw) for g in e.arcs
    ]
    out.circles = [Circle(_t(g.center, s, tx, ty), g.radius * s) for g in e.circles]
    out.ellipses = [
        Ellipse(_t(g.center, s, tx, ty), g.rmaj * s, g.rmin * s, g.rot, g.a0, g.a1)
        for g in e.ellipses
    ]
    out.polylines = [Polyline([_t(p, s, tx, ty) for p in g.pts]) for g in e.polylines]
    return out


# --------------------------------------------------------------------------
# Placed views
# --------------------------------------------------------------------------


@dataclass
class PlacedView:
    name: str
    visible: ProjectedEdges
    hidden: ProjectedEdges
    bbox: tuple[float, float, float, float]  # sheet-mm


@dataclass
class Layout:
    scale: float
    views: list[PlacedView]
    cluster_bbox: tuple[float, float, float, float]


def select_scale(front_w: float, front_h: float, depth: float) -> float:
    """Largest ladder scale whose 3-view cluster fits the usable envelope.

    Cluster width  = scale*front_w + GAP_H + scale*depth
    Cluster height = scale*front_h + GAP_V + scale*depth
    """
    for s in SCALE_LADDER:
        cw = s * (front_w + depth) + VIEW_GAP_H_MM
        ch = s * (front_h + depth) + VIEW_GAP_V_MM
        if cw <= ENVELOPE_W_MM and ch <= ENVELOPE_H_MM:
            return s
    return SCALE_LADDER[-1]


def _local_bbox(vp: ViewProjection) -> tuple[float, float, float, float]:
    return vp.bbox(include_hidden=True)


def build_layout(
    front: ViewProjection, top: ViewProjection, right: ViewProjection
) -> Layout:
    fb = _local_bbox(front)
    tb = _local_bbox(top)
    rb = _local_bbox(right)

    front_w = fb[2] - fb[0]
    front_h = fb[3] - fb[1]
    top_h = tb[3] - tb[1]
    right_w = rb[2] - rb[0]
    # depth used for scale selection: prefer the top/right extents (object depth)
    depth = max(top_h, right_w, 1e-6)

    scale = select_scale(max(front_w, 1e-6), max(front_h, 1e-6), depth)

    # scaled view sizes
    Wf, Hf = front_w * scale, front_h * scale
    Ht = top_h * scale
    Wr = right_w * scale

    # front centre at origin; place others relative, then centre the cluster.
    fcx = 0.0
    fcy = 0.0
    tcx = fcx
    tcy = fcy + Hf / 2 + VIEW_GAP_V_MM + Ht / 2
    rcx = fcx + Wf / 2 + VIEW_GAP_H_MM + Wr / 2
    rcy = fcy

    # per-view translation maps the view's own scaled-bbox centre to (cx,cy)
    def place(vp, lb, cx, cy, name):
        lcx = (lb[0] + lb[2]) / 2
        lcy = (lb[1] + lb[3]) / 2
        tx = cx - lcx * scale
        ty = cy - lcy * scale
        vis = transform_edges(vp.visible, scale, tx, ty)
        hid = transform_edges(vp.hidden, scale, tx, ty)
        bb = (
            lb[0] * scale + tx,
            lb[1] * scale + ty,
            lb[2] * scale + tx,
            lb[3] * scale + ty,
        )
        return PlacedView(name, vis, hid, bb)

    pv_front = place(front, fb, fcx, fcy, "front")
    pv_top = place(top, tb, tcx, tcy, "top")
    pv_right = place(right, rb, rcx, rcy, "right")
    views = [pv_front, pv_top, pv_right]

    # cluster bbox and centring shift
    cx0 = min(v.bbox[0] for v in views)
    cy0 = min(v.bbox[1] for v in views)
    cx1 = max(v.bbox[2] for v in views)
    cy1 = max(v.bbox[3] for v in views)
    shift_x = SHEET_CENTER_MM[0] - (cx0 + cx1) / 2
    shift_y = SHEET_CENTER_MM[1] - (cy0 + cy1) / 2

    placed = []
    for v in views:
        vis = transform_edges(v.visible, 1.0, shift_x, shift_y)
        hid = transform_edges(v.hidden, 1.0, shift_x, shift_y)
        bb = (
            v.bbox[0] + shift_x,
            v.bbox[1] + shift_y,
            v.bbox[2] + shift_x,
            v.bbox[3] + shift_y,
        )
        placed.append(PlacedView(v.name, vis, hid, bb))

    cluster = (cx0 + shift_x, cy0 + shift_y, cx1 + shift_x, cy1 + shift_y)
    return Layout(scale=scale, views=placed, cluster_bbox=cluster)
