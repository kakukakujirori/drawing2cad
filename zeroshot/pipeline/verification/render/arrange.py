"""Place the three projections on the A4 sheet at the canonical scale.

Third-angle L-arrangement, verified against GT by raster-IoU::

    front  (main, bottom-left) : screen ( X,  Y)
    top    (above front)       : screen ( X, -Z)  -> shares front's x-centre
    right  (right of front)    : screen (-Z,  Y)  -> shares front's y-centre

(see project.py for the frames these come from)

One drawing scale is chosen from the standard ladder so the whole cluster fits
the usable envelope, the three views are placed with fixed gaps, and the cluster
is centred on the sheet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from zeroshot.pipeline.verification.render._hlr import (
    Arc,
    Circle,
    Ellipse,
    Polyline,
    ProjectedEdges,
    Segment,
    ViewProjection,
)
from zeroshot.pipeline.verification.render.constants import (
    ENVELOPE_H_MM,
    ENVELOPE_W_MM,
    SHEET_CENTER_MM,
    SHEET_H_MM,
    SHEET_MARGIN_MM,
    SHEET_W_MM,
)
from zeroshot.pipeline.verification.render.project import ViewProjections

# Inter-view gaps.  GT gaps are scale-coupled (median v~44, h~72 mm) and the
# absolute scale is unrecoverable (GT STEPs are normalised); these fixed gaps
# only affect global composition, never per-view geometry.
VIEW_GAP_V_MM = 25.0
VIEW_GAP_H_MM = 35.0
# Standard drawing scales SolidWorks chooses from (drawing-mm per model-unit).
SCALE_LADDER = (
    100.0,
    50.0,
    20.0,
    10.0,
    5.0,
    2.0,
    1.0,
    0.5,
    0.2,
    0.1,
    0.05,
    0.02,
    0.01,
    0.005,
    0.002,
    0.001,
)

# A placed view narrower than this (sheet-mm, either axis) carries no
# recoverable shape. It only ever fires on pathological inputs: an empty HLR
# projection, or a knife-edge view of a near-zero-thickness plate that collapses
# to a single line.
MIN_VIEW_EXTENT_MM = 0.05


class DegenerateDrawingError(RuntimeError):
    """A view could not be placed into a valid positive-area, on-sheet bbox."""


@dataclass(frozen=True)
class PlacedView:
    """One view's geometry in sheet-mm."""

    visible: ProjectedEdges
    hidden: ProjectedEdges
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class Layout:
    """The placed drawing.  Which view is which is carried by the field, so a
    layout cannot be short a view, carry a duplicate, or mislabel one.  The view
    names themselves belong to the consumers: export_dxf.py turns them into DXF
    layers, and _validate uses them to say which view failed."""

    front: PlacedView
    top: PlacedView
    right: PlacedView


def _transform_edges(
    edges: ProjectedEdges, scale: float, tx: float, ty: float
) -> ProjectedEdges:
    """Uniform scale + translate. No rotation or reflection, so arc angles and
    ellipse orientations carry over unchanged."""

    def point(p):
        return (p[0] * scale + tx, p[1] * scale + ty)

    return ProjectedEdges(
        segments=[Segment(point(e.p0), point(e.p1)) for e in edges.segments],
        arcs=[
            Arc(point(e.center), e.radius * scale, e.a0, e.a1, e.ccw)
            for e in edges.arcs
        ],
        circles=[Circle(point(e.center), e.radius * scale) for e in edges.circles],
        ellipses=[
            Ellipse(point(e.center), e.rmaj * scale, e.rmin * scale, e.rot, e.a0, e.a1)
            for e in edges.ellipses
        ],
        polylines=[Polyline([point(p) for p in e.pts]) for e in edges.polylines],
    )


def select_scale(front_w: float, front_h: float, depth: float) -> float:
    """Largest ladder scale whose 3-view cluster fits the usable envelope.

    Cluster width  = scale*(front_w + depth) + GAP_H
    Cluster height = scale*(front_h + depth) + GAP_V
    """
    for scale in SCALE_LADDER:
        width = scale * (front_w + depth) + VIEW_GAP_H_MM
        height = scale * (front_h + depth) + VIEW_GAP_V_MM
        if width <= ENVELOPE_W_MM and height <= ENVELOPE_H_MM:
            return scale
    return SCALE_LADDER[-1]


def _check(view: PlacedView, name: str) -> None:
    """Reject a degenerate or runaway placement before anything is written.

    ``run_render.py`` runs one part per isolated process, so raising here makes
    the part fail cleanly rather than emitting an invalid drawing.  ``name`` only
    exists to say which view failed.
    """
    x_min, y_min, x_max, y_max = view.bbox
    if not all(math.isfinite(c) for c in view.bbox):
        raise DegenerateDrawingError(f"view {name!r} bbox is not finite: {view.bbox}")
    width = x_max - x_min
    height = y_max - y_min
    if width < MIN_VIEW_EXTENT_MM or height < MIN_VIEW_EXTENT_MM:
        raise DegenerateDrawingError(
            f"view {name!r} has near-zero extent "
            f"(w={width:.4f}, h={height:.4f} mm): {view.bbox}"
        )
    if (
        x_min < -SHEET_MARGIN_MM
        or y_min < -SHEET_MARGIN_MM
        or x_max > SHEET_W_MM + SHEET_MARGIN_MM
        or y_max > SHEET_H_MM + SHEET_MARGIN_MM
    ):
        raise DegenerateDrawingError(
            f"view {name!r} bbox runs off the sheet "
            f"(sheet {SHEET_W_MM}x{SHEET_H_MM} mm): {view.bbox}"
        )


def _offset_onto(
    local_bbox: tuple[float, float, float, float], scale: float, cx: float, cy: float
) -> tuple[float, float]:
    """Translation putting the view's scaled bbox centre at ``(cx, cy)``."""
    x0, y0, x1, y1 = local_bbox
    return cx - (x0 + x1) / 2 * scale, cy - (y0 + y1) / 2 * scale


def _scaled_bbox(
    local_bbox: tuple[float, float, float, float], scale: float, tx: float, ty: float
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = local_bbox
    return x0 * scale + tx, y0 * scale + ty, x1 * scale + tx, y1 * scale + ty


def arrange(projections: ViewProjections) -> Layout:
    """Scale and place the three projections onto the sheet."""
    front_bb = projections.front.bbox(include_hidden=True)
    top_bb = projections.top.bbox(include_hidden=True)
    right_bb = projections.right.bbox(include_hidden=True)

    front_w = front_bb[2] - front_bb[0]
    front_h = front_bb[3] - front_bb[1]
    top_h = top_bb[3] - top_bb[1]
    right_w = right_bb[2] - right_bb[0]
    # Depth drives scale selection alongside the front view: prefer the top /
    # right extents, which are the object's depth.
    depth = max(top_h, right_w, 1e-6)

    scale = select_scale(max(front_w, 1e-6), max(front_h, 1e-6), depth)

    # Front centred at the origin; top shares its x-centre, right its y-centre.
    top_cy = (front_h * scale) / 2 + VIEW_GAP_V_MM + (top_h * scale) / 2
    right_cx = (front_w * scale) / 2 + VIEW_GAP_H_MM + (right_w * scale) / 2
    front_off = _offset_onto(front_bb, scale, 0.0, 0.0)
    top_off = _offset_onto(top_bb, scale, 0.0, top_cy)
    right_off = _offset_onto(right_bb, scale, right_cx, 0.0)

    # Centre the whole cluster on the sheet.  Only the bboxes are needed to work
    # the shift out, so the edges themselves are transformed just once, below.
    boxes = (
        _scaled_bbox(front_bb, scale, *front_off),
        _scaled_bbox(top_bb, scale, *top_off),
        _scaled_bbox(right_bb, scale, *right_off),
    )
    shift_x = (
        SHEET_CENTER_MM[0] - (min(b[0] for b in boxes) + max(b[2] for b in boxes)) / 2
    )
    shift_y = (
        SHEET_CENTER_MM[1] - (min(b[1] for b in boxes) + max(b[3] for b in boxes)) / 2
    )

    def place(
        projection: ViewProjection,
        local_bbox: tuple[float, float, float, float],
        offset: tuple[float, float],
    ) -> PlacedView:
        tx = offset[0] + shift_x
        ty = offset[1] + shift_y
        return PlacedView(
            visible=_transform_edges(projection.visible, scale, tx, ty),
            hidden=_transform_edges(projection.hidden, scale, tx, ty),
            bbox=_scaled_bbox(local_bbox, scale, tx, ty),
        )

    layout = Layout(
        front=place(projections.front, front_bb, front_off),
        top=place(projections.top, top_bb, top_off),
        right=place(projections.right, right_bb, right_off),
    )
    _check(layout.front, "front")
    _check(layout.top, "top")
    _check(layout.right, "right")
    return layout
