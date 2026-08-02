"""Export an arranged three-view drawing as a GT-compatible DXF.

This module combines the two DXF-output responsibilities of the reference
renderer: serialising placed primitives and adding centre-mark annotations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import ezdxf
from ezdxf.layouts import Modelspace

from zeroshot.pipeline.verification.render._hlr import Circle, ProjectedEdges
from zeroshot.pipeline.verification.render.arrange import Layout, PlacedView
from zeroshot.pipeline.verification.render.constants import SHEET_H_MM, SHEET_W_MM

# Measured from SolidWorks SW_CENTERMARKSYMBOL blocks.  These are fixed sheet-mm
# annotation dimensions, not model-space dimensions and not runtime knobs.
CENTERMARK_CROSS_MM = 2.5
CENTERMARK_GAP_MM = 2.5
CENTERMARK_EXT_MM = 2.5
CENTERMARK_DEDUPE_TOL_MM = 0.15

_TEMPLATE = Path(__file__).with_name("techdraw_template.dxf")


@dataclass(frozen=True)
class _CenterMark:
    cx: float
    cy: float
    arm_half_length: float


def _dedupe_circles(circles: list[Circle]) -> list[tuple[float, float, float]]:
    """Collapse concentric circles, retaining the largest radius in each group."""
    groups: list[list[float]] = []
    for circle in circles:
        cx, cy = circle.center
        for group in groups:
            if (
                abs(group[0] - cx) <= CENTERMARK_DEDUPE_TOL_MM
                and abs(group[1] - cy) <= CENTERMARK_DEDUPE_TOL_MM
            ):
                group[2] = max(group[2], circle.radius)
                break
        else:
            groups.append([cx, cy, circle.radius])
    return [(cx, cy, radius) for cx, cy, radius in groups]


def _marks_for_view(view: PlacedView) -> list[_CenterMark]:
    """Create one mark per visible circular feature in ``view``.

    Concentric counterbore/countersink circles share a mark.
    Hidden circles do not receive marks, matching the reference renderer.
    """
    return [
        _CenterMark(cx, cy, radius + CENTERMARK_EXT_MM)
        for cx, cy, radius in _dedupe_circles(view.visible.circles)
    ]


def _mark_block_lines(
    mark: _CenterMark,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Return the eight SolidWorks mark segments in block-local coordinates."""
    cross = CENTERMARK_CROSS_MM
    gap_end = CENTERMARK_CROSS_MM + CENTERMARK_GAP_MM
    length = mark.arm_half_length
    if length <= cross:
        raise ValueError(
            f"center-mark arm must extend beyond its {cross} mm cross: {length}"
        )

    # SolidWorks keeps eight LINE entities even when the arm is too short to
    # contain the usual gap.  Long arms leave [cross, gap_end] blank; short arms
    # join the extension directly to the cross tip.
    outer_start = gap_end if length > gap_end else cross
    lines = [
        ((0.0, 0.0), (0.0, cross)),
        ((0.0, 0.0), (-cross, 0.0)),
        ((0.0, 0.0), (0.0, -cross)),
        ((0.0, 0.0), (cross, 0.0)),
        ((0.0, outer_start), (0.0, length)),
        ((-outer_start, 0.0), (-length, 0.0)),
        ((0.0, -outer_start), (0.0, -length)),
        ((outer_start, 0.0), (length, 0.0)),
    ]
    return lines


def _add_edges(
    modelspace: Modelspace, edges: ProjectedEdges, linetype: str, layer: str
) -> None:
    attribs = {"layer": layer, "linetype": linetype}
    for segment in edges.segments:
        modelspace.add_line(segment.p0, segment.p1, dxfattribs=attribs)
    for arc in edges.arcs:
        if arc.ccw:
            start_angle, end_angle = math.degrees(arc.a0), math.degrees(arc.a1)
        else:
            start_angle, end_angle = math.degrees(arc.a1), math.degrees(arc.a0)
        # DXF ARC traversal is always counter-clockwise.  Some readers interpret
        # negative or multi-turn angles as full circles, so normalise both.
        modelspace.add_arc(
            arc.center,
            arc.radius,
            start_angle % 360.0,
            end_angle % 360.0,
            dxfattribs=attribs,
        )
    for circle in edges.circles:
        modelspace.add_circle(circle.center, circle.radius, dxfattribs=attribs)
    for ellipse in edges.ellipses:
        major_axis = (
            ellipse.rmaj * math.cos(ellipse.rot),
            ellipse.rmaj * math.sin(ellipse.rot),
            0.0,
        )
        ratio = ellipse.rmin / ellipse.rmaj if ellipse.rmaj else 1.0
        modelspace.add_ellipse(
            (ellipse.center[0], ellipse.center[1], 0.0),
            major_axis,
            ratio,
            ellipse.a0,
            ellipse.a1,
            dxfattribs=attribs,
        )
    for polyline in edges.polylines:
        if len(polyline.pts) >= 4:
            # GT represents curved silhouettes as SPLINE.  Degenerate fit-point
            # sets can still be rejected by ezdxf, in which case retain the
            # geometry as an LWPOLYLINE instead of dropping it.
            try:
                modelspace.add_spline(fit_points=polyline.pts, dxfattribs=attribs)
            except Exception:  # noqa: BLE001
                modelspace.add_lwpolyline(polyline.pts, dxfattribs=attribs)
        elif len(polyline.pts) >= 2:
            modelspace.add_lwpolyline(polyline.pts, dxfattribs=attribs)


def export_dxf(dxf_path: Path, layout: Layout) -> None:
    """Serialise one arranged drawing to an existing output directory."""
    doc = ezdxf.readfile(_TEMPLATE)
    modelspace = doc.modelspace()

    views = (
        ("front", layout.front),
        ("top", layout.top),
        ("right", layout.right),
    )
    for layer, view in views:
        if layer not in doc.layers:
            doc.layers.add(layer)
        _add_edges(modelspace, view.visible, "Continuous", layer)
        _add_edges(modelspace, view.hidden, "HIDDEN", layer)

    marks = [mark for _, view in views for mark in _marks_for_view(view)]
    for index, mark in enumerate(marks):
        block_name = f"SW_CENTERMARKSYMBOL_{index}"
        if block_name in doc.blocks:
            doc.blocks.delete_block(block_name, safe=False)
        block = doc.blocks.new(name=block_name)
        for start, end in _mark_block_lines(mark):
            block.add_line(
                start,
                end,
                dxfattribs={"layer": "10", "linetype": "BYLAYER"},
            )
        modelspace.add_blockref(
            block_name,
            (mark.cx, mark.cy),
            dxfattribs={"layer": "10"},
        )

    # Keep the sheet extents header consistent with the A4 template.
    doc.header["$EXTMIN"] = (0.0, 0.0, 0.0)
    doc.header["$EXTMAX"] = (SHEET_W_MM, SHEET_H_MM, 0.0)
    doc.saveas(Path(dxf_path))
