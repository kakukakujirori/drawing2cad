"""Export one projected view as a DXF of its own, and a picture of it.

The view is written at 1:1 in model millimetres, read from its own bottom-left
corner, so it lines up with the sheet coordinates the drawing contract uses.
The picture is the same linework rasterised, so a reader can look at a view
without rasterising it first.
"""

from __future__ import annotations

import math
from pathlib import Path

import ezdxf
from ezdxf.addons.drawing.matplotlib import qsave
from ezdxf.layouts import Modelspace

from zeroshot.pipeline.verification.render._hlr import ProjectedEdges, ViewProjection

_TEMPLATE = Path(__file__).with_name("techdraw_template.dxf")

# Enough to read a hole from, and small enough that several views in a message
# stay affordable.
_PNG_DPI = 100


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


def export_view(dxf_path: Path, projection: ViewProjection, layer: str) -> None:
    """Serialise one view to an existing output directory."""
    doc = ezdxf.readfile(_TEMPLATE)
    modelspace = doc.modelspace()
    if layer not in doc.layers:
        doc.layers.add(layer)
    _add_edges(modelspace, projection.visible, "Continuous", layer)
    _add_edges(modelspace, projection.hidden, "HIDDEN", layer)

    x_min, y_min, x_max, y_max = projection.bbox(include_hidden=True)
    doc.header["$EXTMIN"] = (x_min, y_min, 0.0)
    doc.header["$EXTMAX"] = (x_max, y_max, 0.0)
    doc.saveas(Path(dxf_path))


def export_to_png(dxf_path: Path) -> Path:
    """Rasterise a written view beside itself, black on white."""
    image_path = Path(dxf_path).with_suffix(".png")
    qsave(
        ezdxf.readfile(dxf_path).modelspace(),
        image_path,
        bg="#FFFFFF",
        fg="#000000",
        dpi=_PNG_DPI,
        backend="agg",
    )
    return image_path
