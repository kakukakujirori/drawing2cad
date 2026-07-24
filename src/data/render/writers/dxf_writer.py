"""DXF writer: typed sheet-mm primitives -> GT-native R2000 DXF.

Loads the embedded template (assets/techdraw_template.dxf) so the header
tables (linetypes, the SolidWorks Chinese-named layer template, standard
blocks) are byte-identical to GT.  Geometry goes on the per-view layer named
for its orthographic view ("front"/"top"/"right") -- that layer *is* the view
consumed by the DXF parser -- with linetype Continuous (visible) or HIDDEN;
center marks go on layer "10" as SW_CENTERMARKSYMBOL_k INSERT blocks with 8
line segments each.
"""

from __future__ import annotations

import math
from pathlib import Path

import ezdxf

from src.data.render.centermarks import mark_block_lines

_TEMPLATE = Path(__file__).resolve().parent.parent / "assets" / "techdraw_template.dxf"


def _add_edges(msp, edges, linetype, layer):
    attribs = {"layer": layer, "linetype": linetype}
    for s in edges.segments:
        msp.add_line(s.p0, s.p1, dxfattribs=attribs)
    for a in edges.arcs:
        if a.ccw:
            sa, ea = math.degrees(a.a0), math.degrees(a.a1)
        else:
            sa, ea = math.degrees(a.a1), math.degrees(a.a0)
        msp.add_arc(a.center, a.radius, sa, ea, dxfattribs=attribs)
    for c in edges.circles:
        msp.add_circle(c.center, c.radius, dxfattribs=attribs)
    for e in edges.ellipses:
        major = (e.rmaj * math.cos(e.rot), e.rmaj * math.sin(e.rot), 0.0)
        ratio = (e.rmin / e.rmaj) if e.rmaj else 1.0
        msp.add_ellipse(
            (e.center[0], e.center[1], 0.0),
            major,
            ratio,
            e.a0,
            e.a1,
            dxfattribs=attribs,
        )
    for pl in edges.polylines:
        if len(pl.pts) >= 4:
            # GT represents curved silhouettes as SPLINE
            try:
                msp.add_spline(fit_points=pl.pts, dxfattribs=attribs)
            except Exception:  # noqa: BLE001
                msp.add_lwpolyline(pl.pts, dxfattribs=attribs)
        elif len(pl.pts) >= 2:
            msp.add_lwpolyline(pl.pts, dxfattribs=attribs)


def write_dxf(path, views, marks):
    doc = ezdxf.readfile(_TEMPLATE)
    msp = doc.modelspace()

    for v in views:
        # The view's name is its DXF layer: the parser reads each primitive's
        # orthographic view straight from entity.dxf.layer.
        if v.name not in doc.layers:
            doc.layers.add(v.name)
        _add_edges(msp, v.visible, "Continuous", v.name)
        _add_edges(msp, v.hidden, "HIDDEN", v.name)

    for i, m in enumerate(marks):
        name = f"SW_CENTERMARKSYMBOL_{i}"
        if name in doc.blocks:
            doc.blocks.delete_block(name, safe=False)
        blk = doc.blocks.new(name=name)
        for p0, p1 in mark_block_lines(m):
            blk.add_line(p0, p1, dxfattribs={"layer": "10", "linetype": "BYLAYER"})
        msp.add_blockref(name, (m.cx, m.cy), dxfattribs={"layer": "10"})

    # keep the sheet extents header consistent
    doc.header["$EXTMIN"] = (0.0, 0.0, 0.0)
    doc.header["$EXTMAX"] = (297.0, 210.0, 0.0)
    doc.saveas(path)
