"""Read a DXF drawing into flat geometry records.

SPLINE/ELLIPSE are flattened to polylines and LWPOLYLINE/INSERT are expanded to
their virtual entities, so everything downstream sees only lines, circles and
arcs. Dashed linetypes are separated out as hidden lines.
"""

import math
from itertools import pairwise

import ezdxf
from ezdxf.lldxf.const import DXFError

# Degenerate geometry that neither expands nor flattens is dropped, not fatal.
MALFORMED = (DXFError, ValueError, TypeError, ZeroDivisionError)

DASH_LINETYPES = frozenset(
    {"HIDDEN", "DASHED", "CENTER", "CENTERX2", "PHANTOM", "DOT2"}
)

KINDS = ("line", "dash_line", "circle", "arc")


def _resolve_linetype(entity, doc):
    name = entity.dxf.get("linetype", "BYLAYER")
    if name.upper() in ("BYLAYER", "BYBLOCK"):
        name = doc.layers.get(entity.dxf.layer).dxf.linetype
    return name.upper()


def _flatten(entities, doc, depth=0):
    """Expand blocks and polylines recursively into primitive entities."""
    for entity in entities:
        if entity.dxftype() in ("INSERT", "LWPOLYLINE", "POLYLINE") and depth < 4:
            try:
                virtual = list(entity.virtual_entities())
            except MALFORMED:
                continue
            yield from _flatten(virtual, doc, depth + 1)
        else:
            yield entity


def extract(dxf_path, sagitta=0.05, segments=24):
    doc = ezdxf.readfile(dxf_path)
    data = {kind: [] for kind in KINDS}

    for entity in _flatten(doc.modelspace(), doc):
        kind = entity.dxftype()
        dashed = _resolve_linetype(entity, doc) in DASH_LINETYPES

        if kind == "LINE":
            data["dash_line" if dashed else "line"].append(
                {
                    "start_x": entity.dxf.start.x,
                    "start_y": entity.dxf.start.y,
                    "end_x": entity.dxf.end.x,
                    "end_y": entity.dxf.end.y,
                }
            )
        elif kind == "CIRCLE" and entity.dxf.radius > 0:
            data["circle"].append(
                {
                    "center_x": entity.dxf.center.x,
                    "center_y": entity.dxf.center.y,
                    "radius": entity.dxf.radius,
                    "dashed": dashed,
                }
            )
        elif kind == "ARC" and entity.dxf.radius > 0:
            data["arc"].append(
                {
                    "center_x": entity.dxf.center.x,
                    "center_y": entity.dxf.center.y,
                    "radius": entity.dxf.radius,
                    "start_angle": entity.dxf.start_angle,
                    "end_angle": entity.dxf.end_angle,
                    "dashed": dashed,
                }
            )
        elif kind in ("SPLINE", "ELLIPSE"):
            try:
                points = list(entity.flattening(distance=sagitta, segments=segments))
            except MALFORMED:
                continue
            key = "dash_line" if dashed else "line"
            for a, b in pairwise(points):
                if math.dist((a.x, a.y), (b.x, b.y)) > 1e-9:
                    # `curved` keeps these out of the axis-aligned edge search
                    data[key].append(
                        {
                            "start_x": a.x,
                            "start_y": a.y,
                            "end_x": b.x,
                            "end_y": b.y,
                            "curved": True,
                        }
                    )

    return data
