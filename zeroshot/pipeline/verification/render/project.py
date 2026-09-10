"""STEP -> the orthographic projections a drawing asks for.

Reads the solid, runs hidden-line removal once per view frame, and returns the
projected 2D primitives in model units. `to_own_corner` then reads each view
from its own bottom-left corner, which is where the drawing contract measures a
sheet from.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from pathlib import Path

from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopoDS import TopoDS_Shape

from zeroshot.pipeline.stages.drawings.contracts import VIEW_FRAME, View
from zeroshot.pipeline.verification.render._hlr import (
    Arc,
    Circle,
    Ellipse,
    Polyline,
    ProjectedEdges,
    Segment,
    ViewProjection,
    project,
)

type _Direction = tuple[float, float, float]

_AXES: Mapping[str, _Direction] = {
    "+x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "+z": (0.0, 0.0, 1.0),
    "-z": (0.0, 0.0, -1.0),
}

# What a third-angle drawing shows, and what this module draws when a caller
# does not say. Verified against GT by raster-IoU.
THIRD_ANGLE: tuple[View, ...] = (View.FRONT, View.TOP, View.RIGHT)


def frame_of(view: View) -> tuple[_Direction, _Direction]:
    """The (eye_dir, up_dir) the contract fixes for `view`.

    A frame's `out` axis is the projection plane's outward normal, so it points
    from the model toward the viewer and the gaze runs the other way. `up`
    becomes screen +Y, and screen +X is then up x eye, which is the frame's
    `right`. Model XY is horizontal and +Z is vertical; in particular, top
    reads (+X, +Y), front reads (+X, +Z), and right reads (+Y, +Z).
    """
    if view not in VIEW_FRAME:
        raise ValueError(f"{view.value} is not an orthographic view")
    _, up, out = VIEW_FRAME[view]
    return _AXES[out], _AXES[up]


# Curve discretisation and edge-merge tolerance, expressed as fractions of the
# model's bounding-box diagonal so the renderer is unit-agnostic.
DEFLECTION_FRAC = 0.02
MERGE_TOL_FRAC = 5e-4

# An exactly axis-aligned view direction can make HLRBRep_Algo return *zero*
# edges when the shape has faces exactly perpendicular / parallel to it
# (observed on a thin disc for the right view).  A tiny tilt of the frame breaks
# the degeneracy and recovers the full projection; the sub-milliradian rotation
# is far below the drawing's calibration tolerance and is only ever applied to a
# view that would otherwise be empty, so good views are bit-for-bit unchanged.
_TILT_EPS = (1e-3, 3e-3, 1e-2)

# A view narrower than this on either axis carries no recoverable shape. It
# only fires on pathological inputs: an empty projection, or a knife-edge view
# of a near-zero-thickness plate that collapses to a single line.
MIN_VIEW_EXTENT_MM = 0.05


class DegenerateDrawingError(RuntimeError):
    """A projection holds nothing a drawing could be made of."""


def load_shape(step_path: Path) -> TopoDS_Shape:
    """Read the single shape out of a STEP file."""
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))
    if status != 1:  # IFSelect_RetDone
        raise RuntimeError(f"STEP read failed ({status}): {step_path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    if shape is None or shape.IsNull():
        raise RuntimeError(f"empty shape: {step_path}")
    return shape


def bbox_diagonal(shape: TopoDS_Shape) -> float:
    box = Bnd_Box()
    brepbndlib.Add(shape, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
    return max(math.sqrt(dx * dx + dy * dy + dz * dz), 1e-6)


def _project_nonempty(
    shape: TopoDS_Shape,
    frame: tuple[tuple[float, float, float], tuple[float, float, float]],
    deflection: float,
    include_smooth: bool,
    merge_tol: float,
) -> ViewProjection:
    """Project one view, retrying with a tilted frame if HLR returns nothing."""
    eye_dir, up_dir = frame
    kwargs = {
        "deflection": deflection,
        "include_smooth": include_smooth,
        "merge_tol": merge_tol,
    }
    projection = project(shape, eye_dir, up_dir, **kwargs)
    if projection.visible.count() + projection.hidden.count() > 0:
        return projection
    for eps in _TILT_EPS:
        tilted = tuple(component + eps for component in eye_dir)
        projection = project(shape, tilted, up_dir, **kwargs)
        if projection.visible.count() + projection.hidden.count() > 0:
            return projection
    return projection


def project_views(
    shape: TopoDS_Shape,
    views: Iterable[View] = THIRD_ANGLE,
    include_smooth: bool = False,
) -> dict[View, ViewProjection]:
    """Project ``shape`` for each of ``views``.

    ``include_smooth`` adds tangent (Rg1) edges, which GT suppresses.
    """
    diagonal = bbox_diagonal(shape)
    kwargs = {
        "deflection": DEFLECTION_FRAC * diagonal,
        "include_smooth": include_smooth,
        "merge_tol": MERGE_TOL_FRAC * diagonal,
    }
    return {view: _project_nonempty(shape, frame_of(view), **kwargs) for view in views}


def _translated(edges: ProjectedEdges, dx: float, dy: float) -> ProjectedEdges:
    """Move every primitive. No scale or rotation, so angles carry over."""

    def point(p: tuple[float, float]) -> tuple[float, float]:
        return (p[0] + dx, p[1] + dy)

    return ProjectedEdges(
        segments=[Segment(point(e.p0), point(e.p1)) for e in edges.segments],
        arcs=[Arc(point(e.center), e.radius, e.a0, e.a1, e.ccw) for e in edges.arcs],
        circles=[Circle(point(e.center), e.radius) for e in edges.circles],
        ellipses=[
            Ellipse(point(e.center), e.rmaj, e.rmin, e.rot, e.a0, e.a1)
            for e in edges.ellipses
        ],
        polylines=[Polyline([point(p) for p in e.pts]) for e in edges.polylines],
    )


def to_own_corner(projection: ViewProjection, name: str) -> ViewProjection:
    """The same view read from its own bottom-left corner, at 1:1.

    `name` only exists to say which view was refused.
    """
    x_min, y_min, x_max, y_max = projection.bbox(include_hidden=True)
    if not all(math.isfinite(c) for c in (x_min, y_min, x_max, y_max)):
        raise DegenerateDrawingError(
            f"view {name!r} bbox is not finite: {(x_min, y_min, x_max, y_max)}"
        )
    width, height = x_max - x_min, y_max - y_min
    if width < MIN_VIEW_EXTENT_MM or height < MIN_VIEW_EXTENT_MM:
        raise DegenerateDrawingError(
            f"view {name!r} has near-zero extent (w={width:.4f}, h={height:.4f} mm)"
        )
    return ViewProjection(
        visible=_translated(projection.visible, -x_min, -y_min),
        hidden=_translated(projection.hidden, -x_min, -y_min),
    )
