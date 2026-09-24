"""STEP -> the orthographic projections a drawing asks for.

Reads the solid, runs hidden-line removal once per view frame, and returns the
projected 2D primitives in model units, which is what the view is written in:
a coordinate on a projection is a model measurement, not a sheet position, so
it does not line up with an input drawing's own coordinates.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopoDS import TopoDS_Shape

from zeroshot.pipeline.stages.interpretation.contracts import (
    AXIS_VECTOR,
    Axis,
    View,
    cross_axis,
)
from zeroshot.pipeline.verification.render._hlr import (
    ViewProjection,
    project,
)

type _Direction = tuple[float, float, float]

# The sheet axes to draw each view in, as a drawing declared them.
type ViewFrames = Mapping[View, tuple[Axis, Axis]]

# Every orthographic view, drawn the way a page that turns no view lays it out.
STANDARD_VIEW_FRAMES: ViewFrames = MappingProxyType(
    {
        View.FRONT: ("+x", "+z"),
        View.BACK: ("-x", "+z"),
        View.TOP: ("+x", "+y"),
        View.BOTTOM: ("+x", "-y"),
        View.LEFT: ("-y", "+z"),
        View.RIGHT: ("+y", "+z"),
    }
)


def frame_of(u_axis: Axis, v_axis: Axis) -> tuple[_Direction, _Direction]:
    """The (eye_dir, up_dir) drawing a sheet whose +U and +V are these axes.

    `up` becomes screen +Y and screen +X is up x eye, so gazing along -(u x v)
    with up = v puts +U rightwards, which is what the drawing declared.
    """
    out = cross_axis(u_axis, v_axis)
    if out is None:
        raise ValueError(f"u_axis {u_axis} and v_axis {v_axis} span no plane")
    return AXIS_VECTOR[out], AXIS_VECTOR[v_axis]


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
    views: ViewFrames = STANDARD_VIEW_FRAMES,
    include_smooth: bool = False,
) -> dict[View, ViewProjection]:
    """Project ``shape`` into each view, in the sheet axes that view declared.

    ``include_smooth`` adds tangent (Rg1) edges, which GT suppresses.
    """
    diagonal = bbox_diagonal(shape)
    kwargs = {
        "deflection": DEFLECTION_FRAC * diagonal,
        "include_smooth": include_smooth,
        "merge_tol": MERGE_TOL_FRAC * diagonal,
    }
    return {
        view: _project_nonempty(shape, frame_of(*axes), **kwargs)
        for view, axes in views.items()
    }


def require_drawable(projection: ViewProjection, name: str) -> None:
    """Refuse a view with no finite extent worth drawing.

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
