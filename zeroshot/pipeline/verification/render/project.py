"""STEP -> the three orthographic projections of a third-angle drawing.

Reads the solid, runs hidden-line removal once per view frame, and returns the
projected 2D primitives in model units, centred on the projection origin.
arrange.py turns them into sheet-mm positions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopoDS import TopoDS_Shape

from zeroshot.pipeline.verification.render._hlr import ViewProjection, project

# Third-angle view frames, verified against GT by raster-IoU.  Each entry is
# (eye_dir, up_dir).  eye_dir points from the model *toward the viewer* -- it is
# the projection plane's outward normal, so the gaze runs the other way; up_dir
# becomes screen +Y, and screen +X is then up_dir x eye_dir.
#
# The last column is where a model point (x, y, z) lands in screen coordinates.
# top's screen +Y is -Z because in third-angle projection moving up the top view
# means moving *away* from the front view's viewer, and +Z faces that viewer.
FRONT = ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0))  # eye +Z, up +Y  -> screen ( X,  Y)
TOP = ((0.0, 1.0, 0.0), (0.0, 0.0, -1.0))  # eye +Y, up -Z  -> screen ( X, -Z)
RIGHT = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))  # eye +X, up +Y  -> screen (-Z,  Y)

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


@dataclass(frozen=True)
class ViewProjections:
    """The three projections of one part.  Field name == view name == DXF layer."""

    front: ViewProjection
    top: ViewProjection
    right: ViewProjection


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


def project_views(shape: TopoDS_Shape, include_smooth: bool = False) -> ViewProjections:
    """Project ``shape`` for the front, top and right frames.

    ``include_smooth`` adds tangent (Rg1) edges, which GT suppresses.
    """
    diagonal = bbox_diagonal(shape)
    kwargs = {
        "deflection": DEFLECTION_FRAC * diagonal,
        "include_smooth": include_smooth,
        "merge_tol": MERGE_TOL_FRAC * diagonal,
    }
    return ViewProjections(
        front=_project_nonempty(shape, FRONT, **kwargs),
        top=_project_nonempty(shape, TOP, **kwargs),
        right=_project_nonempty(shape, RIGHT, **kwargs),
    )
