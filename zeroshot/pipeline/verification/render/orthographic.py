"""STEP -> the visible and hidden edges of each orthographic view.

OCC hidden-line removal projects the solid, and the edges come back as exact
curves in the view's own U and V, in model units: a coordinate read off a view
is a measurement of the solid, not a sheet position.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import cadquery as cq
import numpy as np
from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Curve2d, BRepAdaptor_Surface
from OCP.BRepLib import BRepLib
from OCP.BRepLProp import BRepLProp_SLProps
from OCP.GeomAbs import GeomAbs_C0, GeomAbs_G1
from OCP.gp import gp_Ax2, gp_Circ, gp_Dir, gp_Lin, gp_Pnt
from OCP.HLRAlgo import HLRAlgo_Projector
from OCP.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape
from OCP.ShapeAnalysis import ShapeAnalysis_CanonicalRecognition
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED
from OCP.TopExp import TopExp
from OCP.TopoDS import TopoDS, TopoDS_Edge, TopoDS_Face, TopoDS_Shape
from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
from scipy.spatial import cKDTree

from zeroshot.pipeline.stages.interpretation.contracts import (
    AXIS_VECTOR,
    Axis,
    View,
    cross_axis,
)

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

# How far apart two lines may be and still count as one, as a fraction of the
# part's diagonal.
TOL_FRAC = 1e-3

# Faces that meet at a smaller angle join smoothly. The STEP reader marks many
# such seams sharp, e.g. between fillet patches.
SMOOTH_ANGLE = math.radians(1.0)


def load_shape(step_path: Path) -> cq.Shape:
    """Every solid a STEP file holds, as one shape."""
    read = cq.importers.importStep(str(step_path)).vals()
    return cq.Compound.makeCompound([s for s in read if isinstance(s, cq.Shape)])


def project(
    shape: cq.Shape, u_axis: Axis, v_axis: Axis
) -> tuple[list[cq.Edge], list[cq.Edge]]:
    """The visible and hidden edges of `shape`, drawn with +U right and +V up."""
    out = cross_axis(u_axis, v_axis)
    if out is None:
        raise ValueError(f"u_axis {u_axis} and v_axis {v_axis} span no plane")
    # The eye is on `out`, and naming U as the sheet's X leaves V as its Y.
    eye = gp_Dir(*AXIS_VECTOR[out])
    frame = gp_Ax2(gp_Pnt(), eye, gp_Dir(*AXIS_VECTOR[u_axis]))
    algo = HLRBRep_Algo()
    algo.Add(_smoothed(shape, eye).wrapped)
    algo.Projector(HLRAlgo_Projector(frame))
    algo.Update()
    algo.Hide()
    drawn = HLRBRep_HLRToShape(algo)
    # Smooth (tangent) edges are left out, as the input drawings leave them out.
    visible = _edges(drawn.VCompound(), drawn.OutLineVCompound())
    hidden = _edges(drawn.HCompound(), drawn.OutLineHCompound())
    if not visible and not hidden:
        return [], []

    # Tight: a loose box can be several times the part and let strays through.
    part = shape.BoundingBox()
    tol = TOL_FRAC * part.DiagonalLength
    footprint = (_span(part, u_axis), _span(part, v_axis))
    visible = [_simplest(e, tol) for e in visible if _drawable(e, footprint, tol)]
    hidden = [_simplest(e, tol) for e in hidden if _drawable(e, footprint, tol)]
    return visible, _uncovered(hidden, visible, tol)


def _smoothed(shape: cq.Shape, eye: gp_Dir) -> cq.Shape:
    """A copy whose seams between tangent faces are smooth, save on the outline."""
    copy = shape.copy()
    builder = BRep_Builder()
    faces_of = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndUniqueAncestors_s(
        copy.wrapped, TopAbs_EDGE, TopAbs_FACE, faces_of
    )
    for i in range(1, faces_of.Extent() + 1):
        edge = TopoDS.Edge(faces_of.FindKey(i))
        faces = [TopoDS.Face(face) for face in faces_of.FindFromIndex(i)]
        if len(faces) != 2 or BRep_Tool.Continuity_s(edge, *faces) != GeomAbs_C0:
            continue
        normals = [
            (a, b)
            for a, b in zip(_normals(edge, faces[0]), _normals(edge, faces[1]))
            if a is not None and b is not None
        ]
        tangent = all(a.Angle(b) < SMOOTH_ANGLE for a, b in normals)
        # Along an outline both faces turn edge-on, and HLR may draw no silhouette.
        edge_on = all(abs(a.Dot(eye)) < math.sin(SMOOTH_ANGLE) for a, _ in normals)
        if normals and tangent and not edge_on:
            builder.Continuity(edge, *faces, GeomAbs_G1)
    return copy


def _normals(edge: TopoDS_Edge, face: TopoDS_Face) -> list[gp_Dir | None]:
    """The face's outward normal at points along `edge`, None where undefined."""
    curve, surface = BRepAdaptor_Curve2d(edge, face), BRepAdaptor_Surface(face)
    normals = []
    # Short of the ends, where a patch may pinch to a point.
    for t in np.linspace(curve.FirstParameter(), curve.LastParameter(), 25)[1:-1]:
        uv = curve.Value(t)
        props = BRepLProp_SLProps(surface, uv.X(), uv.Y(), 1, 1e-9)
        normal = props.Normal() if props.IsNormalDefined() else None
        if normal is not None and face.Orientation() == TopAbs_REVERSED:
            normal.Reverse()
        normals.append(normal)
    return normals


def _span(box: cq.BoundBox, axis: Axis) -> tuple[float, float]:
    """The range `box` covers along a sheet axis."""
    low, high = getattr(box, f"{axis[1]}min"), getattr(box, f"{axis[1]}max")
    return (low, high) if axis[0] == "+" else (-high, -low)


def _drawable(
    edge: cq.Edge, footprint: tuple[tuple[float, float], ...], tol: float
) -> bool:
    """Long enough to see, and inside the part's footprint.

    Where surfaces meet tangentially, HLR can trace an outline far off the part.
    """
    if edge.Length() < tol:
        return False
    # Sampled: an optimal bounding box per edge costs more than the HLR itself.
    u, v = _samples(edge, edge.Length() / 16).T
    (u_low, u_high), (v_low, v_high) = footprint
    return bool(
        u.min() >= u_low - tol
        and u.max() <= u_high + tol
        and v.min() >= v_low - tol
        and v.max() <= v_high + tol
    )


def _edges(*compounds: TopoDS_Shape) -> list[cq.Edge]:
    edges: list[cq.Edge] = []
    for compound in compounds:
        if not compound.IsNull():
            # HLR edges carry only 2D curves; build 3D ones, as cadquery's getSVG does.
            BRepLib.BuildCurves3d_s(compound, 1e-6)
            edges += cq.Shape.cast(compound).Edges()
    return edges


def _simplest(edge: cq.Edge, tol: float) -> cq.Edge:
    """A spline that runs along a line or a circle, as that line or circle."""
    if edge.geomType() != "BSPLINE":
        return edge
    simpler = _as_line_or_circle(edge, tol)
    # An out-and-back sliver passes for a line or circle it does not follow.
    if simpler is None or abs(simpler.Length() - edge.Length()) > tol:
        return edge
    return simpler


def _as_line_or_circle(edge: cq.Edge, tol: float) -> cq.Edge | None:
    recognition = ShapeAnalysis_CanonicalRecognition(edge.wrapped)
    if not edge.IsClosed() and recognition.IsLine(tol, gp_Lin()):
        return cq.Edge.makeLine(edge.startPoint(), edge.endPoint())
    circle = gp_Circ()
    if not recognition.IsCircle(tol, circle):
        return None
    if edge.IsClosed():
        return cq.Edge.makeCircle(circle.Radius(), cq.Vector(circle.Location()))
    # Through its own midpoint, so the arc cannot come out as its complement.
    return cq.Edge.makeThreePointArc(
        edge.startPoint(), edge.positionAt(0.5), edge.endPoint()
    )


def _uncovered(
    hidden: list[cq.Edge], visible: list[cq.Edge], tol: float
) -> list[cq.Edge]:
    """Hidden edges that do not lie under a visible one along their whole run."""
    if not hidden or not visible:
        return hidden
    ink = cKDTree(np.vstack([_samples(edge, tol / 2) for edge in visible]))
    return [e for e in hidden if ink.query(_samples(e, tol / 2))[0].max() > tol]


def _samples(edge: cq.Edge, step: float) -> np.ndarray:
    # Spaced by parameter: spacing by length costs a solve per point.
    curve = BRepAdaptor_Curve(edge.wrapped)
    count = max(2, int(edge.Length() / step) + 1)
    ts = np.linspace(curve.FirstParameter(), curve.LastParameter(), count)
    return np.array([(p.X(), p.Y()) for p in map(curve.Value, ts)])
