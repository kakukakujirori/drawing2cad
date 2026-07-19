"""render3d.py: STEP -> three ECCV-style perspective PNGs.

Public entry point::

    generate_render3d(step_path, paths, cfg=None) -> dict

Pure and synchronous: no multiprocessing, no timeouts, no import-time work.
Batch isolation / timeouts are handled by render_dataset.py.

Styles (data/eccv2026-cad-challenge-data/train/render_3d/<style>/NNNNNN.png):
  - hlg_perspective: pure line-art hidden-line-grayed drawing (visible edges
    thin solid black, hidden edges faint gray dashed), perspective camera.
  - transparent_shaded_edges_perspective: translucent gray-shaded solid with
    ALL edges (front and back, seen through the translucency) drawn on top.
  - hlg_translucent_faces_perspective: the hlg line art composited over a
    very faint translucent face fill (same camera, pixel-aligned).

Camera (calibrated on data/eccv2026-cad-challenge-data/train, see
calibrate_render3d.py): world up is +Y (matches src/render/techdraw.py's
IoU-verified FRONT/TOP/RIGHT frames: FRONT looks -Z with up +Y). The 3D
renders use a single fixed SolidWorks-style trimetric/isometric direction
(same relative camera for every part, auto-framed to each part's bounding
box) with a mild perspective (see Render3dConfig defaults below).

Pipeline:
  1. OCC HLR (perspective projector) -> visible / hidden 2D polylines ->
     fit-to-canvas affine -> SVG -> cairosvg PNG  (hlg_perspective).
  2. OCC incremental-mesh tessellation -> pyvista PolyData -> off-screen
     translucent-shaded render, camera derived from the SAME eye/N/right/up
     and the SAME fit affine as pass 1, so the two passes are pixel-aligned
     (transparent_shaded_edges_perspective).
  3. Composite: faint alpha-blended copy of the shaded pass under the
     hlg line art (hlg_translucent_faces_perspective).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cairosvg
import numpy as np
from PIL import Image
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.GCPnts import GCPnts_QuasiUniformDeflection
from OCC.Core.GeomAbs import GeomAbs_BSplineCurve, GeomAbs_Line
from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt
from OCC.Core.HLRAlgo import HLRAlgo_Projector
from OCC.Core.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import (
    GeomAbs_Cone,
    GeomAbs_Cylinder,
    GeomAbs_Plane,
    GeomAbs_Sphere,
    GeomAbs_Torus,
)
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer, topexp
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import topods
from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape

from src.render.config import RENDER3D_SIZE, Render3dPaths


# ---------------------------------------------------------------------------
# Calibrated defaults (see calibrate_render3d.py; world up is +Y).
# ---------------------------------------------------------------------------


@dataclass
class Render3dConfig:
    # camera direction: unit vector from the model centre TOWARD the eye,
    # expressed in the model's own (X, Y, Z) with +Y "up" (matches techdraw.py).
    # Calibrated on GT (visual octant selection on the chiral part 000123 after
    # fixing the horizontal-mirror bug): SolidWorks standard Isometric (1,1,1).
    eye_dir: tuple[float, float, float] = (1.0, 1.0, 1.0)
    world_up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    roll_deg: float = 0.0  # rotation of the image plane about the view axis
    dist_factor: float = 8.0  # eye distance = bbox_diagonal * dist_factor
    focus_factor: float = 0.15  # focal-plane distance = eye_distance * focus_factor
    # Framing: GT fits the part's TRUE bounding sphere (max vertex distance
    # from the AABB centre), not the projected ink bbox -- consistent incl.
    # bodies of revolution, where the AABB-diagonal sphere over-estimates the
    # radius. fill_factor = (projected sphere diameter) / (canvas height);
    # median implied value over 12 GT parts with the calibrated camera is
    # 0.887 (p10-p90 spread 0.82-1.2 -- GT's own framing varies mildly).
    fill_factor: float = 0.88
    margin_frac: float = 0.08  # fallback 2D-fit margin if tessellation fails
    canvas_w: int = RENDER3D_SIZE[0]
    canvas_h: int = RENDER3D_SIZE[1]
    deflection_frac: float = 0.0015  # curve discretisation, fraction of bbox diagonal

    # hlg line style
    vis_stroke_width: float = 1.3
    hid_stroke_width: float = 1.0
    hid_color: str = "#b5b5b5"
    hid_dash: str = "4,3"

    # shaded pass style (calibrated against GT transparent_shaded_edges pixel
    # stats on 000123: face mean/p25/p75 within ~5 gray levels of GT)
    shaded_color: str = "#c4cdd4"
    shaded_opacity: float = 0.6
    shaded_ambient: float = 0.5
    shaded_diffuse: float = 0.8
    shaded_edge_color: str = "#404040"
    shaded_edge_width: float = 1.0
    shaded_view_angle_pad: float = 1.0  # multiply the derived vertical FOV by this

    # hlg_translucent_faces compositing: GT face pixels average ~239 gray
    # (near-white with soft shading); calibrated so white*(1-a)+shaded*a with
    # the shaded face mean ~195 lands on the GT mean.
    translucent_face_opacity: float = 0.27


# ---------------------------------------------------------------------------
# Small vector helpers
# ---------------------------------------------------------------------------


def _norm(v):
    length = math.sqrt(sum(c * c for c in v))
    return tuple(c / length for c in v)


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


@dataclass
class Camera:
    eye: tuple[float, float, float]
    N: tuple[float, float, float]  # view direction, eye -> scene
    right: tuple[float, float, float]  # screen +X
    up2d: tuple[float, float, float]  # screen +Y (true screen up, = right x N)
    focus: float
    D: float


def _build_camera(shape, cfg: Render3dConfig) -> Camera:
    xmin, ymin, zmin, xmax, ymax, zmax = _bbox(shape)
    center = ((xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2)
    diag = math.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2)
    diag = max(diag, 1e-9)

    eye_dir = _norm(cfg.eye_dir)
    D = diag * cfg.dist_factor
    eye = tuple(center[i] + eye_dir[i] * D for i in range(3))
    N = tuple(-c for c in eye_dir)

    # Screen frame, following the GT-verified techdraw convention: the OCC
    # projector's Z axis points from the scene TOWARD the viewer (= eye_dir)
    # and its Y axis is set explicitly to the screen-up; OCC then derives
    # screen X = Y x Z, giving an unmirrored right-handed view (a marker-box
    # probe confirmed the earlier N-as-Z construction was horizontally
    # mirrored).  screen-up = world_up made orthogonal to eye_dir, then
    # rolled about the view axis by roll_deg.
    w_dot_e = sum(cfg.world_up[i] * eye_dir[i] for i in range(3))
    up0 = _norm(tuple(cfg.world_up[i] - w_dot_e * eye_dir[i] for i in range(3)))
    right0 = _cross(up0, eye_dir)  # screen X = Y x Z

    theta = math.radians(cfg.roll_deg)
    ct, st = math.cos(theta), math.sin(theta)
    up2d = _norm(tuple(up0[i] * ct + right0[i] * st for i in range(3)))
    right = _cross(up2d, eye_dir)

    focus = D * cfg.focus_factor
    return Camera(eye=eye, N=N, right=right, up2d=up2d, focus=focus, D=D)


def _bbox(shape):
    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox)
    return bbox.Get()


def _load_shape(step_path: Path):
    step_path = Path(step_path)
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))
    if status != 1:  # IFSelect_RetDone
        raise RuntimeError(f"STEP read failed ({status}): {step_path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    if shape is None or shape.IsNull():
        raise RuntimeError(f"empty shape: {step_path}")
    return shape


# ---------------------------------------------------------------------------
# Pass 1: HLR perspective line art
# ---------------------------------------------------------------------------

_VIS_NAMES = ("VCompound", "OutLineVCompound")
_HID_NAMES = ("HCompound", "OutLineHCompound")
# NOTE: Rg1Line* (smooth/tangent + seam edges) deliberately excluded -- GT
# (SolidWorks) hides them in all three render_3d styles; including them adds
# revolution-surface seam lines GT never draws (ablation: rg1_ablation.png).


def _smooth_resample(pts, factor=8, max_pts=600):
    """Cubic-spline resample of a coarse polyline whose vertices lie ON the
    true curve. OCC's perspective HLR emits every projected curve as a
    degree-1 B-spline (a ~14-segment polyline, visibly faceted at 1400 px);
    its poles are exact curve samples, so interpolating through them
    reconstructs the curve to <0.1 px."""
    from scipy.interpolate import CubicSpline  # lazy: keep spawn imports light

    arr = np.asarray(pts, dtype=np.float64)
    seg = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    total = float(seg.sum())
    if total <= 0.0:
        return pts
    t = np.concatenate([[0.0], np.cumsum(seg)])
    keep = np.concatenate([[True], seg > 1e-12 * max(total, 1.0)])
    arr, t = arr[keep], t[keep]
    if len(arr) < 3:
        return [tuple(p) for p in arr]
    closed = np.linalg.norm(arr[0] - arr[-1]) < 1e-9 * max(total, 1.0)
    bc = "periodic" if closed and len(arr) >= 4 else "not-a-knot"
    spline = CubicSpline(t, arr, axis=0, bc_type=bc)
    n = min(max(len(arr) * factor, 32), max_pts)
    dense = spline(np.linspace(t[0], t[-1], n))
    return [tuple(p) for p in dense]


def _polylines_from_compound(compound, deflection):
    lines = []
    if compound is None or compound.IsNull():
        return lines
    exp = TopExp_Explorer(compound, TopAbs_EDGE)
    while exp.More():
        edge = topods.Edge(exp.Current())
        exp.Next()
        try:
            curve = BRepAdaptor_Curve(edge)
        except Exception:  # noqa: BLE001
            continue
        u0, u1 = curve.FirstParameter(), curve.LastParameter()
        pts = None
        ctype = curve.GetType()
        if ctype == GeomAbs_Line:
            p1, p2 = curve.Value(u0), curve.Value(u1)
            pts = [(p1.X(), p1.Y()), (p2.X(), p2.Y())]
        elif ctype == GeomAbs_BSplineCurve and curve.BSpline().Degree() == 1:
            bs = curve.BSpline()
            poles = [
                (bs.Pole(i).X(), bs.Pole(i).Y()) for i in range(1, bs.NbPoles() + 1)
            ]
            pts = poles if len(poles) == 2 else _smooth_resample(poles)
        else:
            try:
                disc = GCPnts_QuasiUniformDeflection(curve, deflection, u0, u1)
                if disc.IsDone() and disc.NbPoints() >= 2:
                    pts = [
                        (disc.Value(i).X(), disc.Value(i).Y())
                        for i in range(1, disc.NbPoints() + 1)
                    ]
            except Exception:  # noqa: BLE001
                pts = None
            if not pts:
                # Dense uniform-in-parameter sampling. NEVER a 2-point chord:
                # HLR outline (silhouette) edges are curved B-splines and a
                # chord fallback erases them from the drawing.
                n = 96
                pts = []
                for i in range(n):
                    p = curve.Value(u0 + (u1 - u0) * i / (n - 1))
                    pts.append((p.X(), p.Y()))
        lines.append(pts)
    return lines


def _hlr_project(
    shape,
    camera: Camera,
    cfg: Render3dConfig,
    deflection_scale: float = 1.0,
    deflection: float | None = None,
):
    """Runs OCC HLR with a perspective projector; returns (visible, hidden)
    lists of 2D polylines in camera-plane (model) units.

    ``deflection`` (projection-plane units) overrides the legacy bbox-diagonal
    heuristic; pass ``target_px / tf.scale`` for a screen-space chord error.
    """
    if deflection is None:
        xmin, ymin, zmin, xmax, ymax, zmax = _bbox(shape)
        diag = math.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2)
        deflection = max(diag * cfg.deflection_frac * deflection_scale, 1e-6)

    # Z toward the viewer + explicit Y (techdraw's verified convention).
    ax2 = gp_Ax2(gp_Pnt(*camera.eye), gp_Dir(*(-c for c in camera.N)))
    ax2.SetYDirection(gp_Dir(*camera.up2d))
    projector = HLRAlgo_Projector(ax2, camera.focus)

    hlr = HLRBRep_Algo()
    hlr.Add(shape)
    hlr.Projector(projector)
    hlr.Update()
    hlr.Hide()
    hts = HLRBRep_HLRToShape(hlr)

    visible = []
    for name in _VIS_NAMES:
        visible += _polylines_from_compound(getattr(hts, name)(), deflection)
    hidden = []
    for name in _HID_NAMES:
        hidden += _polylines_from_compound(getattr(hts, name)(), deflection)
    return visible, hidden


# ---------------------------------------------------------------------------
# Mesh-based hidden-line pass (default). OCC's exact perspective HLR has two
# defects this replaces (both confirmed on val parts, 2026-07-18):
#   - silhouette generator lines of conical faces (chamfer bands on circular
#     edges) are emitted over an UNTRIMMED parameter range: full-length phantom
#     lines crossing the body;
#   - misclassification around tangency (visible feature edges reported
#     hidden, e.g. counterbore rims on a front-facing cap).
# Here the drawing is rebuilt from first principles: real BRep edges (minus
# parameterisation seams and same-surface split edges) plus true mesh
# silhouettes, each sampled densely and classified against a VTK depth buffer
# rendered with the SAME camera as the shaded pass. Tangent (fillet/chamfer
# boundary) edges are INCLUDED: GT draws them (000029: visible tangent circles
# solid black, hidden ones gray dashed); only seams stay hidden.
# ---------------------------------------------------------------------------


def _same_surface_domain(f1, f2, tol) -> bool:
    """True if two faces lie on the same geometric surface (a modelling split,
    e.g. coplanar halves or a cylinder split at 180deg) -- such shared edges
    are artefacts and are never drawn."""
    s1, s2 = BRepAdaptor_Surface(f1), BRepAdaptor_Surface(f2)
    t1, t2 = s1.GetType(), s2.GetType()
    if t1 != t2:
        return False

    def close(a, b, t=tol):
        return abs(a - b) <= t

    def pclose(p, q, t=tol):
        return p.Distance(q) <= t

    def parallel(d1, d2):
        return abs(d1.Dot(d2)) >= 1.0 - 1e-9

    if t1 == GeomAbs_Plane:
        p1, p2 = s1.Plane(), s2.Plane()
        if not parallel(p1.Axis().Direction(), p2.Axis().Direction()):
            return False
        n = p1.Axis().Direction()
        d = p2.Location().XYZ().Subtracted(p1.Location().XYZ())
        return abs(d.Dot(n.XYZ())) <= tol
    if t1 == GeomAbs_Cylinder:
        c1, c2 = s1.Cylinder(), s2.Cylinder()
        if not parallel(c1.Axis().Direction(), c2.Axis().Direction()):
            return False
        if not close(c1.Radius(), c2.Radius()):
            return False
        d = c2.Location().XYZ().Subtracted(c1.Location().XYZ())
        ax = c1.Axis().Direction().XYZ()
        d.Subtract(ax.Multiplied(d.Dot(ax)))
        return d.Modulus() <= tol
    if t1 == GeomAbs_Cone:
        c1, c2 = s1.Cone(), s2.Cone()
        return (
            parallel(c1.Axis().Direction(), c2.Axis().Direction())
            and close(c1.SemiAngle(), c2.SemiAngle(), 1e-9)
            and pclose(c1.Apex(), c2.Apex())
        )
    if t1 == GeomAbs_Sphere:
        c1, c2 = s1.Sphere(), s2.Sphere()
        return close(c1.Radius(), c2.Radius()) and pclose(c1.Location(), c2.Location())
    if t1 == GeomAbs_Torus:
        c1, c2 = s1.Torus(), s2.Torus()
        return (
            parallel(c1.Axis().Direction(), c2.Axis().Direction())
            and close(c1.MajorRadius(), c2.MajorRadius())
            and close(c1.MinorRadius(), c2.MinorRadius())
            and pclose(c1.Location(), c2.Location())
        )
    return False


def _smooth_same_curvature(edge, f1, f2, diag) -> bool:
    """Numeric fallback for non-analytic surfaces (revolutions, B-splines):
    True if the junction is tangent-continuous WITH matching transverse
    curvature -- i.e. two patches of one surface (STEP splits closed
    revolutions into halves; GT never draws those meridians). A genuine
    tangent edge (fillet boundary) keeps a curvature JUMP and stays drawn."""
    from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
    from OCC.Core.GeomLProp import GeomLProp_SLProps

    curve = BRepAdaptor_Curve(edge)
    u0, u1 = curve.FirstParameter(), curve.LastParameter()
    surf = []
    for f in (f1, f2):
        s = BRep_Tool.Surface(f)
        if s is None:
            return False
        surf.append(s)
    for frac in (0.28, 0.5, 0.77):
        u = u0 + (u1 - u0) * frac
        p = curve.Value(u)
        v = curve.DN(u, 1)
        if v.Magnitude() < 1e-12:
            continue
        tangent = np.array([v.X(), v.Y(), v.Z()])
        tangent /= np.linalg.norm(tangent)
        data = []
        for s in surf:
            proj = GeomAPI_ProjectPointOnSurf(p, s)
            if proj.NbPoints() == 0 or proj.LowerDistance() > 1e-4 * diag:
                return False
            uu, vv = proj.LowerDistanceParameters()
            props = GeomLProp_SLProps(s, uu, vv, 2, 1e-9)
            if not props.IsNormalDefined() or not props.IsCurvatureDefined():
                return False
            nrm = props.Normal()
            n = np.array([nrm.X(), nrm.Y(), nrm.Z()])
            d1, d2 = gp_Dir(), gp_Dir()
            props.CurvatureDirections(d1, d2)
            e1 = np.array([d1.X(), d1.Y(), d1.Z()])
            k1, k2 = props.MaxCurvature(), props.MinCurvature()
            # normal curvature transverse to the edge (Euler's theorem)
            w = np.cross(n, tangent)
            wn = np.linalg.norm(w)
            if wn < 1e-9:
                return False
            w /= wn
            ca = float(np.clip(w @ e1, -1.0, 1.0))
            kw = k1 * ca * ca + k2 * (1.0 - ca * ca)
            data.append((n, kw))
        (n_a, k_a), (n_b, k_b) = data
        if abs(float(n_a @ n_b)) < 1.0 - 1e-6:
            return False  # sharp edge
        # curvature sign flips with normal orientation; compare magnitudes
        # in a common frame
        if float(n_a @ n_b) < 0.0:
            k_b = -k_b
        scale = max(abs(k_a), abs(k_b), 0.05 / diag)
        if abs(k_a - k_b) > 0.02 * scale:
            return False  # tangent edge with curvature jump: draw it
    return True


def _drawable_edges(shape, tol, diag):
    """Real BRep edges worth ink: everything except degenerated edges,
    parameterisation seams (GT hides them) and same-surface split edges
    (analytic match, or numerically G2-continuous for splines/revolutions)."""
    emap = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, emap)
    edges = []
    for i in range(1, emap.Size() + 1):
        edge = topods.Edge(emap.FindKey(i))
        try:
            if BRep_Tool.Degenerated(edge):
                continue
            faces = [topods.Face(f) for f in emap.FindFromIndex(i)]
            if any(BRep_Tool.IsClosed(edge, f) for f in faces):
                continue  # seam
            if len(faces) == 2:
                if _same_surface_domain(faces[0], faces[1], tol):
                    continue
                if _smooth_same_curvature(edge, faces[0], faces[1], diag):
                    continue
        except Exception:  # noqa: BLE001 - malformed topology: keep the ink
            pass
        edges.append(edge)
    return edges


def _sample_edge_points(edge, camera, tf, px_step=1.2, max_pts=2400):
    """3D sample points along an edge, dense enough that consecutive samples
    are ~px_step pixels apart on the canvas."""
    curve = BRepAdaptor_Curve(edge)
    u0, u1 = curve.FirstParameter(), curve.LastParameter()
    coarse = np.array(
        [
            [curve.Value(u0 + (u1 - u0) * i / 15).Coord(j) for j in (1, 2, 3)]
            for i in range(16)
        ]
    )
    px = np.array([tf(x, y) for x, y in _plane_project(coarse, camera)])
    length_px = float(np.linalg.norm(np.diff(px, axis=0), axis=1).sum())
    n = int(np.clip(math.ceil(length_px / px_step) + 1, 2, max_pts))
    if n <= 16:
        return coarse[np.linspace(0, 15, n).round().astype(int)]
    us = np.linspace(u0, u1, n)
    return np.array([[curve.Value(u).Coord(j) for j in (1, 2, 3)] for u in us])


def _densify_polyline(
    pts3, camera: Camera, tf: "FitTransform", px_step=1.2, max_pts=4000
):
    """Insert linear subdivisions so consecutive samples are ~px_step px
    apart. Straight silhouette generators of cylinders/cones tessellate to a
    SINGLE axial segment -- 2 samples cannot support per-sample visibility
    classification (one noisy endpoint discards the whole line)."""
    pts3 = np.asarray(pts3, dtype=np.float64)
    px = np.array([tf(x, y) for x, y in _plane_project(pts3, camera)])
    seg_px = np.linalg.norm(np.diff(px, axis=0), axis=1)
    total = int(min(np.ceil(seg_px / px_step).sum() + len(pts3), max_pts))
    out = [pts3[0]]
    budget = max(total - len(pts3), 0)
    for i, L in enumerate(seg_px):
        n_sub = int(min(np.ceil(L / px_step) - 1, budget))
        if n_sub > 0:
            budget -= n_sub
            t = np.linspace(0.0, 1.0, n_sub + 2)[1:-1, None]
            out.extend(pts3[i] * (1.0 - t) + pts3[i + 1] * t)
        out.append(pts3[i + 1])
    return np.asarray(out)


def _plane_project(pts3, camera: Camera):
    """Perspective projection of (n,3) world points to camera-plane coords
    (the same x*f/(f+depth) mapping the exact HLR pass produced)."""
    d = np.asarray(pts3, dtype=np.float64) - np.asarray(camera.eye)
    dep = d @ np.asarray(camera.N)
    s = camera.focus / (camera.focus + dep)
    return np.stack(
        [(d @ np.asarray(camera.right)) * s, (d @ np.asarray(camera.up2d)) * s], axis=1
    )


def _silhouette_chains(verts, tris, eye):
    """Polyline chains (lists of vertex indices) along mesh edges where the
    facing of adjacent triangles flips -- the true outline/silhouette curves,
    perspective-correct w.r.t. the eye point."""
    v = verts[tris]
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    front = np.einsum("ij,ij->i", n, v.mean(axis=1) - np.asarray(eye)) < 0.0

    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    owner = np.tile(np.arange(len(tris)), 3)
    e.sort(axis=1)
    key = e[:, 0].astype(np.int64) * len(verts) + e[:, 1]
    order = np.argsort(key, kind="stable")
    ks, os_ = key[order], owner[order]
    starts = np.flatnonzero(np.concatenate([[True], ks[1:] != ks[:-1]]))
    counts = np.diff(np.concatenate([starts, [len(ks)]]))
    # interior mesh edges (2 owners) with opposite facing; _tessellate keeps
    # per-face vertex indexing, so these are silhouettes INSIDE smooth faces
    # (cross-face contours coincide with real BRep edges, drawn separately).
    pair = starts[counts == 2]
    flip = front[os_[pair]] != front[os_[pair + 1]]
    sil = ks[pair[flip]]
    sil_edges = np.stack(np.divmod(sil, len(verts)), axis=1).astype(np.int64)

    adj: dict[int, list[int]] = {}
    for i, (a, b) in enumerate(sil_edges):
        adj.setdefault(int(a), []).append(i)
        adj.setdefault(int(b), []).append(i)
    used = np.zeros(len(sil_edges), dtype=bool)
    chains = []
    for i in range(len(sil_edges)):
        if used[i]:
            continue
        used[i] = True
        a, b = (int(x) for x in sil_edges[i])
        chain = [a, b]
        for grow_end in (True, False):
            while True:
                tip = chain[-1] if grow_end else chain[0]
                nxt = next((j for j in adj.get(tip, ()) if not used[j]), None)
                if nxt is None:
                    break
                used[nxt] = True
                p, q = (int(x) for x in sil_edges[nxt])
                other = q if p == tip else p
                if grow_end:
                    chain.append(other)
                else:
                    chain.insert(0, other)
        chains.append(chain)
    return chains


def _depth_buffer(
    camera: Camera, cfg: Render3dConfig, tf: "FitTransform", verts, tris, supersample=2
):
    """Depth-from-eye buffer (supersampled canvas) rendered by VTK with the
    exact same camera model as the shaded pass. NaN where no geometry."""
    import pyvista as pv  # deferred, matches _render_shaded

    w, h = cfg.canvas_w * supersample, cfg.canvas_h * supersample
    n = len(tris)
    conn = np.hstack([np.full((n, 1), 3, dtype=np.int64), tris]).ravel()
    mesh = pv.PolyData(verts, conn)
    pl = pv.Plotter(off_screen=True, window_size=(w, h))
    pl.add_mesh(mesh, color="white", lighting=False, culling=False)
    cam = pl.camera
    cam.position = tuple(camera.eye[i] - camera.N[i] * camera.focus for i in range(3))
    cam.focal_point = camera.eye
    cam.up = camera.up2d
    half_h_world = cfg.canvas_h / (2.0 * tf.scale)
    cam.view_angle = 2.0 * math.degrees(math.atan(half_h_world / camera.focus))
    cam.SetWindowCenter(
        2.0 * tf.cx * tf.scale / cfg.canvas_w, 2.0 * tf.cy * tf.scale / cfg.canvas_h
    )
    # generous near plane: keeps 24-bit z-buffer precision ~1e-4 of the scene
    near = max(0.25 * (camera.focus + camera.D), camera.focus * 0.01)
    cam.clipping_range = (near, (camera.focus + camera.D) * 50.0)
    pl.screenshot(None, return_img=True)
    z = pl.get_image_depth(fill_value=np.nan)
    pl.close()
    dep = -np.asarray(z, dtype=np.float64) - camera.focus  # depth from the eye
    # 3x3 max filter with background = +inf: a silhouette sample sees only its
    # own NEARER surface inside the body (grazing incidence) plus background
    # outside -- background must certify visibility (nothing occludes the
    # outline), while a truly occluded line has its whole neighbourhood
    # covered by the nearer occluder and stays hidden.
    padded = np.full((dep.shape[0] + 2, dep.shape[1] + 2), np.inf)
    padded[1:-1, 1:-1] = np.where(np.isnan(dep), np.inf, dep)
    stack = [
        padded[dy : dy + dep.shape[0], dx : dx + dep.shape[1]]
        for dy in range(3)
        for dx in range(3)
    ]
    return np.maximum.reduce(stack)


def _make_ray_visible(camera: Camera, verts, tris, r_excl, pen_min):
    """Exact eye->point occlusion test against the mesh (vtkOBBTree).

    Used to arbitrate grazing-zone samples where the z-buffer is ill-posed.
    A hit only counts as an occluder when the ray actually dives deeper than
    ``pen_min`` into the body (signed distance at the midpoint of each
    entry/exit hit pair): tangency artifacts -- knife-edge double hits,
    terminal skims along the surface the sample sits on, chord-error
    grazes -- all have ~zero penetration, while a genuine occluder is
    crossed face-on.  Silhouette-chain samples are additionally pushed
    ``eps_out`` along the outward vertex normal before casting, because the
    chains are quantised to the angular tessellation step and self-occlude
    by a whisker (~R*dphi^2)."""
    import pyvista as pv
    import vtk
    from scipy.spatial import cKDTree

    n = len(tris)
    conn = np.hstack([np.full((n, 1), 3, dtype=np.int64), tris]).ravel()
    poly = pv.PolyData(verts, conn)
    obb = vtk.vtkOBBTree()
    obb.SetDataSet(poly)
    obb.BuildLocator()
    sdf = vtk.vtkImplicitPolyDataDistance()
    sdf.SetInput(poly)
    eye = np.asarray(camera.eye, dtype=np.float64)

    v = verts[tris]
    fn = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    vn = np.zeros_like(verts)
    for c in range(3):
        np.add.at(vn, tris[:, c], fn)
    norms = np.linalg.norm(vn, axis=1, keepdims=True)
    vn /= np.maximum(norms, 1e-30)
    tree = cKDTree(verts)

    def visible(samples, eps_out=0.0):
        samples = np.asarray(samples, dtype=np.float64)
        if eps_out > 0.0:
            _, idx = tree.query(samples)
            pushed = samples + vn[idx] * eps_out
        else:
            pushed = samples
        out = np.zeros(len(samples), dtype=bool)
        hits = vtk.vtkPoints()
        for k, p in enumerate(pushed):
            dist = float(np.linalg.norm(p - eye))
            ray = (p - eye) / dist
            obb.IntersectWithLine(eye.tolist(), p.tolist(), hits, None)
            ts = sorted(
                t
                for m in range(hits.GetNumberOfPoints())
                if (t := float(np.dot(np.asarray(hits.GetPoint(m)) - eye, ray)))
                < dist - r_excl
            )
            ok = True
            # entry/exit pairs (odd tail closes at the sample itself)
            for m in range(0, len(ts), 2):
                t_in = ts[m]
                t_out = ts[m + 1] if m + 1 < len(ts) else dist
                mid = eye + ray * (0.5 * (t_in + t_out))
                if -sdf.EvaluateFunction(mid.tolist()) > pen_min:
                    ok = False
                    break
            out[k] = ok
        return out

    return visible


def _classify_split(
    pts3,
    camera: Camera,
    tf: "FitTransform",
    depmax,
    supersample,
    tol,
    graze_recheck=None,
    graze_band=0.0,
):
    """Split one 3D polyline into visible / hidden runs by comparing each
    sample's depth against the (max-filtered) depth buffer.

    ``graze_recheck``: exact ray-visibility callback used to re-arbitrate
    samples the buffer calls hidden by no more than ``graze_band``.  A
    silhouette generator running nearly along the line of sight has its own
    surface skimming a few units in front of it over MANY pixels (depth error
    ~ sqrt(2*R*px)), so neither tol nor the 3x3 filter can rescue it -- but
    the exact tangent ray never crosses the body, while a genuinely occluded
    line has its ray blocked, so the ray test separates the two."""
    pts3 = np.asarray(pts3, dtype=np.float64)
    plane = _plane_project(pts3, camera)
    d = pts3 - np.asarray(camera.eye)
    dep = d @ np.asarray(camera.N)
    px = np.array([tf(x, y) for x, y in plane]) * supersample
    ix = np.clip(px[:, 0].astype(np.int64), 0, depmax.shape[1] - 1)
    iy = np.clip(px[:, 1].astype(np.int64), 0, depmax.shape[0] - 1)
    vis = dep <= depmax[iy, ix] + tol
    if graze_recheck is not None:
        cand = np.flatnonzero(~vis & (dep - depmax[iy, ix] <= graze_band))
        if cand.size:
            vis[cand] = graze_recheck(pts3[cand])
    if len(vis) >= 3:  # absorb single-sample flickers (z-buffer noise)
        v = vis.copy()
        v[1:-1] = np.where(vis[:-2] == vis[2:], vis[:-2], vis[1:-1])
        vis = v
    runs = []
    start = 0
    for i in range(1, len(vis) + 1):
        if i == len(vis) or vis[i] != vis[start]:
            if i - start >= 2:
                runs.append((bool(vis[start]), [tuple(p) for p in plane[start:i]]))
            elif runs and i < len(vis):  # 1-sample run: glue to the previous
                runs[-1][1].append(tuple(plane[i - 1]))
            start = i
    return runs


def _mesh_hlr_project(
    shape, camera: Camera, cfg: Render3dConfig, tf: "FitTransform", verts, tris
):
    """(visible, hidden) polylines in camera-plane coords, built from real
    edges + mesh silhouettes with depth-buffer visibility."""
    xmin, ymin, zmin, xmax, ymax, zmax = _bbox(shape)
    diag = max(
        math.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2), 1e-9
    )
    tol = 2e-3 * diag
    depmax = _depth_buffer(camera, cfg, tf, verts, tris)

    polylines3d = []
    for edge in _drawable_edges(shape, tol=1e-4 * diag, diag=diag):
        try:
            polylines3d.append((_sample_edge_points(edge, camera, tf), False))
        except Exception:  # noqa: BLE001 - skip broken edge geometry
            continue
    for chain in _silhouette_chains(verts, tris, camera.eye):
        pts3 = verts[np.asarray(chain, dtype=np.int64)]
        polylines3d.append((_densify_polyline(pts3, camera, tf), True))

    # grazing band: worst-case z-buffer depth error at a tangent contact,
    # sqrt(2*R*w) with R <= diag and w ~ 1.5px + margin of pixel footprint.
    # tf.scale is px per CAMERA-PLANE unit; world units at scene depth are
    # larger by the perspective factor (focus+D)/focus.
    wpp = (camera.focus + camera.D) / (camera.focus * tf.scale)
    graze_band = math.sqrt(4.5 * diag * wpp)
    # silhouette chains are quantised to the angular tessellation step, so
    # their samples self-occlude by a whisker (~R*dphi^2) and need the
    # outward push; exact BRep edge samples sit ON the true surface, above
    # the inscribed mesh, so no push is needed (and pushing would bleed
    # solid ink past true tangency transitions, e.g. far-cap arcs).
    ray_visible = _make_ray_visible(
        camera, verts, tris, r_excl=2.0 * wpp, pen_min=5.4e-4 * diag
    )
    eps_sil = 4e-3 * diag

    visible, hidden = [], []
    for pts3, is_sil in polylines3d:
        recheck = (lambda s: ray_visible(s, eps_sil)) if is_sil else ray_visible
        runs = _classify_split(
            pts3,
            camera,
            tf,
            depmax,
            2,
            tol,
            graze_recheck=recheck,
            graze_band=graze_band,
        )
        for is_vis, poly in runs:
            (visible if is_vis else hidden).append(poly)
    if not visible and not hidden:
        raise RuntimeError("mesh HLR produced no geometry")
    return visible, hidden


@dataclass
class FitTransform:
    scale: float
    cx: float
    cy: float
    canvas_w: int
    canvas_h: int

    def __call__(self, x, y):
        sx = self.canvas_w / 2 + (x - self.cx) * self.scale
        sy = self.canvas_h / 2 - (y - self.cy) * self.scale  # SVG/raster y-down
        return sx, sy


def _fit_transform(all_lines, canvas_w, canvas_h, margin_frac) -> FitTransform:
    """Fallback 2D bbox fit (used only when the sphere framing is unavailable)."""
    xs = [p[0] for line in all_lines for p in line]
    ys = [p[1] for line in all_lines for p in line]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    w = max(xmax - xmin, 1e-9)
    h = max(ymax - ymin, 1e-9)
    avail_w = canvas_w * (1 - 2 * margin_frac)
    avail_h = canvas_h * (1 - 2 * margin_frac)
    scale = min(avail_w / w, avail_h / h)
    return FitTransform(
        scale=scale,
        cx=(xmin + xmax) / 2,
        cy=(ymin + ymax) / 2,
        canvas_w=canvas_w,
        canvas_h=canvas_h,
    )


def _sphere_fit_transform(
    shape, camera: Camera, cfg: Render3dConfig, verts: "np.ndarray"
) -> FitTransform:
    """GT-style framing: the part's true bounding sphere (max vertex distance
    from the AABB centre) projects to a circle whose diameter fills
    ``cfg.fill_factor`` of the canvas height; the sphere centre projects to the
    canvas centre."""
    xmin, ymin, zmin, xmax, ymax, zmax = _bbox(shape)
    center = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2])
    radius = float(np.max(np.linalg.norm(verts - center, axis=1)))
    radius = max(radius, 1e-9)
    rel = center - np.array(camera.eye)
    z = float(rel @ np.array(camera.N))
    d_proj = 2.0 * radius * camera.focus / z
    cx = float(rel @ np.array(camera.right)) * camera.focus / z
    cy = float(rel @ np.array(camera.up2d)) * camera.focus / z
    scale = cfg.fill_factor * cfg.canvas_h / d_proj
    return FitTransform(
        scale=scale, cx=cx, cy=cy, canvas_w=cfg.canvas_w, canvas_h=cfg.canvas_h
    )


def _write_hlg_svg(
    visible,
    hidden,
    tf: FitTransform,
    cfg: Render3dConfig,
    background: str | None = "white",
) -> str:
    w, h = cfg.canvas_w, cfg.canvas_h
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
    ]
    if background:
        parts.append(f'<rect width="{w}" height="{h}" fill="{background}"/>')
    for line in hidden:
        pts = [tf(x, y) for x, y in line]
        d = "M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
        parts.append(
            f'<path d="{d}" fill="none" stroke="{cfg.hid_color}" '
            f'stroke-width="{cfg.hid_stroke_width}" stroke-dasharray="{cfg.hid_dash}"/>'
        )
    for line in visible:
        pts = [tf(x, y) for x, y in line]
        d = "M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
        parts.append(
            f'<path d="{d}" fill="none" stroke="black" '
            f'stroke-width="{cfg.vis_stroke_width}" stroke-linecap="round" stroke-linejoin="round"/>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def _write_alledges_svg(
    visible,
    hidden,
    tf: FitTransform,
    cfg: Render3dConfig,
    background: str | None = None,
) -> str:
    """All projected edges (visible AND hidden) drawn with the SAME solid
    style -- used to overlay edges on the translucent-shaded pass, where both
    near and far edges are visible through the translucency (no dashing)."""
    w, h = cfg.canvas_w, cfg.canvas_h
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
    ]
    if background:
        parts.append(f'<rect width="{w}" height="{h}" fill="{background}"/>')
    for line in visible + hidden:
        pts = [tf(x, y) for x, y in line]
        d = "M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
        parts.append(
            f'<path d="{d}" fill="none" stroke="{cfg.shaded_edge_color}" '
            f'stroke-width="{cfg.shaded_edge_width}" stroke-linecap="round" stroke-linejoin="round"/>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def _render_hlg(
    shape,
    camera: Camera,
    cfg: Render3dConfig,
    out_path: Path,
    tf: FitTransform | None = None,
    verts=None,
    tris=None,
):
    if tf is not None and verts is not None and tris is not None:
        visible, hidden = _mesh_hlr_project(shape, camera, cfg, tf, verts, tris)
    else:
        # Fallback (tessellation failed): legacy exact-HLR pass. Known defects
        # (phantom cone generators, tangency misclassification, no tangent
        # edges) accepted in this degraded mode.
        visible, hidden = _hlr_project(shape, camera, cfg)
        all_lines = visible + hidden
        if not all_lines:
            raise RuntimeError("HLR produced no geometry")
        if tf is None:
            tf = _fit_transform(all_lines, cfg.canvas_w, cfg.canvas_h, cfg.margin_frac)
    svg = _write_hlg_svg(visible, hidden, tf, cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cairosvg.svg2png(
        bytestring=svg.encode(),
        write_to=str(out_path),
        output_width=cfg.canvas_w,
        output_height=cfg.canvas_h,
    )
    return visible, hidden, tf


# ---------------------------------------------------------------------------
# Pass 2: tessellate + pyvista translucent shaded render (same camera/fit)
# ---------------------------------------------------------------------------


def _tessellate(shape, linear_deflection, angular_deflection=0.3):
    BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)
    verts = []
    tris = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = topods.Face(exp.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation(face, loc)
        if tri is not None:
            trsf = loc.Transformation()
            base = len(verts)
            for i in range(1, tri.NbNodes() + 1):
                p = tri.Node(i).Transformed(trsf)
                verts.append((p.X(), p.Y(), p.Z()))
            reversed_face = face.Orientation() == 1  # TopAbs_REVERSED
            for i in range(1, tri.NbTriangles() + 1):
                i1, i2, i3 = tri.Triangle(i).Get()
                if reversed_face:
                    i1, i2, i3 = i1, i3, i2
                tris.append((base + i1 - 1, base + i2 - 1, base + i3 - 1))
        exp.Next()
    if not tris:
        raise RuntimeError("tessellation produced no triangles")
    return np.asarray(verts, dtype=np.float64), np.asarray(tris, dtype=np.int64)


def _render_shaded(
    shape,
    camera: Camera,
    cfg: Render3dConfig,
    tf: FitTransform,
    out_path: Path,
    verts=None,
    tris=None,
):
    import pyvista as pv  # deferred: heavy import, keep off the module's import path

    if verts is None or tris is None:
        xmin, ymin, zmin, xmax, ymax, zmax = _bbox(shape)
        diag = math.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2)
        linear_defl = max(diag * cfg.deflection_frac * 3.0, 1e-6)
        verts, tris = _tessellate(shape, linear_defl)

    n = len(tris)
    conn = np.hstack([np.full((n, 1), 3, dtype=np.int64), tris]).ravel()
    mesh = pv.PolyData(verts, conn)

    pl = pv.Plotter(off_screen=True, window_size=(cfg.canvas_w, cfg.canvas_h))
    pl.set_background("white")
    # Faces only -- GT draws feature edges, not mesh wireframe; the caller
    # composites the HLR edge pass (same camera) on top.
    pl.add_mesh(
        mesh,
        color=cfg.shaded_color,
        opacity=cfg.shaded_opacity,
        show_edges=False,
        smooth_shading=True,
        split_sharp_edges=True,
        specular=0.15,
        ambient=cfg.shaded_ambient,
        diffuse=cfg.shaded_diffuse,
        culling=False,
    )

    # Derive the SAME projection as the HLR fit. The fit affine maps
    # camera-plane (x, y) [world units at the image plane, distance
    # `camera.focus` from the eye] to canvas pixels via `tf.scale`, with the
    # plane point (tf.cx, tf.cy) landing at the canvas centre. Aiming the VTK
    # camera at that off-axis point would ROTATE the optical axis (a different
    # projection than HLR's principal-point shift) -- instead keep the axis
    # exactly along camera.N and shift the principal point with WindowCenter
    # (an off-centre frustum, VTK's native equivalent).
    # OCC's perspective projector puts the image plane AT the eye with the
    # projection centre `focus` BEHIND it (x' = x*f/(f+depth)) -- verified by a
    # marker probe: HLR ink is uniformly 1/(1+f/D) smaller than a pinhole at
    # the eye. Replicate it exactly: pull the VTK camera back by `focus`.
    pinhole = tuple(camera.eye[i] - camera.N[i] * camera.focus for i in range(3))
    focal_point = camera.eye  # on-axis, one focal length ahead of the pinhole
    half_h_world = cfg.canvas_h / (2.0 * tf.scale)
    vertical_fov = (
        2.0
        * math.degrees(math.atan(half_h_world / camera.focus))
        * cfg.shaded_view_angle_pad
    )

    cam = pl.camera
    cam.position = pinhole
    cam.focal_point = focal_point
    cam.up = camera.up2d
    cam.view_angle = vertical_fov
    # Optical axis must land at pixel (W/2 - cx*s, H/2 + cy*s):
    # VTK maps camera-space x=0 to NDC -WindowCenter.
    cam.SetWindowCenter(
        2.0 * tf.cx * tf.scale / cfg.canvas_w, 2.0 * tf.cy * tf.scale / cfg.canvas_h
    )
    cam.clipping_range = (max(camera.focus * 0.01, 1e-4), camera.D * 50.0)

    arr = pl.screenshot(None, return_img=True)
    pl.close()
    return Image.fromarray(arr).convert("RGB")


# ---------------------------------------------------------------------------
# Pass 3: hlg_translucent_faces = hlg line art over a faint shaded fill
# ---------------------------------------------------------------------------


def _composite_translucent(
    shaded_img: Image.Image,
    visible,
    hidden,
    tf: FitTransform,
    cfg: Render3dConfig,
    out_path: Path,
):
    w, h = cfg.canvas_w, cfg.canvas_h
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base = shaded_img.convert("RGB").resize((w, h))
    white = Image.new("RGB", (w, h), "white")
    faint = Image.blend(white, base, cfg.translucent_face_opacity)

    svg = _write_hlg_svg(visible, hidden, tf, cfg, background=None)
    line_png_path = out_path.with_suffix(".lines.tmp.png")
    cairosvg.svg2png(
        bytestring=svg.encode(),
        write_to=str(line_png_path),
        output_width=w,
        output_height=h,
    )
    lines = Image.open(line_png_path).convert("RGBA")
    line_png_path.unlink(missing_ok=True)

    composed = faint.convert("RGBA")
    composed.alpha_composite(lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    composed.convert("RGB").save(out_path)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_render3d(
    step_path: Path, paths: Render3dPaths, cfg: Render3dConfig | None = None
) -> dict:
    step_path = Path(step_path)
    cfg = cfg or Render3dConfig()
    info: dict = {
        "hlg_ok": False,
        "shaded_ok": False,
        "hlg_translucent_ok": False,
        "errors": {},
    }

    shape = _load_shape(step_path)
    camera = _build_camera(shape, cfg)

    # Tessellate once: the mesh drives the GT-style sphere framing, the mesh
    # hidden-line pass (silhouettes + depth buffer) and the shaded pass.
    # Fine deflection: ~0.2 px chord error on the canvas regardless of part
    # size (px_err ~ defl * fill * canvas_h / diag).
    verts = tris = None
    sphere_tf = None
    try:
        xmin, ymin, zmin, xmax, ymax, zmax = _bbox(shape)
        diag = math.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2)
        verts, tris = _tessellate(
            shape, max(diag * 2e-4, 1e-6), angular_deflection=0.15
        )
        sphere_tf = _sphere_fit_transform(shape, camera, cfg, verts)
    except Exception as exc:  # noqa: BLE001
        info["errors"]["tessellation"] = f"{type(exc).__name__}: {exc}"

    visible = hidden = tf = None
    try:
        visible, hidden, tf = _render_hlg(
            shape, camera, cfg, Path(paths.hlg), tf=sphere_tf, verts=verts, tris=tris
        )
        info["hlg_ok"] = True
    except Exception as exc:  # noqa: BLE001
        info["errors"]["hlg"] = f"{type(exc).__name__}: {exc}"

    shaded_img = None
    if tf is not None and verts is not None:
        try:
            shaded_img = _render_shaded(
                shape, camera, cfg, tf, Path(paths.shaded), verts=verts, tris=tris
            )
            # GT's transparent_shaded_edges draws feature edges (near AND far,
            # solid, seen through the translucency) -- composite the HLR edge
            # pass, which shares the camera and fit, over the face render.
            edges_svg = _write_alledges_svg(visible, hidden, tf, cfg)
            edges_png = cairosvg.svg2png(
                bytestring=edges_svg.encode(),
                output_width=cfg.canvas_w,
                output_height=cfg.canvas_h,
            )
            import io

            edges = Image.open(io.BytesIO(edges_png)).convert("RGBA")
            composed = shaded_img.convert("RGBA")
            composed.alpha_composite(edges)
            out = Path(paths.shaded)
            out.parent.mkdir(parents=True, exist_ok=True)
            composed.convert("RGB").save(out)
            info["shaded_ok"] = True
        except Exception as exc:  # noqa: BLE001
            info["errors"]["shaded"] = f"{type(exc).__name__}: {exc}"

    if tf is not None:
        try:
            if shaded_img is not None:
                _composite_translucent(
                    shaded_img, visible, hidden, tf, cfg, Path(paths.hlg_translucent)
                )
            else:
                # shaded pass failed: fall back to the plain line art so the
                # style still exists (documented degraded output).
                Path(paths.hlg_translucent).parent.mkdir(parents=True, exist_ok=True)
                svg = _write_hlg_svg(visible, hidden, tf, cfg)
                cairosvg.svg2png(
                    bytestring=svg.encode(),
                    write_to=str(paths.hlg_translucent),
                    output_width=cfg.canvas_w,
                    output_height=cfg.canvas_h,
                )
            info["hlg_translucent_ok"] = True
        except Exception as exc:  # noqa: BLE001
            info["errors"]["hlg_translucent"] = f"{type(exc).__name__}: {exc}"

    return info


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import sys

    from src.render.config import render3d_paths

    step = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/tmp/render3d_out")
    rp = render3d_paths(out, step.stem)
    print(generate_render3d(step, rp))
