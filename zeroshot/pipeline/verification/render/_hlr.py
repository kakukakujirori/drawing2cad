"""Verbatim fork of src/data/render/hlr.py @ e23cc7f -- do not edit; any change
must still reproduce the baseline in outputs/render_golden.

OCC HLR projection -> typed 2D primitives.

Given a shape and a view frame (origin + view direction + up direction) this
runs the OCC hidden-line-removal algorithm and returns the projected edges as
*typed* 2D primitives (segments, arcs, circles, ellipses, polylines from
b-splines).  Coordinates are in the projection plane frame: screen ``x`` is the
frame X direction, screen ``y`` is the frame Y direction, both in model units.

The projection is orthographic.  The projected edges returned by
``HLRBRep_HLRToShape`` live in the projector coordinate system, so their
(X, Y) already are screen coordinates and Z is depth (ignored).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GCPnts import GCPnts_QuasiUniformDeflection
from OCC.Core.GeomAbs import (
    GeomAbs_Circle,
    GeomAbs_Ellipse,
    GeomAbs_Line,
)
from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt
from OCC.Core.HLRAlgo import HLRAlgo_Projector
from OCC.Core.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape
from OCC.Core.TopAbs import TopAbs_EDGE
from OCC.Core.TopExp import TopExp_Explorer

# --------------------------------------------------------------------------
# Typed 2D primitives (screen coordinates, model units)
# --------------------------------------------------------------------------


@dataclass
class Segment:
    p0: tuple[float, float]
    p1: tuple[float, float]


@dataclass
class Arc:
    center: tuple[float, float]
    radius: float
    a0: float  # start angle (rad, CCW from +x)
    a1: float  # end angle (rad)
    ccw: bool = True


@dataclass
class Circle:
    center: tuple[float, float]
    radius: float


@dataclass
class Ellipse:
    center: tuple[float, float]
    rmaj: float
    rmin: float
    rot: float  # major-axis angle (rad)
    a0: float = 0.0
    a1: float = 2.0 * math.pi


@dataclass
class Polyline:
    pts: list[tuple[float, float]]


@dataclass
class ProjectedEdges:
    """Typed primitives for one edge class (visible or hidden)."""

    segments: list[Segment] = field(default_factory=list)
    arcs: list[Arc] = field(default_factory=list)
    circles: list[Circle] = field(default_factory=list)
    ellipses: list[Ellipse] = field(default_factory=list)
    polylines: list[Polyline] = field(default_factory=list)

    def count(self) -> int:
        return (
            len(self.segments)
            + len(self.arcs)
            + len(self.circles)
            + len(self.ellipses)
            + len(self.polylines)
        )


@dataclass
class ViewProjection:
    """Result of projecting a shape for one orthographic view."""

    visible: ProjectedEdges
    hidden: ProjectedEdges

    def bbox(self, include_hidden: bool = True) -> tuple[float, float, float, float]:
        xs: list[float] = []
        ys: list[float] = []
        groups = [self.visible] + ([self.hidden] if include_hidden else [])
        for g in groups:
            for s in g.segments:
                xs += [s.p0[0], s.p1[0]]
                ys += [s.p0[1], s.p1[1]]
            for a in g.arcs:
                # sample the arc for a tight bbox
                for x, y in _sample_arc(a.center, a.radius, a.a0, a.a1, a.ccw, 12):
                    xs.append(x)
                    ys.append(y)
            for c in g.circles:
                xs += [c.center[0] - c.radius, c.center[0] + c.radius]
                ys += [c.center[1] - c.radius, c.center[1] + c.radius]
            for e in g.ellipses:
                r = max(e.rmaj, e.rmin)
                xs += [e.center[0] - r, e.center[0] + r]
                ys += [e.center[1] - r, e.center[1] + r]
            for pl in g.polylines:
                for x, y in pl.pts:
                    xs.append(x)
                    ys.append(y)
        if not xs:
            return (0.0, 0.0, 0.0, 0.0)
        return (min(xs), min(ys), max(xs), max(ys))


def _sample_arc(center, radius, a0, a1, ccw, n):
    if not ccw:
        a0, a1 = a1, a0
    if a1 < a0:
        a1 += 2.0 * math.pi
    pts = []
    for i in range(n + 1):
        a = a0 + (a1 - a0) * i / n
        pts.append((center[0] + radius * math.cos(a), center[1] + radius * math.sin(a)))
    return pts


# --------------------------------------------------------------------------
# Projector construction
# --------------------------------------------------------------------------


def make_projector(
    view_dir: tuple[float, float, float], up_dir: tuple[float, float, float]
) -> HLRAlgo_Projector:
    """Orthographic projector.

    ``view_dir`` points from the model toward the viewer (the plane normal).
    Screen +Y follows ``up_dir``; screen +X = up x view (right-handed screen).
    """
    zdir = gp_Dir(*view_dir)
    ydir = gp_Dir(*up_dir)
    # x = y cross z  gives a right-handed (x,y,z) screen frame with z toward viewer
    ax2 = gp_Ax2(gp_Pnt(0, 0, 0), zdir)
    ax2.SetYDirection(ydir)
    return HLRAlgo_Projector(ax2)


def _footprint_bound(shape, view_dir, up_dir, margin_frac: float = 0.01):
    """Projected-AABB bound (xmin, xmax, ymin, ymax) in the view frame.

    Every real edge / outline lies ON the shape, so its projection must fall
    inside the projected model AABB; OCC's exact HLR occasionally emits
    runaway outline curves in tangency zones (out-and-back b-splines whose
    interior shoots thousands of units away) and those are the only curves
    that can exit this bound.
    """
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib

    b = Bnd_Box()
    brepbndlib.Add(shape, b)
    xmin, ymin, zmin, xmax, ymax, zmax = b.Get()
    ax2 = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(*view_dir))
    ax2.SetYDirection(gp_Dir(*up_dir))
    xd = ax2.XDirection()
    yd = ax2.YDirection()
    us, vs = [], []
    for cx in (xmin, xmax):
        for cy in (ymin, ymax):
            for cz in (zmin, zmax):
                us.append(cx * xd.X() + cy * xd.Y() + cz * xd.Z())
                vs.append(cx * yd.X() + cy * yd.Y() + cz * yd.Z())
    diag = math.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2)
    m = max(margin_frac * diag, 1e-9)
    return min(us) - m, max(us) + m, min(vs) - m, max(vs) + m


def run_hlr(shape, projector: HLRAlgo_Projector) -> HLRBRep_HLRToShape:
    algo = HLRBRep_Algo()
    algo.Add(shape)
    algo.Projector(projector)
    algo.Update()
    algo.Hide()
    return HLRBRep_HLRToShape(algo)


# --------------------------------------------------------------------------
# Typed curve extraction
# --------------------------------------------------------------------------

_TWO_PI = 2.0 * math.pi


def _norm_angle(a: float) -> float:
    a = a % _TWO_PI
    if a < 0:
        a += _TWO_PI
    return a


def _degenerate_sliver(pts) -> bool:
    """Closed, near-zero-area (out-and-back) curve: an OCC tangency-zone
    outline artifact.  Real outlines traverse one way or enclose area."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ext = max(max(xs) - min(xs), max(ys) - min(ys))
    if ext <= 1e-9:
        return True
    endgap = math.hypot(xs[-1] - xs[0], ys[-1] - ys[0])
    if endgap >= 1e-2 * ext:
        return False
    area = 0.5 * abs(
        sum(xs[k] * ys[k + 1] - xs[k + 1] * ys[k] for k in range(len(pts) - 1))
    )
    return area < 0.01 * ext * ext


def _classify_edge(
    edge, out: ProjectedEdges, deflection: float, bound=None, is_outline: bool = False
) -> None:
    ad = BRepAdaptor_Curve(edge)
    t = ad.GetType()
    u0 = ad.FirstParameter()
    u1 = ad.LastParameter()

    if bound is not None:
        bx0, bx1, by0, by1 = bound
        for i in range(25):
            p = ad.Value(u0 + (u1 - u0) * i / 24)
            if not (bx0 <= p.X() <= bx1 and by0 <= p.Y() <= by1):
                return  # runaway HLR outline artifact: drop the whole edge

    if t == GeomAbs_Line:
        p0 = ad.Value(u0)
        p1 = ad.Value(u1)
        out.segments.append(Segment((p0.X(), p0.Y()), (p1.X(), p1.Y())))
        return

    if t == GeomAbs_Circle:
        circ = ad.Circle()
        c = circ.Location()
        r = circ.Radius()
        # angle of the circle's own X axis in the screen frame
        xax = circ.XAxis().Direction()
        base = math.atan2(xax.Y(), xax.X())
        # sense: circle axis Z vs screen +Z
        zsign = 1.0 if circ.Axis().Direction().Z() >= 0 else -1.0
        span = u1 - u0
        full = abs(span) >= _TWO_PI - 1e-6
        if full:
            out.circles.append(Circle((c.X(), c.Y()), r))
        else:
            if zsign >= 0:
                a0 = base + u0
                a1 = base + u1
                ccw = True
            else:
                a0 = base - u0
                a1 = base - u1
                ccw = False
            out.arcs.append(Arc((c.X(), c.Y()), r, a0, a1, ccw))
        return

    if t == GeomAbs_Ellipse:
        el = ad.Ellipse()
        c = el.Location()
        rmaj = el.MajorRadius()
        rmin = el.MinorRadius()
        xax = el.XAxis().Direction()
        rot = math.atan2(xax.Y(), xax.X())
        zsign = 1.0 if el.Axis().Direction().Z() >= 0 else -1.0
        span = u1 - u0
        full = abs(span) >= _TWO_PI - 1e-6
        if full:
            out.ellipses.append(Ellipse((c.X(), c.Y()), rmaj, rmin, rot))
        else:
            if zsign >= 0:
                a0, a1 = u0, u1
            else:
                a0, a1 = -u0, -u1
            out.ellipses.append(Ellipse((c.X(), c.Y()), rmaj, rmin, rot, a0, a1))
        return

    # b-spline / bezier / other -> discretize into a polyline
    pts: list[tuple[float, float]] = []
    try:
        disc = GCPnts_QuasiUniformDeflection(ad, deflection)
        if disc.IsDone() and disc.NbPoints() >= 2:
            for i in range(1, disc.NbPoints() + 1):
                p = disc.Value(i)
                pts.append((p.X(), p.Y()))
    except Exception:  # noqa: BLE001
        pts = []
    if len(pts) < 2:
        n = 32
        for i in range(n + 1):
            p = ad.Value(u0 + (u1 - u0) * i / n)
            pts.append((p.X(), p.Y()))
    if is_outline and _degenerate_sliver(pts):
        return
    _emit_curve(pts, out)


def _closed(pts, tol_ratio: float = 1e-3) -> bool:
    x0, y0 = pts[0]
    x1, y1 = pts[-1]
    d = math.hypot(x1 - x0, y1 - y0)
    # length scale from the polyline extent
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    scale = max(max(xs) - min(xs), max(ys) - min(ys), 1e-9)
    return d <= tol_ratio * scale


def _emit_curve(pts, out: ProjectedEdges) -> None:
    """Classify a sampled curve into Circle / Arc / Segment / Polyline.

    OCC HLR returns projected circular edges (and curved silhouettes) as
    b-splines; recognising circles/arcs restores smooth output and centre marks.
    """
    closed = _closed(pts)
    if not closed and _is_straight(pts):
        out.segments.append(Segment(pts[0], pts[-1]))
        return
    fit = _fit_circle(pts)
    if fit is not None:
        cx, cy, r, resid = fit
        # r must be commensurate with the data extent: a near-degenerate
        # (collinear / out-and-back) curve fits a quasi-infinite circle whose
        # RELATIVE residual resid/r vanishes as r grows, so the ratio test
        # alone accepts garbage.  Real arcs shallow enough to reach here have
        # r <= ~12.5x chord (_is_straight catches flatter ones).
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        extent = max(max(xs) - min(xs), max(ys) - min(ys), 1e-9)
        if r > 1e-9 and resid / r < 0.03 and r <= 25.0 * extent:
            if closed:
                out.circles.append(Circle((cx, cy), r))
            else:
                a0 = math.atan2(pts[0][1] - cy, pts[0][0] - cx)
                a1 = math.atan2(pts[-1][1] - cy, pts[-1][0] - cx)
                mid = pts[len(pts) // 2]
                # sense from the cross product (start->mid)
                cross = (mid[0] - cx) * (pts[0][1] - cy) - (mid[1] - cy) * (
                    pts[0][0] - cx
                )
                ccw = cross < 0
                out.arcs.append(Arc((cx, cy), r, a0, a1, ccw))
            return
    out.polylines.append(Polyline(pts))


def _is_straight(pts, rel_tol: float = 0.01) -> bool:
    x0, y0 = pts[0]
    x1, y1 = pts[-1]
    L = math.hypot(x1 - x0, y1 - y0)
    if L < 1e-9:
        return False  # closed / degenerate -> not a straight segment
    maxd = 0.0
    for x, y in pts:
        d = abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0) / L
        maxd = max(maxd, d)
    return (maxd / L) < rel_tol


def _fit_circle(pts):
    """Kasa algebraic circle fit; returns (cx, cy, r, max_residual) or None."""
    n = len(pts)
    if n < 4:
        return None
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    u = [p[0] - mx for p in pts]
    v = [p[1] - my for p in pts]
    Suu = sum(ui * ui for ui in u)
    Svv = sum(vi * vi for vi in v)
    Suv = sum(ui * vi for ui, vi in zip(u, v))
    Suuu = sum(ui**3 for ui in u)
    Svvv = sum(vi**3 for vi in v)
    Suvv = sum(ui * vi * vi for ui, vi in zip(u, v))
    Svuu = sum(vi * ui * ui for ui, vi in zip(u, v))
    det = Suu * Svv - Suv * Suv
    if abs(det) < 1e-20:
        return None
    b1 = 0.5 * (Suuu + Suvv)
    b2 = 0.5 * (Svvv + Svuu)
    uc = (b1 * Svv - b2 * Suv) / det
    vc = (Suu * b2 - Suv * b1) / det
    cx, cy = uc + mx, vc + my
    r = math.sqrt(max(uc * uc + vc * vc + (Suu + Svv) / n, 0.0))
    resid = max(abs(math.hypot(p[0] - cx, p[1] - cy) - r) for p in pts)
    return cx, cy, r, resid


def _extract_compound(
    comp, out: ProjectedEdges, deflection: float, bound=None, is_outline: bool = False
) -> None:
    if comp is None or comp.IsNull():
        return
    exp = TopExp_Explorer(comp, TopAbs_EDGE)
    while exp.More():
        _classify_edge(exp.Current(), out, deflection, bound, is_outline)
        exp.Next()


# HLR compound accessor names, grouped by visible / hidden.
_VISIBLE_ACCESSORS = ("VCompound", "OutLineVCompound")
_VISIBLE_SMOOTH = ("Rg1LineVCompound",)
_HIDDEN_ACCESSORS = ("HCompound", "OutLineHCompound")
_HIDDEN_SMOOTH = ("Rg1LineHCompound",)


def _safe_compound(hts: HLRBRep_HLRToShape, name: str):
    try:
        return getattr(hts, name)()
    except Exception:  # noqa: BLE001
        return None


def _recombine_circles(edges: ProjectedEdges, tol: float) -> None:
    """Coalesce arcs on the same circle.

    OCC HLR commonly splits one topological circular edge at visibility or
    face-boundary points.  Keeping those fragments independently is not only
    noisy: clockwise fragments can be written with negative / >360 degree DXF
    angles, and some CAD readers then display the almost-closed arc as a full
    circle.  Union the fragments geometrically, preserving a real angular gap,
    and promote to ``Circle`` only when the residual gap is within the model
    tolerance.
    """
    if not edges.arcs:
        return
    groups: dict = {}
    key_scale = 1.0 / max(tol, 1e-9)
    for a in edges.arcs:
        k = (
            round(a.center[0] * key_scale),
            round(a.center[1] * key_scale),
            round(a.radius * key_scale),
        )
        groups.setdefault(k, []).append(a)
    coalesced: list[Arc] = []
    for arcs in groups.values():
        c = arcs[0]
        # Angular tolerance implied by the linear merge tolerance.  Cap it so
        # a deliberate slot in an almost-complete circle is never filled.
        angle_tol = max(1e-7, min(0.02, 2.0 * tol / max(c.radius, tol)))
        intervals: list[tuple[float, float]] = []
        for a in arcs:
            s = _norm_angle(a.a0)
            if a.ccw:
                span = (a.a1 - a.a0) % _TWO_PI
            else:
                span = (a.a0 - a.a1) % _TWO_PI
                s = _norm_angle(a.a1)
            if span < 1e-9:
                span = _TWO_PI
            e = s + span
            if e > _TWO_PI:
                intervals.append((s, _TWO_PI))
                intervals.append((0.0, e - _TWO_PI))
            else:
                intervals.append((s, e))

        intervals.sort()
        merged: list[list[float]] = []
        for s, e in intervals:
            if merged and s <= merged[-1][1] + angle_tol:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])

        coverage = sum(e - s for s, e in merged)
        if coverage >= _TWO_PI - angle_tol:
            edges.circles.append(Circle(c.center, c.radius))
            continue

        # Join the intervals on either side of angle zero into one wrapping
        # ARC.  DXF arcs are CCW, so start > end is the canonical representation
        # of an arc crossing zero (e.g. 15deg -> 1deg is almost a full circle).
        if (
            len(merged) >= 2
            and merged[0][0] <= angle_tol
            and merged[-1][1] >= _TWO_PI - angle_tol
        ):
            wrap = (merged[-1][0], merged[0][1])
            merged = merged[1:-1]
            coalesced.append(Arc(c.center, c.radius, *wrap, ccw=True))
        for s, e in merged:
            coalesced.append(
                Arc(c.center, c.radius, _norm_angle(s), _norm_angle(e), ccw=True)
            )
    edges.arcs = coalesced


def _drop_hidden_duplicates(
    visible: ProjectedEdges, hidden: ProjectedEdges, tol: float
) -> None:
    """Prefer visible ink when OCC emits the same projected edge twice.

    A shared BRep edge can be returned by both ``VCompound`` and ``HCompound``
    when one adjacent face is visible and the other is hidden.  SolidWorks
    emits it once as visible; retaining both creates doubled DXF primitives and
    can make a slotted arc look closed in downstream CAD viewers.
    """

    def point_close(p, q) -> bool:
        return math.hypot(p[0] - q[0], p[1] - q[1]) <= tol

    def same_segment(a: Segment, b: Segment) -> bool:
        return (point_close(a.p0, b.p0) and point_close(a.p1, b.p1)) or (
            point_close(a.p0, b.p1) and point_close(a.p1, b.p0)
        )

    def arc_signature(a: Arc) -> tuple[float, float]:
        if a.ccw:
            start = _norm_angle(a.a0)
            span = (a.a1 - a.a0) % _TWO_PI
        else:
            start = _norm_angle(a.a1)
            span = (a.a0 - a.a1) % _TWO_PI
        return start, span

    def angle_close(a: float, b: float, angle_tol: float) -> bool:
        return abs((a - b + math.pi) % _TWO_PI - math.pi) <= angle_tol

    def same_arc(a: Arc, b: Arc) -> bool:
        if not point_close(a.center, b.center) or abs(a.radius - b.radius) > tol:
            return False
        angle_tol = max(1e-7, min(0.02, 2.0 * tol / max(a.radius, b.radius, tol)))
        a_start, a_span = arc_signature(a)
        b_start, b_span = arc_signature(b)
        return angle_close(a_start, b_start, angle_tol) and abs(a_span - b_span) <= (
            2.0 * angle_tol
        )

    def same_circle(a: Circle, b: Circle) -> bool:
        return point_close(a.center, b.center) and abs(a.radius - b.radius) <= tol

    def same_polyline(a: Polyline, b: Polyline) -> bool:
        if len(a.pts) != len(b.pts):
            return False
        return all(point_close(p, q) for p, q in zip(a.pts, b.pts)) or all(
            point_close(p, q) for p, q in zip(a.pts, reversed(b.pts))
        )

    hidden.segments = [
        h
        for h in hidden.segments
        if not any(same_segment(h, v) for v in visible.segments)
    ]
    hidden.arcs = [
        h for h in hidden.arcs if not any(same_arc(h, v) for v in visible.arcs)
    ]
    hidden.circles = [
        h for h in hidden.circles if not any(same_circle(h, v) for v in visible.circles)
    ]
    hidden.polylines = [
        h
        for h in hidden.polylines
        if not any(same_polyline(h, v) for v in visible.polylines)
    ]


def _merge_segments(edges: ProjectedEdges, tol: float) -> None:
    """Deduplicate and merge collinear, overlapping/touching segments so the
    hidden/visible line counts approach GT (SolidWorks merges such runs)."""
    segs = edges.segments
    if len(segs) < 2:
        return
    # bucket by infinite-line signature (angle mod pi, signed distance to origin)
    buckets: dict = {}
    for s in segs:
        (x0, y0), (x1, y1) = s.p0, s.p1
        dx, dy = x1 - x0, y1 - y0
        L = math.hypot(dx, dy)
        if L < 1e-9:
            continue
        ang = math.atan2(dy, dx) % math.pi
        ux, uy = math.cos(ang), math.sin(ang)
        # perpendicular offset of the line from origin
        off = -uy * x0 + ux * y0
        k = (round(ang / (tol * 2)), round(off / tol))
        buckets.setdefault(k, (ux, uy, []))[2].append(s)
    merged: list[Segment] = []
    for ux, uy, group in buckets.values():
        # project endpoints onto the line direction, merge overlapping ranges
        ivals = []
        for s in group:
            t0 = s.p0[0] * ux + s.p0[1] * uy
            t1 = s.p1[0] * ux + s.p1[1] * uy
            ivals.append((min(t0, t1), max(t0, t1)))
        ivals.sort()
        base = group[0].p0
        b_t = base[0] * ux + base[1] * uy
        px = base[0] - b_t * ux
        py = base[1] - b_t * uy
        cur_s, cur_e = ivals[0]
        out_ivals = []
        for s, e in ivals[1:]:
            if s <= cur_e + tol:
                cur_e = max(cur_e, e)
            else:
                out_ivals.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        out_ivals.append((cur_s, cur_e))
        for s, e in out_ivals:
            merged.append(
                Segment((px + ux * s, py + uy * s), (px + ux * e, py + uy * e))
            )
    edges.segments = merged


def _drop_degenerate(edges: ProjectedEdges, tol: float) -> None:
    """Remove sub-tolerance fragments (exact HLR emits them at tangency
    endpoints); they survive rounding as zero-length LINE / LWPOLYLINE
    entities in the DXF output."""
    edges.segments = [
        s
        for s in edges.segments
        if math.hypot(s.p1[0] - s.p0[0], s.p1[1] - s.p0[1]) >= tol
    ]

    def _ext(pts):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return max(max(xs) - min(xs), max(ys) - min(ys))

    edges.polylines = [p for p in edges.polylines if _ext(p.pts) >= tol]
    edges.arcs = [a for a in edges.arcs if a.radius >= tol]
    edges.circles = [c for c in edges.circles if c.radius >= 0.5 * tol]
    edges.ellipses = [e for e in edges.ellipses if e.rmaj >= 0.5 * tol]


def _cleanup(edges: ProjectedEdges, tol: float) -> None:
    _drop_degenerate(edges, tol)
    _recombine_circles(edges, tol)
    _merge_segments(edges, tol)


def project(
    shape,
    view_dir: tuple[float, float, float],
    up_dir: tuple[float, float, float],
    deflection: float = 0.05,
    include_smooth: bool = False,
    merge_tol: float = 1e-4,
) -> ViewProjection:
    """Project ``shape`` and return typed visible / hidden primitives."""
    projector = make_projector(view_dir, up_dir)
    hts = run_hlr(shape, projector)
    bound = _footprint_bound(shape, view_dir, up_dir)

    visible = ProjectedEdges()
    hidden = ProjectedEdges()

    vis_names = list(_VISIBLE_ACCESSORS) + (
        list(_VISIBLE_SMOOTH) if include_smooth else []
    )
    hid_names = list(_HIDDEN_ACCESSORS) + (
        list(_HIDDEN_SMOOTH) if include_smooth else []
    )

    for name in vis_names:
        _extract_compound(
            _safe_compound(hts, name),
            visible,
            deflection,
            bound,
            is_outline="OutLine" in name,
        )
    for name in hid_names:
        _extract_compound(
            _safe_compound(hts, name),
            hidden,
            deflection,
            bound,
            is_outline="OutLine" in name,
        )

    _cleanup(visible, merge_tol)
    _cleanup(hidden, merge_tol)
    _drop_hidden_duplicates(visible, hidden, merge_tol)

    return ViewProjection(visible=visible, hidden=hidden)
