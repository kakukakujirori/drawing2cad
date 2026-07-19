try:
    import freecad
except ImportError:
    pass
import math
import FreeCAD as App
from typing import Any
from .base import BaseProjector
from .ellipse import ellipse_from_conjugate


class ConeProjector(BaseProjector):
    """Projects a conical face (Part::GeomCone) — plain cone, frustum, countersink.

    Analogous to CylinderProjector but the silhouette generators are not at a
    fixed angle from the view: for a cone of semi-angle alpha, axis a, apex A,
    a surface point P is on the silhouette iff the view direction V lies in the
    tangent plane, i.e. (with V decomposed in a base-plane frame (u, w, a)):

        Vu*cos(theta) + Vw*sin(theta) = tan(alpha) * Va

    which has solutions theta = phi +- acos(tan(alpha)*Va/R), phi = atan2(Vw, Vu),
    R = hypot(Vu, Vw). Verified against TechDraw.projectEx for plain / frustum /
    tilted / countersunk cones (see __main__).

    Emitted primitives (existing kinds only):
      - silhouette generators -> line, role="silhouette"
      - circular boundary rings -> circle (axis-aligned view), analytic ellipse
        via the conjugate-diameter math (oblique view), or line (edge-on view),
        role="boundary"
    Degeneracies: axis || view -> no silhouette (R ~ 0); view inside the cone
    half-angle (|rhs| > 1) -> no real silhouette; ringless faces -> silhouette
    skipped (provenance-only loss, HLR still draws the geometry).
    """

    def get_supported_types(self) -> list[str]:
        return ["Part::GeomCone"]

    def _on_face(self, pnt: App.Vector, tol: float = 1e-5) -> bool:
        # same gating as CylinderProjector: a partial conical face (half-bore,
        # split countersink) may not reach the silhouette angle at all
        try:
            return self.shape.isInside(pnt, tol, True)
        except Exception:
            try:
                import Part as _Part

                vtx = _Part.Vertex(pnt)
                return self.shape.distToShape(vtx)[0] < tol
            except Exception:
                return True

    def _rings(self, axis: App.Vector):
        """Unique circular boundary edges of this face, sorted along the axis.
        Returns [(center, radius)]. Edges whose Curve cannot be read (FreeCAD
        raises TypeError 'undefined curve type' from the C++ layer) are skipped."""
        rings = []
        seen = set()
        for e in self.shape.Edges:
            try:
                c = e.Curve
                if c.TypeId != "Part::GeomCircle":
                    continue
                cen, r = c.Center, c.Radius
            except Exception:
                continue
            if r < 1e-9:
                continue  # the apex of a full cone is a degenerate r=0 circle
            key = (round(cen.dot(axis), 6), round(r, 6))
            if key in seen:
                continue
            seen.add(key)
            rings.append((cen, r))
        rings.sort(key=lambda cr: cr[0].dot(axis))
        return rings

    def _project_ring(
        self,
        C: App.Vector,
        r: float,
        axis: App.Vector,
        view_direction: tuple[float, float, float],
    ) -> dict[str, Any] | None:
        """One circular ring (center C, radius r, plane normal = axis) -> 2D conic."""
        dir_vec = App.Vector(*view_direction)
        d = axis.dot(dir_vec)
        if abs(d) < 1e-6:  # edge-on -> line segment (the ring's diameter)
            ldir = axis.cross(dir_vec)
            ldir.normalize()
            p1 = self._format_pt(self._project_point(C + ldir * r, view_direction))
            p2 = self._format_pt(self._project_point(C - ldir * r, view_direction))
            if p1 == p2:
                return None
            return {"type": "line", "p1": p1, "p2": p2, "role": "boundary"}
        if abs(abs(d) - 1.0) < 1e-6:  # face-on -> true circle
            c2 = self._format_pt(self._project_point(C, view_direction))
            return {"type": "circle", "center": c2, "radius": r, "role": "boundary"}
        # oblique -> analytic ellipse from conjugate semi-diameters (same math
        # as CircleProjector / EllipseProjector)
        u = axis.cross(dir_vec)
        u.normalize()
        w = axis.cross(u)
        w.normalize()
        o2 = self._project_point(C, view_direction)
        pu = self._project_point(C + u * r, view_direction)
        pv = self._project_point(C + w * r, view_direction)
        f1 = (pu[0] - o2[0], pu[1] - o2[1])
        f2 = (pv[0] - o2[0], pv[1] - o2[1])
        rmaj, rmin, rot = ellipse_from_conjugate(f1, f2)
        if rmaj < 1e-9:
            return None
        return {
            "type": "ellipse",
            "center": self._format_pt(o2),
            "rmaj": round(rmaj, 3),
            "rmin": round(rmin, 3),
            "rot_deg": round(rot, 3),
            "role": "boundary",
        }

    def project(
        self, view_direction: tuple[float, float, float]
    ) -> list[dict[str, Any]]:
        surf = self.shape.Surface
        apex = surf.Apex
        axis = App.Vector(surf.Axis)
        axis.normalize()
        alpha = surf.SemiAngle
        V = App.Vector(*view_direction)

        results = []
        rings = self._rings(axis)

        # 1. boundary rings -> circle / ellipse / line
        for C, r in rings:
            prim = self._project_ring(C, r, axis, view_direction)
            if prim:
                results.append(prim)

        # 2. silhouette generators
        u = axis.cross(App.Vector(1, 0, 0))
        if u.Length < 1e-9:
            u = axis.cross(App.Vector(0, 1, 0))
        u.normalize()
        w = axis.cross(u)
        w.normalize()
        Vu, Vw, Va = V.dot(u), V.dot(w), V.dot(axis)
        R = math.hypot(Vu, Vw)
        if R < 1e-9:
            return results  # axis-aligned view: no lateral silhouette
        rhs = math.tan(alpha) * Va / R
        if abs(rhs) > 1.0:
            return results  # view inside the half-angle: no real silhouette
        phi = math.atan2(Vw, Vu)
        dth = math.acos(rhs)
        for th in (phi + dth, phi - dth):
            radial = u * math.cos(th) + w * math.sin(th)
            if len(rings) >= 2:
                # frustum(-like): generator runs between the extreme rings
                (C1, r1), (C2, r2) = rings[0], rings[-1]
                P1 = C1 + radial * r1
                P2 = C2 + radial * r2
            elif len(rings) == 1:
                # full cone: ring -> apex (if the face doesn't reach the apex,
                # the _on_face midpoint gate below rejects the segment)
                (C1, r1) = rings[0]
                P1 = C1 + radial * r1
                P2 = apex
            else:
                continue  # no circular boundary: skip silhouette
            if not self._on_face((P1 + P2) * 0.5):
                continue
            p1 = self._format_pt(self._project_point(P1, view_direction))
            p2 = self._format_pt(self._project_point(P2, view_direction))
            if p1 != p2:
                results.append(
                    {"type": "line", "p1": p1, "p2": p2, "role": "silhouette"}
                )
        return results


if __name__ == "__main__":
    import sys
    import Part
    import TechDraw

    print("=== Unit Test: ConeProjector ===")

    def occ_prims(shape, vd):
        """HLR edges (visible + hidden groups) as (segments, circles, ellipses).
        Segment = the edge's endpoint pair, for ANY curve kind: HLR represents an
        edge-on circle as a degenerate flat B-spline, and the production matcher
        (render_dataset._record) matches lines by point_on_segment of the HLR
        endpoints against the oracle segment — mimic exactly that here."""
        res = TechDraw.projectEx(shape, App.Vector(*vd))
        edges = []
        for gi in (0, 1, 3, 5, 8):
            edges += list(res[gi].Edges)
        segs, circles, ellipses = [], [], []
        for e in edges:
            try:
                c = e.Curve
            except Exception:
                c = None
            if c is not None and c.TypeId == "Part::GeomCircle":
                circles.append(((c.Center.x, c.Center.y), c.Radius))
                continue
            if c is not None and c.TypeId == "Part::GeomEllipse":
                ellipses.append(
                    ((c.Center.x, c.Center.y), c.MajorRadius, c.MinorRadius)
                )
                continue
            vs = e.Vertexes
            if len(vs) >= 2:
                segs.append(((vs[0].X, vs[0].Y), (vs[-1].X, vs[-1].Y)))
        return segs, circles, ellipses

    def on_seg(p, q1, q2, tol=1e-3):
        dx, dy = q2[0] - q1[0], q2[1] - q1[1]
        L2 = dx * dx + dy * dy
        if L2 < 1e-12:
            return math.hypot(p[0] - q1[0], p[1] - q1[1]) < tol
        t = ((p[0] - q1[0]) * dx + (p[1] - q1[1]) * dy) / L2
        if t < -tol or t > 1 + tol:
            return False
        return math.hypot(p[0] - (q1[0] + t * dx), p[1] - (q1[1] + t * dy)) < tol

    def seg_matched(oracle_seg, segs):
        """production semantics: some HLR edge's endpoints both lie ON the oracle segment"""
        (a1, a2) = oracle_seg
        return any(on_seg(o1, a1, a2) and on_seg(o2, a1, a2) for (o1, o2) in segs)

    # NOTE on frames: for the primary ortho views the oracle frame from
    # BaseProjector._project_point equals HLR's local frame (that is the whole
    # point of the empirically-derived (lx, ly) mappings in base.py), so oracle
    # coords compare directly against raw projectEx output here.
    fails = 0

    def check(name, shape, vd, expect_sil):
        global fails
        cone_faces = [
            f
            for f in shape.Faces
            if hasattr(f, "Surface") and f.Surface.TypeId == "Part::GeomCone"
        ]
        segs, occ_circ, occ_ell = occ_prims(shape, vd)
        nsil = nsil_ok = nring = nring_ok = 0
        for f in cone_faces:
            for p in ConeProjector(f).project(vd):
                if p["role"] == "silhouette":
                    nsil += 1
                    nsil_ok += seg_matched((p["p1"], p["p2"]), segs)
                elif p["type"] == "circle":
                    nring += 1
                    nring_ok += any(
                        abs(p["center"][0] - c[0]) < 1e-3
                        and abs(p["center"][1] - c[1]) < 1e-3
                        and abs(p["radius"] - r) < 1e-3
                        for (c, r) in occ_circ
                    )
                elif p["type"] == "ellipse":
                    nring += 1
                    nring_ok += any(
                        abs(p["center"][0] - c[0]) < 1e-3
                        and abs(p["center"][1] - c[1]) < 1e-3
                        and abs(p["rmaj"] - rmaj) < 1e-3
                        and abs(p["rmin"] - rmin) < 1e-3
                        for (c, rmaj, rmin) in occ_ell
                    )
                elif p["type"] == "line":  # edge-on ring
                    nring += 1
                    nring_ok += seg_matched((p["p1"], p["p2"]), segs)
        ok = nsil == expect_sil and nsil_ok == nsil and nring_ok == nring and nring > 0
        print(
            "  [%s] %-14s vd=%s  sil %d/%d matched, rings %d/%d matched"
            % ("OK" if ok else "FAIL", name, vd, nsil_ok, nsil, nring_ok, nring)
        )
        if not ok:
            fails += 1

    plain = Part.makeCone(10, 0, 20)  # base r=10, apex, h=20 (+Z axis)
    frus = Part.makeCone(10, 5, 15)  # frustum
    tilt = Part.makeCone(10, 0, 20)
    tilt.rotate(App.Vector(0, 0, 0), App.Vector(1, 0, 0), 35)
    box = Part.makeBox(40, 40, 20, App.Vector(-20, -20, -20))
    csk = box.cut(
        Part.makeCone(3, 9, 6, App.Vector(0, 0, -6))
    )  # countersink-ish pocket

    # front view (0,-1,0): oracle frame (-z, x); HLR front frame == same (see base.py)
    check("plain_cone", plain, (0, -1, 0), expect_sil=2)  # edge-on axis
    check("frustum", frus, (0, -1, 0), expect_sil=2)
    check("tilted_cone", tilt, (0, -1, 0), expect_sil=2)  # oblique axis
    check("tilted_cone", tilt, (0, 0, 1), expect_sil=2)  # oblique from top
    check("countersunk", csk, (0, -1, 0), expect_sil=2)  # internal cone (hidden lines)
    # axis-aligned view: silhouette degenerates to nothing, rings -> circles
    check("plain_cone", plain, (0, 0, 1), expect_sil=0)
    check("frustum", frus, (0, 0, 1), expect_sil=0)
    check("countersunk", csk, (0, 0, 1), expect_sil=0)

    if fails:
        print("FAILED: %d case(s)" % fails)
        sys.exit(1)
    print("All ConeProjector cases matched TechDraw.projectEx.")
