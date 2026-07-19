try:
    import freecad
except ImportError:
    pass
import math
import FreeCAD as App
from typing import Any
from .base import BaseProjector


def ellipse_from_conjugate(f1: tuple[float, float], f2: tuple[float, float]):
    """Conjugate semi-diameters f1,f2 of an ellipse (P(t)=c+cos t*f1+sin t*f2) ->
    (rmaj, rmin, rot_deg). Any parallel projection of a circle/ellipse maps its two
    in-plane radius vectors to a pair of conjugate semi-diameters; the principal axes
    are recovered analytically (extremize |P(t)-c| at t0=½·atan2(2 f1·f2, |f1|²-|f2|²)).
    rot_deg is the major-axis direction in [0,180)."""
    ax, ay = f1
    bx, by = f2
    t0 = 0.5 * math.atan2(
        2 * (ax * bx + ay * by), (ax * ax + ay * ay) - (bx * bx + by * by)
    )
    c0, s0 = math.cos(t0), math.sin(t0)
    A = (c0 * ax + s0 * bx, c0 * ay + s0 * by)  # semi-axis at t0
    B = (-s0 * ax + c0 * bx, -s0 * ay + c0 * by)  # semi-axis at t0+90
    la, lb = math.hypot(*A), math.hypot(*B)
    (rmaj, rmin, mv) = (la, lb, A) if la >= lb else (lb, la, B)
    return rmaj, rmin, math.degrees(math.atan2(mv[1], mv[0])) % 180.0


class EllipseProjector(BaseProjector):
    """Projects a 3D elliptical edge to its analytic 2D ellipse (the general oblique
    circle case is handled by CircleProjector, which reuses ellipse_from_conjugate)."""

    def get_supported_types(self) -> list[str]:
        return ["Part::GeomEllipse"]

    def project(
        self, view_direction: tuple[float, float, float]
    ) -> list[dict[str, Any]]:
        cur = self.shape.Curve
        loc = cur.Center
        u = cur.XAxis
        v = cur.YAxis if hasattr(cur, "YAxis") else cur.Axis.cross(u)
        a, b = cur.MajorRadius, cur.MinorRadius

        o2 = self._project_point(loc, view_direction)
        pu = self._project_point(loc + u * a, view_direction)
        pv = self._project_point(loc + v * b, view_direction)
        f1 = (pu[0] - o2[0], pu[1] - o2[1])
        f2 = (pv[0] - o2[0], pv[1] - o2[1])
        rmaj, rmin, rot = ellipse_from_conjugate(f1, f2)
        if rmaj < 1e-9:
            return []
        return [
            {
                "type": "ellipse",
                "center": self._format_pt(o2),
                "rmaj": round(rmaj, 3),
                "rmin": round(rmin, 3),
                "rot_deg": round(rot, 3),
                "role": "edge",
            }
        ]


if __name__ == "__main__":
    import Part

    print("=== Unit Test: EllipseProjector ===")
    # a real elliptical edge: cut a cylinder with a tilted plane is nontrivial; instead
    # build an ellipse edge directly and project it head-on (identity) + tilted.
    el = Part.Ellipse(App.Vector(0, 0, 0), 10, 6)
    edge = el.toShape()
    proj = EllipseProjector(edge)
    print("top (plane of ellipse == XY, head-on):", proj.project((0, 0, 1)))
    print("front (edge-on -> degenerate):", proj.project((0, -1, 0)))
