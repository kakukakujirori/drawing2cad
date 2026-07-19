try:
    import freecad
except ImportError:
    pass
import FreeCAD as App
from typing import Any
from .base import BaseProjector


class PlaneProjector(BaseProjector):
    def get_supported_types(self) -> list[str]:
        return ["Part::GeomPlane"]

    def project(
        self, view_direction: tuple[float, float, float]
    ) -> list[dict[str, Any]]:
        surf = self.shape.Surface
        normal = surf.Axis
        dir_vec = App.Vector(*view_direction)

        # Check if face is viewed edge-on (normal is perpendicular to view direction)
        dot = normal.dot(dir_vec)
        if abs(dot) < 1e-6:
            # The whole face projects onto a single line. Its extent must be
            # measured along that line from points ON the face (the bbox
            # diagonal used before produced a wrong, positive-slope segment
            # for slanted faces such as chamfers). Gather candidate extreme
            # points from the face's own edges:
            cand = []
            for e in self.shape.Edges:
                curve = getattr(e, "Curve", None)
                tid = curve.TypeId if curve is not None else None
                if tid == "Part::GeomCircle":
                    # circle lies in the face plane; its extremes along the
                    # projected line are center +/- r * (normal x view_dir)
                    d3 = normal.cross(dir_vec)
                    if d3.Length > 1e-9:
                        d3.normalize()
                        c = curve.Center
                        r = curve.Radius
                        cand.append(c + d3 * r)
                        cand.append(c - d3 * r)
                    for v in e.Vertexes:
                        cand.append(v.Point)
                elif tid == "Part::GeomLine":
                    for v in e.Vertexes:
                        cand.append(v.Point)
                else:
                    try:
                        cand.extend(e.discretize(Deflection=1e-3))
                    except Exception:
                        for v in e.Vertexes:
                            cand.append(v.Point)
            if len(cand) < 2:
                return []
            proj = [self._project_point(p, view_direction) for p in cand]
            # extremes along the (collinear) projected direction
            u0, v0 = proj[0]
            du = dv = 0.0
            for u, v in proj[1:]:
                if abs(u - u0) + abs(v - v0) > abs(du) + abs(dv):
                    du, dv = u - u0, v - v0
            L = (du * du + dv * dv) ** 0.5
            if L < 1e-9:
                return []
            du, dv = du / L, dv / L
            ts = [(p[0] - u0) * du + (p[1] - v0) * dv for p in proj]
            tmin, tmax = min(ts), max(ts)
            p1 = self._format_pt((u0 + du * tmin, v0 + dv * tmin))
            p2 = self._format_pt((u0 + du * tmax, v0 + dv * tmax))
            if p1 == p2:
                return []
            return [{"type": "line", "p1": p1, "p2": p2, "role": "edge-on"}]

        return []


if __name__ == "__main__":
    # Isolated Unit Test for PlaneProjector
    import Part
    import TechDraw

    print("=== Unit Test: PlaneProjector ===")
    box = Part.makeBox(10, 20, 30)
    direction = (0, -1, 0)

    # Get OCC Ground Truth
    res = TechDraw.projectEx(box, App.Vector(*direction))
    try:
        edges = list(res[0].Edges) + list(res[1].Edges) + list(res[3].Edges)
    except Exception:
        edges = []

    occ_lines = []
    for e in edges:
        vs = e.Vertexes
        if len(vs) == 2:
            p1 = (round(vs[0].X, 3), round(vs[0].Y, 3))
            p2 = (round(vs[1].X, 3), round(vs[1].Y, 3))
            occ_lines.append((p1, p2))

    # Run Projector
    matches = 0
    for face in box.Faces:
        projector = PlaneProjector(face)
        results = projector.project(direction)

        for r in results:
            if r["type"] == "line":
                p1, p2 = r["p1"], r["p2"]
                matched = False
                for oline in occ_lines:
                    if (oline[0] == p1 and oline[1] == p2) or (
                        oline[0] == p2 and oline[1] == p1
                    ):
                        matched = True
                        break
                if matched:
                    print(f"[SUCCESS] Edge-on Plane Line {p1}-{p2} matched OCC.")
                    matches += 1
                else:
                    print(f"[FAIL] Line {p1}-{p2} did NOT match.")

    print(f"Total Matches: {matches} (Expected 4 for a Box front view)")
