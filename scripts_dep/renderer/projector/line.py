try:
    import freecad
except ImportError:
    pass
import FreeCAD as App
from typing import Any
from .base import BaseProjector


class LineProjector(BaseProjector):
    def get_supported_types(self) -> list[str]:
        return ["Part::GeomLine"]

    def project(
        self, view_direction: tuple[float, float, float]
    ) -> list[dict[str, Any]]:
        if len(self.shape.Vertexes) != 2:
            return []

        v1 = self.shape.Vertexes[0].Point
        v2 = self.shape.Vertexes[1].Point

        p1 = self._format_pt(self._project_point(v1, view_direction))
        p2 = self._format_pt(self._project_point(v2, view_direction))

        # Check for degeneracy (point-on view)
        if p1 == p2:
            return []

        return [{"type": "line", "p1": p1, "p2": p2, "role": "edge"}]


if __name__ == "__main__":
    import Part
    import TechDraw

    print("=== Unit Test: LineProjector ===")
    box = Part.makeBox(10, 20, 30)
    direction = (0, -1, 0)

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

    matches = 0
    generated = 0
    for edge in box.Edges:
        if hasattr(edge, "Curve") and edge.Curve.TypeId == "Part::GeomLine":
            projector = LineProjector(edge)
            results = projector.project(direction)

            for r in results:
                generated += 1
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
                        print(f"[SUCCESS] Edge Line {p1}-{p2} matched OCC.")
                        matches += 1
                    else:
                        print(f"[FAIL] Line {p1}-{p2} did NOT match.")

    print(f"Total Matches: {matches} / {generated} generated lines")
