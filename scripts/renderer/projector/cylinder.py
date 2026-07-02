try:
    import freecad
except ImportError:
    pass
import FreeCAD as App
from typing import Any
from .base import BaseProjector

class CylinderProjector(BaseProjector):

    def get_supported_types(self) -> list[str]:
        return ["Part::GeomCylinder"]

    def _on_face(self, pnt: App.Vector, tol: float = 1e-5) -> bool:
        try:
            return self.shape.isInside(pnt, tol, True)
        except Exception:
            # fall back to distance check if the isInside API is unavailable
            try:
                import Part as _Part
                vtx = _Part.Vertex(pnt)
                return self.shape.distToShape(vtx)[0] < tol
            except Exception:
                return True

    def project(self, view_direction: tuple[float, float, float]) -> list[dict[str, Any]]:
        surf = self.shape.Surface
        axis = surf.Axis
        loc = surf.Center
        r = surf.Radius
        
        dir_vec = App.Vector(*view_direction)
        
        results = []
        
        # Check if cylinder is viewed from the side (axis perpendicular to view direction)
        if abs(axis.dot(dir_vec)) < 1e-6:
            # Silhouette edges
            sil_normal = dir_vec.cross(axis)
            sil_normal.normalize()

            circular_edges = [e for e in self.shape.Edges if e.Curve.TypeId == "Part::GeomCircle"]
            if len(circular_edges) >= 2:
                # Get the two centers
                c1 = circular_edges[0].Curve.Center
                c2 = circular_edges[1].Curve.Center

                for sign in (1.0, -1.0):
                    q1 = c1 + sil_normal * (r * sign)
                    q2 = c2 + sil_normal * (r * sign)
                    # a partial cylindrical face (fillet, half-bore) may not
                    # reach the silhouette angle at all — only emit the
                    # silhouette generator if it actually lies on the face
                    if not self._on_face((q1 + q2) * 0.5):
                        continue
                    p1 = self._format_pt(self._project_point(q1, view_direction))
                    p2 = self._format_pt(self._project_point(q2, view_direction))
                    if p1 != p2:
                        results.append({"type": "line", "p1": p1, "p2": p2, "role": "silhouette"})
                
        # Check if cylinder is viewed head-on (axis parallel to view direction)
        elif abs(abs(axis.dot(dir_vec)) - 1.0) < 1e-6:
            # It projects to a circle.
            c_2d = self._format_pt(self._project_point(loc, view_direction))
            results.append({"type": "circle", "center": c_2d, "radius": r, "role": "boundary"})
            
        return results

if __name__ == "__main__":
    import Part
    import TechDraw
    
    print("=== Unit Test: CylinderProjector ===")
    cyl = Part.makeCylinder(5, 20)
    direction = (0, -1, 0) # Side view
    
    # Get OCC Ground Truth
    res = TechDraw.projectEx(cyl, App.Vector(*direction))
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
    for face in cyl.Faces:
        if face.Surface.TypeId == "Part::GeomCylinder":
            projector = CylinderProjector(face)
            results = projector.project(direction)
            
            for r in results:
                if r["type"] == "line":
                    p1, p2 = r["p1"], r["p2"]
                    matched = False
                    for oline in occ_lines:
                        if (oline[0] == p1 and oline[1] == p2) or (oline[0] == p2 and oline[1] == p1):
                            matched = True
                            break
                    if matched:
                        print(f"[SUCCESS] Silhouette Line {p1}-{p2} matched OCC.")
                        matches += 1
                    else:
                        print(f"[FAIL] Line {p1}-{p2} did NOT match.")
                    
    print(f"Total Matches: {matches} (Expected 2 for a Cylinder side view)")
