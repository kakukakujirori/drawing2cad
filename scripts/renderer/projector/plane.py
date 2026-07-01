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
        
    def project(self, view_direction: tuple[float, float, float]) -> list[dict[str, Any]]:
        surf = self.shape.Surface
        normal = surf.Axis
        dir_vec = App.Vector(*view_direction)
        
        # Check if face is viewed edge-on (normal is perpendicular to view direction)
        dot = normal.dot(dir_vec)
        if abs(dot) < 1e-6:
            bb = self.shape.BoundBox
            pts = [
                App.Vector(bb.XMin, bb.YMin, bb.ZMin),
                App.Vector(bb.XMax, bb.YMin, bb.ZMin),
                App.Vector(bb.XMin, bb.YMax, bb.ZMin),
                App.Vector(bb.XMax, bb.YMax, bb.ZMin),
                App.Vector(bb.XMin, bb.YMin, bb.ZMax),
                App.Vector(bb.XMax, bb.YMin, bb.ZMax),
                App.Vector(bb.XMin, bb.YMax, bb.ZMax),
                App.Vector(bb.XMax, bb.YMax, bb.ZMax),
            ]
            
            proj_pts = [self._project_point(p, view_direction) for p in pts]
            
            min_u = min(p[0] for p in proj_pts)
            max_u = max(p[0] for p in proj_pts)
            min_v = min(p[1] for p in proj_pts)
            max_v = max(p[1] for p in proj_pts)
            
            p1 = self._format_pt((min_u, min_v))
            p2 = self._format_pt((max_u, max_v))
            
            # When edge-on, we return a single line segment
            return [{
                "type": "line",
                "p1": p1,
                "p2": p2,
                "role": "edge-on"
            }]
        
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
                    if (oline[0] == p1 and oline[1] == p2) or (oline[0] == p2 and oline[1] == p1):
                        matched = True
                        break
                if matched:
                    print(f"[SUCCESS] Edge-on Plane Line {p1}-{p2} matched OCC.")
                    matches += 1
                else:
                    print(f"[FAIL] Line {p1}-{p2} did NOT match.")
                    
    print(f"Total Matches: {matches} (Expected 4 for a Box front view)")
