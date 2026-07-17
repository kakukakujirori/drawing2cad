try:
    import freecad
except ImportError:
    pass
import FreeCAD as App
from typing import Any
from .base import BaseProjector
from .ellipse import ellipse_from_conjugate

class CircleProjector(BaseProjector):
    
    def get_supported_types(self) -> list[str]:
        return ["Part::GeomCircle"]
        
    def project(self, view_direction: tuple[float, float, float]) -> list[dict[str, Any]]:
        curve = self.shape.Curve
        axis = curve.Axis
        loc = curve.Center
        r = curve.Radius
        
        dir_vec = App.Vector(*view_direction)
        
        # Check if circle is viewed edge-on (axis perpendicular to view direction)
        if abs(axis.dot(dir_vec)) < 1e-6:
            # It projects to a line segment.
            # The direction of the line in 3D is axis x view_direction
            line_dir = axis.cross(dir_vec)
            line_dir.normalize()
            
            p1_3d = loc + line_dir * r
            p2_3d = loc - line_dir * r
            
            p1 = self._format_pt(self._project_point(p1_3d, view_direction))
            p2 = self._format_pt(self._project_point(p2_3d, view_direction))
            
            return [{
                "type": "line",
                "p1": p1,
                "p2": p2,
                "role": "edge"
            }]
            
        # Check if circle is viewed face-on (axis parallel to view direction)
        elif abs(abs(axis.dot(dir_vec)) - 1.0) < 1e-6:
            # It projects to a circle in 2D.
            c_2d = self._format_pt(self._project_point(loc, view_direction))
            return [{
                "type": "circle",
                "center": c_2d,
                "radius": r,
                "role": "edge"
            }]
            
        # Oblique view: the circle projects to an analytic ellipse. Two in-plane
        # radius vectors (u, v) map to conjugate semi-diameters under the parallel
        # projection; ellipse_from_conjugate recovers the principal axes.
        u = axis.cross(dir_vec)
        if u.Length < 1e-9:
            u = axis.cross(App.Vector(1, 0, 0))
        if u.Length < 1e-9:
            u = axis.cross(App.Vector(0, 1, 0))
        u.normalize()
        v = axis.cross(u)
        v.normalize()

        o2 = self._project_point(loc, view_direction)
        pu = self._project_point(loc + u * r, view_direction)
        pv = self._project_point(loc + v * r, view_direction)
        f1 = (pu[0] - o2[0], pu[1] - o2[1])
        f2 = (pv[0] - o2[0], pv[1] - o2[1])
        rmaj, rmin, rot = ellipse_from_conjugate(f1, f2)
        return [{
            "type": "ellipse",
            "center": self._format_pt(o2),
            "rmaj": round(rmaj, 3),
            "rmin": round(rmin, 3),
            "rot_deg": round(rot, 3),
            "role": "edge",
        }]

if __name__ == "__main__":
    import Part
    import TechDraw
    
    print("=== Unit Test: CircleProjector ===")
    cyl = Part.makeCylinder(5, 20)
    
    # 1. Test Face-on view (Top view of a Z-up cylinder)
    print("\n--- Testing Face-On View ---")
    direction = (0, 0, 1)
    
    for edge in cyl.Edges:
        if hasattr(edge, "Curve") and edge.Curve.TypeId == "Part::GeomCircle":
            projector = CircleProjector(edge)
            results = projector.project(direction)
            print(f"Projected Circle (Face-On): {results}")
            
    # 2. Test Edge-on view (Side view of a Z-up cylinder)
    print("\n--- Testing Edge-On View ---")
    direction = (0, -1, 0)
    
    for edge in cyl.Edges:
        if hasattr(edge, "Curve") and edge.Curve.TypeId == "Part::GeomCircle":
            projector = CircleProjector(edge)
            results = projector.project(direction)
            print(f"Projected Circle (Edge-On): {results}")
