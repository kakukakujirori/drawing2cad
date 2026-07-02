import abc
from typing import Any
import FreeCAD as App
import Part

class BaseProjector(abc.ABC):
    """
    Abstract base class for all 3D mathematical projection handlers.
    
    A projector takes a specific FreeCAD topological entity (Face, Edge) 
    and a view direction, and mathematically calculates its 2D projection 
    coordinates, which can then be matched against OCC TechDraw outputs.
    """
    
    def __init__(self, shape: Part.Shape):
        """
        :param shape: The FreeCAD topological entity (e.g. Part.Face)
        """
        self.shape = shape
        
        # Sanity check: Ensure the shape is supported by this projector
        type_id = None
        if hasattr(shape, "Surface"):
            type_id = shape.Surface.TypeId
        elif hasattr(shape, "Curve"):
            type_id = shape.Curve.TypeId
            
        if type_id not in self.get_supported_types():
            supported = ", ".join(self.get_supported_types())
            actual = type_id if type_id else type(shape)
            raise ValueError(f"{self.__class__.__name__} expects one of [{supported}], but got {actual}")
        
    def _project_point(self, p: App.Vector, view_direction: tuple[float, float, float]) -> tuple[float, float]:
        """
        Projects a 3D point to 2D OCC HLR coordinates based on view_direction.
        Currently supports primary orthographic views.

        The (lx, ly) frames below are measured empirically against
        TechDraw.projectEx output (axis-bar probe): they are HLR's local
        frames, not a free choice — View.REMAP in render_dataset.py undoes
        them into canonical model-axis (u, v) per view.
        """
        x, y, z = p.x, p.y, p.z
        if view_direction == (0, -1, 0): # Front
            return (-z, x)
        elif view_direction == (0, 0, 1): # Top
            return (x, y)
        elif view_direction == (1, 0, 0): # Right
            return (z, -y)
        else:
            return (x, y)
            
    def _format_pt(self, p2d: tuple[float, float]) -> tuple[float, float]:
        return (round(p2d[0], 3), round(p2d[1], 3))
        
    @abc.abstractmethod
    def get_supported_types(self) -> list[str]:
        """
        Returns a list of surface TypeIds (e.g., ["Part::GeomPlane"]) 
        that this projector can handle.
        """
        pass
    
    @abc.abstractmethod
    def project(self, view_direction: tuple[float, float, float]) -> list[dict[str, Any]]:
        """
        Mathematically projects the 3D shape into 2D based on the view direction.
        
        :param view_direction: A tuple or App.Vector representing the view direction (e.g. (0, -1, 0))
        :return: A list of geometric definitions (e.g., lines, circles) that represent 
                 the 2D projection of the shape, formatted in a standard way to be 
                 matched against OCC outputs.
                 Example return:
                 [
                     {"type": "line", "p1": (x1, y1), "p2": (x2, y2), "role": "edge-on"},
                     {"type": "circle", "center": (cx, cy), "radius": r, "role": "boundary"}
                 ]
        """
        pass
