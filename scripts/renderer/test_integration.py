try:
    import freecad
except ImportError:
    pass
import FreeCAD as App
import Part

from scripts.renderer.cad_projector import CADProjector
from scripts.renderer.graph_builder import GraphBuilder

def test_ray_casting():
    print("=== Ray Casting Topological Test ===")
    
    # 1. Setup CAD Model
    shape = Part.makeBox(10, 20, 30)
    direction = (0, -1, 0) # Front view (Y is into the screen for standard Z-up)
    
    # 2. Run Pipeline
    projector = CADProjector(shape)
    prims = projector.project(direction)
    
    graph_builder = GraphBuilder(shape)
    enriched_prims = graph_builder.enrich_topo_origins(prims)
    
    # 3. Ray Casting Validation
    matches = 0
    errors = 0
    
    for i, prim in enumerate(enriched_prims):
        if prim["type"] != "line":
            continue
            
        p1 = prim["p1"]
        p2 = prim["p2"]
        
        # Midpoint in 2D
        mid_u = (p1[0] + p2[0]) / 2.0
        mid_v = (p1[1] + p2[1]) / 2.0
        
        # Back-project to 3D ray (Line segment)
        # For direction = (0, -1, 0): X -> -u, Z -> v ?
        # Wait, in base.py: 
        # if (0, -1, 0): return (-z, x)
        # So u = -z => z = -u
        # v = x => x = v
        
        # To be general, we know project_point. We can just inverse map for (0, -1, 0).
        x = mid_v
        z = -mid_u
        
        # Create a Ray along Y axis (the view direction)
        # Since view_direction is (0, -1, 0), the ray goes from +Y to -Y.
        ray_start = App.Vector(x, 100, z)
        ray_end = App.Vector(x, -100, z)
        ray_edge = Part.makeLine(ray_start, ray_end)
        
        # Find which Edges and Faces the ray intersects
        hit_edges = []
        for e_idx, edge in enumerate(shape.Edges):
            if edge.distToShape(ray_edge)[0] < 1e-4:
                hit_edges.append(f"Edge_{e_idx}")
                
        hit_faces = []
        for f_idx, face in enumerate(shape.Faces):
            if face.distToShape(ray_edge)[0] < 1e-4:
                hit_faces.append(f"Face_{f_idx}")
                
        expected_origins = set(hit_edges + hit_faces)
        actual_origins = {o["id"] for o in prim.get("topo_origins", [])}
        
        # Check if actual origins is a subset of expected origins (or exact match).
        # Note: A ray through an edge will hit the edge, its 2 parent faces, 
        # and possibly other faces/edges behind it. 
        # Our projector only projects visible or all? It projects all.
        # But GraphBuilder only adds PARENT faces of the edge, not faces behind it.
        # Actually, for an Edge, expected parent faces = the faces hit by the ray that share the edge.
        # Let's just check if all actual_origins are in expected_origins.
        if actual_origins.issubset(expected_origins):
            matches += 1
            print(f"[OK] Line {i} origins {actual_origins} subset of Ray Hits {expected_origins}")
        else:
            errors += 1
            print(f"[FAIL] Line {i} origins {actual_origins} NOT in Ray Hits {expected_origins}")
            
    print(f"Test Complete: {matches} Matches, {errors} Errors.")

if __name__ == "__main__":
    test_ray_casting()
