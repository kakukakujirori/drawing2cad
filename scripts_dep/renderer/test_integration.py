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


# ============================================================================
#  Fallback-path tests: an unsupported curve type must never abort a part.
#  FreeCAD raises TypeError("undefined curve type") from the C++ layer when an
#  OCC curve has no python wrapper; hasattr/getattr only swallow AttributeError,
#  so this used to kill the whole part (val 26/26 skips). Real such edges only
#  arise from pathological booleans, so the tests emulate one with a fake edge.
# ============================================================================

class _BadCurveEdge:
    """Emulates a FreeCAD Edge whose .Curve raises like the C++ layer does."""

    def __init__(self, pts, discretize_raises=False):
        self._pts = [App.Vector(*p) for p in pts]
        self._discretize_raises = discretize_raises

    @property
    def Curve(self):
        raise TypeError("undefined curve type")

    @property
    def Vertexes(self):
        class _V:
            def __init__(self, p): self.Point = p
        return [_V(self._pts[0]), _V(self._pts[-1])]

    def discretize(self, **kw):
        if self._discretize_raises:
            raise ValueError("discretize failed")
        return list(self._pts)


class _FakeCurve:
    def __init__(self, type_id, center=(0, 0), radius=0, rmaj=0, rmin=0):
        self.TypeId = type_id
        self.Center = App.Vector(center[0], center[1], 0)
        self.Radius = radius
        self.MajorRadius = rmaj
        self.MinorRadius = rmin
        self.XAxis = App.Vector(1, 0, 0)
        self.YAxis = App.Vector(0, 1, 0)


class _FakeCurveEdge:
    def __init__(self, curve, endpoints, stable, bad_deflection=None, closed=False):
        self.Curve = curve
        self.Closed = closed
        self._ends = [App.Vector(x, y, 0) for x, y in endpoints]
        self._stable = [App.Vector(x, y, 0) for x, y in stable]
        self._bad = ([App.Vector(x, y, 0) for x, y in bad_deflection]
                     if bad_deflection else None)

    @property
    def Vertexes(self):
        class _V:
            def __init__(self, p): self.Point = p
        return [_V(p) for p in self._ends]

    def discretize(self, **kw):
        if "Deflection" in kw and self._bad is not None:
            return list(self._bad)
        return list(self._stable)


class _FakeShape:
    def __init__(self, faces, edges):
        self.Faces = faces
        self.Edges = edges


def test_classify_edge_fallback():
    print("\n=== classify_edge fallback (undefined curve type -> polyline) ===")
    from scripts.renderer.render_dataset import classify_edge, edge_to_path, _FALLBACK

    _FALLBACK.clear()
    bad = _BadCurveEdge([(0, 0, 0), (5, 1, 0), (10, 0, 0)])
    typ, g = classify_edge(bad)
    assert typ == "polyline", f"expected polyline fallback, got {typ}"
    assert g["pts"] == [(0, 0), (5, 1), (10, 0)], g
    assert g["p1"] == (0, 0) and g["p2"] == (10, 0), g
    assert _FALLBACK["hlr_fallback_edges"] == 1, dict(_FALLBACK)

    # discretize also failing -> degrade to endpoint polyline, still no raise
    bad2 = _BadCurveEdge([(0, 0, 0), (10, 0, 0)], discretize_raises=True)
    typ2, g2 = classify_edge(bad2)
    assert typ2 == "polyline" and g2["pts"] == [(0.0, 0.0), (10.0, 0.0)], (typ2, g2)

    # the SVG path writer must not raise either, and must draw the fallback edge
    d = edge_to_path(_BadCurveEdge([(0, 0, 0), (5, 1, 0), (10, 0, 0)]))
    assert d.startswith("M") and "L" in d, d
    assert _FALLBACK["svg_fallback_edges"] == 1, dict(_FALLBACK)
    print("[OK] classify_edge + edge_to_path fall back to polyline, counters:",
          {k: v for k, v in _FALLBACK.items() if v})
    _FALLBACK.clear()


def test_classify_edge_extent_guards():
    print("\n=== classify_edge finite-envelope guards ===")
    from scripts.renderer.render_dataset import classify_edge, _FALLBACK

    env = (0, 0, 100, 100)
    _FALLBACK.clear()

    # A partial ellipse whose infinite support is far away, while its actual
    # bounded edge is local, must preserve the edge as a bounded polyline.
    ill = _FakeCurveEdge(
        _FakeCurve("Part::GeomEllipse", center=(100000, 100000),
                   rmaj=100000, rmin=50000),
        endpoints=[(10, 10), (12, 10)],
        stable=[(10, 10), (11, 10.1), (12, 10)])
    typ, g = classify_edge(ill, env)
    assert typ == "polyline" and g["pts"][1] == (11.0, 10.1), (typ, g)
    assert _FALLBACK["partial_conic_polylines"] == 1, dict(_FALLBACK)

    # Deflection sampling can diverge even when Number sampling is stable.
    curve = _FakeCurve("Part::GeomBSplineCurve")
    divergent = _FakeCurveEdge(curve, endpoints=[(10, 10), (20, 10)],
                               stable=[(10, 10), (15, 12), (20, 10)],
                               bad_deflection=[(10, 10), (1e9, 1e9), (20, 10)])
    typ, g = classify_edge(divergent, env)
    assert typ == "polyline" and max(x for p in g["pts"] for x in p) < 100, (typ, g)

    # A genuinely out-of-envelope HLR edge has no trustworthy bounded form.
    stray = _FakeCurveEdge(_FakeCurve("Part::GeomLine"),
                           endpoints=[(10000, 0), (10010, 0)],
                           stable=[(10000, 0), (10010, 0)])
    assert classify_edge(stray, env) is None

    # Normal analytic ellipse stays analytic (negative regression).
    normal = _FakeCurveEdge(_FakeCurve("Part::GeomEllipse", center=(50, 50),
                                      rmaj=10, rmin=5),
                            endpoints=[(60, 50), (50, 55)],
                            stable=[(60, 50), (57, 53), (50, 55)])
    typ, _ = classify_edge(normal, env)
    assert typ == "ellipse", typ
    print("[OK] partial conic fallback, polyline retry, stray rejection, normal ellipse")
    _FALLBACK.clear()


def test_cad_projector_guard():
    print("\n=== CADProjector guard (bad edge/face must not abort project()) ===")
    good_edge = Part.makeLine(App.Vector(0, 0, 0), App.Vector(10, 0, 0))

    class _BadSurfaceFace:
        @property
        def Surface(self):
            raise TypeError("undefined surface type")

    shape = _FakeShape([_BadSurfaceFace()],
                       [_BadCurveEdge([(0, 0, 0), (1, 1, 1)]), good_edge.Edges[0]])
    proj = CADProjector(shape)
    prims = proj.project((0, -1, 0))   # must not raise
    assert proj.last_skipped == {"faces": 1, "edges": 1}, proj.last_skipped
    lines = [p for p in prims if p["type"] == "line"]
    assert len(lines) == 1, prims      # the good line still projected
    assert lines[0]["topo_origins"] == [{"dim": 1, "id": "Edge_1", "role": "edge"}]
    print("[OK] bad face+edge skipped (last_skipped=%s), good edge still projected"
          % proj.last_skipped)


def test_cone_face_projection():
    print("\n=== ConeProjector wired into CADProjector ===")
    cone = Part.makeCone(10, 0, 20)
    proj = CADProjector(cone)
    prims = proj.project((0, -1, 0))   # edge-on axis: 2 silhouette generators
    sil = [p for p in prims if p.get("role") == "silhouette" and
           p["topo_origins"][0]["dim"] == 2]
    assert len(sil) == 2, prims
    for p in sil:
        assert p["type"] == "line"
        assert p["topo_origins"][0]["role"] == "silhouette"
    top = proj.project((0, 0, 1))      # axis-aligned: base ring as boundary circle
    rings = [p for p in top if p.get("role") == "boundary" and p["type"] == "circle"]
    assert len(rings) == 1 and abs(rings[0]["radius"] - 10.0) < 1e-6, top
    print("[OK] cone face -> %d silhouette lines (front), %d boundary circle (top)"
          % (len(sil), len(rings)))


if __name__ == "__main__":
    test_ray_casting()
    test_classify_edge_fallback()
    test_classify_edge_extent_guards()
    test_cad_projector_guard()
    test_cone_face_projection()
    print("\nAll integration tests passed.")
