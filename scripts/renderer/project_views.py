#!/usr/bin/env python
"""Phase-1 renderer, stage 1 (FreeCAD env): solid -> multi-view HLR primitives +
dimensions + 3D provenance, emitted as the Tier-1/Tier-2 "drawing graph" JSON.

This is the *desirable 2D intermediate representation* for the 2D->3D stage:
instead of raster pixels, the downstream model can consume parsed, typed
geometric primitives (line/circle/arc, visible/hidden) and dimension annotations
that are bound to those primitives AND to the originating 3D feature/parameter.
Because data is synthetic we get all of this as ground truth for free; a real
OCR+vectorization stage (Morpho's 2D->2D CAD) would produce the same schema, noisily.

Run (FreeCAD env):
  PYTHONPATH=/home/ryotaro/github/FreeCAD/build/release/lib \
    conda run -n freecad python scripts/renderer/project_views.py --out experiments/renderer_demo
"""
import os, json, math, argparse
import FreeCAD as App
import Part
import TechDraw

# projectEx returns 10 edge groups; 0-4 are visible, 5-9 are hidden (HLR types
# V,V1,VN,VO,VI, H,H1,HN,HO,HI). We merge within visible / hidden.
VISIBLE_GROUPS = range(0, 5)
HIDDEN_GROUPS = range(5, 10)

# Standard third-angle-ish views. NOTE: TechDraw.projectEx rotates the projection
# onto the global XY sheet plane, so per-view 2D coords are always (x, y) of the
# returned edges (reading the original Z collapses non-top views). ax/ay kept as
# XY for all; `dir` selects the view.
VIEWS = {
    "front": {"dir": (0, -1, 0), "ax": (1, 0, 0), "ay": (0, 1, 0)},
    "top":   {"dir": (0, 0, -1), "ax": (1, 0, 0), "ay": (0, 1, 0)},
    "right": {"dir": (1, 0, 0),  "ax": (1, 0, 0), "ay": (0, 1, 0)},
}


def v3(t):
    return App.Vector(*t)


def to2d(pt, ax, ay):
    return [round(pt.dot(v3(ax)), 4), round(pt.dot(v3(ay)), 4)]


def edge_to_primitive(edge, ax, ay, visibility, pid):
    """Classify a projected OCC edge as line / circle / arc and emit 2D params."""
    c = edge.Curve
    prim = {"id": pid, "visibility": visibility}
    tname = type(c).__name__
    try:
        if tname == "Line":
            p1, p2 = edge.Vertexes[0].Point, edge.Vertexes[-1].Point
            prim.update(type="line", p1=to2d(p1, ax, ay), p2=to2d(p2, ax, ay))
        elif tname == "Circle":
            center = to2d(c.Center, ax, ay)
            r = round(c.Radius, 4)
            # full circle vs arc: compare param range to 2*pi
            u0, u1 = edge.FirstParameter, edge.LastParameter
            if abs((u1 - u0) - 2 * math.pi) < 1e-6:
                prim.update(type="circle", center=center, r=r)
            else:
                p_s, p_e = edge.valueAt(u0), edge.valueAt(u1)
                a0 = math.degrees(math.atan2(to2d(p_s, ax, ay)[1] - center[1],
                                             to2d(p_s, ax, ay)[0] - center[0]))
                a1 = math.degrees(math.atan2(to2d(p_e, ax, ay)[1] - center[1],
                                             to2d(p_e, ax, ay)[0] - center[0]))
                prim.update(type="arc", center=center, r=r, a0=round(a0, 2), a1=round(a1, 2))
        else:
            # fallback: discretize unknown curve to a polyline
            pts = [to2d(edge.valueAt(edge.FirstParameter + i * (edge.LastParameter - edge.FirstParameter) / 16), ax, ay)
                   for i in range(17)]
            prim.update(type="polyline", pts=pts)
    except Exception as e:
        prim.update(type="unknown", err=str(e))
    return prim


def project_view(shape, vname, vdef):
    res = TechDraw.projectEx(shape, v3(vdef["dir"]))
    prims, n = [], 0
    for groups, vis in ((VISIBLE_GROUPS, "visible"), (HIDDEN_GROUPS, "hidden")):
        for gi in groups:
            grp = res[gi]
            if grp is None or not hasattr(grp, "Edges"):
                continue
            for e in grp.Edges:
                prims.append(edge_to_primitive(e, vdef["ax"], vdef["ay"], vis, f"{vname[0]}{n}"))
                n += 1
    return {"name": vname, "dir": list(vdef["dir"]),
            "x_axis": list(vdef["ax"]), "y_axis": list(vdef["ay"]),
            "primitives": prims}


def find_circle_prim(view, target_r):
    for p in view["primitives"]:
        if p.get("type") == "circle" and abs(p["r"] - target_r) < 1e-3:
            return f'{view["name"]}:{p["id"]}'
    return None


def demo_solid():
    """Box 80x60x40 with a vertical Ø20 through hole (default when no input given)."""
    box = Part.makeBox(80, 60, 40)
    hole = Part.makeCylinder(10, 60, App.Vector(40, 30, -10), App.Vector(0, 0, 1))
    return box.cut(hole)


def load_shape(args):
    """Seed solids must be B-rep (STL meshes give triangulated garbage under HLR).
    Accept STEP, or a CadQuery .py (executes code, result in `r`/`result`)."""
    if args.step:
        return Part.read(args.step)
    if args.cadquery:
        import cadquery as cq  # noqa
        g = {}
        exec(open(args.cadquery).read(), g)
        wp = g.get("r", g.get("result"))
        tmp = "/tmp/_cq_seed.step"
        cq.exporters.export(wp, tmp)
        return Part.read(tmp)
    return demo_solid()


def detect_cylinder_diameters(shape, max_n=8):
    """Auto-detect circular features (cylindrical faces) -> distinct diameters.
    Prototype heuristic: any cylindrical face counts; a production version would
    classify hole vs boss (concavity / normal direction) and read depth/thread."""
    rads = []
    for f in shape.Faces:
        if type(f.Surface).__name__ == "Cylinder":
            rads.append(round(f.Surface.Radius, 3))
    out, seen = [], set()
    for r in sorted(rads):
        if all(abs(r - s) > 1e-2 for s in seen):
            seen.add(r); out.append(r)
    return out[:max_n]


def build_graph(shape, part_id, provenance=None):
    """provenance: optional dict of ground-truth params (when seed came from code).
    When absent (e.g. raw STEP), dimensions are auto-derived (bbox + cylinders)."""
    views = [project_view(shape, n, d) for n, d in VIEWS.items()]
    vmap = {v["name"]: v for v in views}
    bb = shape.BoundBox
    dx, dy, dz = round(bb.XLength, 3), round(bb.YLength, 3), round(bb.ZLength, 3)
    graph = {"part_id": part_id, "units": "mm",
             "bbox_3d": {"dx": dx, "dy": dy, "dz": dz},
             "provenance_source": "code" if provenance else "auto(bbox+cylinders)",
             "views": views, "annotations": [], "correspondences": []}
    # --- Tier-1 overall dims + Tier-2 provenance ---
    graph["annotations"] += [
        {"id": "d_len", "type": "linear", "subtype": "horizontal", "value": dx,
         "refs": ["front"], "provenance": {"feature": "bbox", "param": "dx"}},
        {"id": "d_hgt", "type": "linear", "subtype": "vertical", "value": dz,
         "refs": ["front"], "provenance": {"feature": "bbox", "param": "dz"}},
        {"id": "d_wid", "type": "linear", "subtype": "vertical", "value": dy,
         "refs": ["top"], "provenance": {"feature": "bbox", "param": "dy"}},
    ]
    # --- circular features -> diameter dims bound to circle primitives ---
    for k, r in enumerate(detect_cylinder_diameters(shape)):
        ref = find_circle_prim(vmap["top"], r) or find_circle_prim(vmap["front"], r) \
              or find_circle_prim(vmap["right"], r)
        dia = round(2 * r, 3)
        graph["annotations"].append(
            {"id": f"d_cyl{k}", "type": "diameter", "value": dia, "text": f"⌀{dia:g}",
             "refs": [ref] if ref else [], "provenance": {"feature": f"cylinder{k}", "param": "diameter"}})
        if ref:
            graph["correspondences"].append(
                {"feature": f"cylinder{k}", "entities": [ref, "front:hidden", "right:hidden"]})
    return graph


def emit_one(shape, part_id, out_dir, write_step=True):
    if write_step:
        Part.export([shape], os.path.join(out_dir, f"{part_id}.step"))
    graph = build_graph(shape, part_id)
    jp = os.path.join(out_dir, f"{part_id}.graph.json")
    with open(jp, "w") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)
    nprim = sum(len(v["primitives"]) for v in graph["views"])
    print(f"[renderer] {part_id}: bbox={graph['bbox_3d']} | {nprim} prims, "
          f"{len(graph['annotations'])} dims, {len(graph['correspondences'])} holes")
    return jp


def main():
    import glob
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/renderer_demo")
    ap.add_argument("--id", default=None, help="part id (default: filename stem)")
    ap.add_argument("--step", default=None, help="single seed B-rep STEP file")
    ap.add_argument("--cadquery", default=None, help="single seed CadQuery .py (result in r/result)")
    ap.add_argument("--step-dir", default=None, help="batch: dir of *.step/*.stp (one process, amortized import)")
    ap.add_argument("--n", type=int, default=0, help="batch: limit number of parts (0=all)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.step_dir:  # batch mode: loop many STEPs in this single FreeCAD process
        files = sorted(glob.glob(os.path.join(args.step_dir, "**", "*.step"), recursive=True) +
                       glob.glob(os.path.join(args.step_dir, "**", "*.stp"), recursive=True))
        if args.n > 0:
            files = files[: args.n]
        ok = 0
        for i, f in enumerate(files):
            pid = os.path.splitext(os.path.basename(f))[0]
            try:
                emit_one(Part.read(f), pid, args.out, write_step=False)
                ok += 1
            except Exception as e:
                print(f"[renderer] SKIP {pid}: {type(e).__name__}: {e}")
        print(f"[renderer] batch done: {ok}/{len(files)} graphs -> {args.out}")
        return

    src = args.step or args.cadquery
    part_id = args.id or (os.path.splitext(os.path.basename(src))[0] if src else "demo_box_hole")
    emit_one(load_shape(args), part_id, args.out)


if __name__ == "__main__":
    main()
