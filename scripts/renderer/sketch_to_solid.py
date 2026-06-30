#!/usr/bin/env python
# sketch_to_solid.py -- Fusion360 sketch FCStd -> extruded B-rep solid STEP.
#
# The Fusion360Gallery `freecad_commands_outsource_split` subset ships SKETCH-only
# FCStd documents (2D Sketcher profiles), not the 3D `reconstruction` STEP subset the
# renderer expects. HLR drawing projection needs a solid, so we synthesize one:
#   sketch edges -> closed wires (Part.sortEdges) -> face with holes (Bullseye
#   facemaker treats nested closed wires as holes) -> prism (extrude along sketch normal).
# Circular inner loops become through-holes => circle primitives in the top view, the
# signal CircleNet is trained on.
#
# Run (drawing2cad env):
#   python sketch_to_solid.py probe  <split_dir> [N]
#   python sketch_to_solid.py build  <split_dir> <out_step_dir> [N]
import os, sys, glob, math, random, json, traceback
import freecad          # conda-forge shim: puts FreeCAD's libs on sys.path
import FreeCAD as App
import Part

MIN_PROFILE_MM = 8.0     # skip degenerate tiny sketches
MAX_PROFILE_MM = 1200.0  # skip absurdly large
MIN_AREA_FRAC  = 0.02    # face area must be a sane fraction of bbox area


def load_sketch_edges(fcstd_path):
    """Open an FCStd doc and return the edges of its first SketchObject (+ doc to close)."""
    doc = App.openDocument(fcstd_path)
    sk = None
    for o in doc.Objects:
        if o.TypeId == "Sketcher::SketchObject":
            sk = o; break
    if sk is None:
        return None, doc
    shp = getattr(sk, "Shape", None)
    if shp is None or not shp.Edges:
        return None, doc
    return shp.Edges, doc


def closed_wires(edges):
    """Group loose edges into wires; keep only the closed, non-degenerate ones."""
    wires = []
    try:
        groups = Part.sortEdges(edges)
    except Exception:
        groups = [edges]
    for g in groups:
        try:
            w = Part.Wire(g)
        except Exception:
            continue
        if w.isClosed() and w.Length > 1e-6:
            wires.append(w)
    return wires


def sketch_to_solid(edges, depth=None, seed=0):
    """edges -> extruded solid (face-with-holes via Bullseye facemaker). Returns Part.Solid or None."""
    wires = closed_wires(edges)
    if not wires:
        return None, "no_closed_wire"
    # profile extent (for depth + sanity)
    cmp = Part.Compound(wires)
    bb = cmp.BoundBox
    ext = max(bb.XLength, bb.YLength)
    if ext < MIN_PROFILE_MM or ext > MAX_PROFILE_MM:
        return None, "bad_extent_%.1f" % ext
    try:
        face = Part.makeFace(wires, "Part::FaceMakerBullseye")
    except Exception as e:
        return None, "facemaker_fail"
    if face.Area < MIN_AREA_FRAC * (bb.XLength * bb.YLength + 1e-9):
        return None, "tiny_area"
    rng = random.Random(seed)
    if depth is None:
        # plate-like thickness proportional to profile, clamped to a realistic band
        depth = max(3.0, min(0.6 * ext, rng.uniform(0.12, 0.5) * ext, 80.0))
    try:
        # sketch lives on XY plane (Fusion360 export) -> extrude along +Z
        solid = face.extrude(App.Vector(0, 0, depth))
    except Exception:
        return None, "extrude_fail"
    if not solid.isValid():
        try:
            solid = solid.removeSplitter()
        except Exception:
            pass
    if (not solid.isValid()) or solid.Volume < 1e-3 or not solid.Solids:
        return None, "invalid_solid"
    return solid, "ok"


def count_cyls(solid):
    n = 0
    for f in solid.Faces:
        try:
            if f.Surface.TypeId == "Part::GeomCylinder":
                n += 1
        except Exception:
            pass
    return n


def probe(split_dir, n=12):
    files = sorted(glob.glob(os.path.join(split_dir, "**", "*.FCStd"), recursive=True))
    random.Random(0).shuffle(files)
    files = files[:n]
    ok = 0
    for i, f in enumerate(files):
        try:
            edges, doc = load_sketch_edges(f)
            if edges is None:
                print("  %-40s SKETCH_NONE" % os.path.basename(f)[:40]); App.closeDocument(doc.Name); continue
            solid, status = sketch_to_solid(edges, seed=i)
            if solid is None:
                print("  %-40s %s" % (os.path.basename(f)[:40], status))
            else:
                ok += 1
                bb = solid.BoundBox
                print("  %-40s OK  vol=%.0f bbox=%.0fx%.0fx%.0f cyls=%d solids=%d" % (
                    os.path.basename(f)[:40], solid.Volume, bb.XLength, bb.YLength, bb.ZLength,
                    count_cyls(solid), len(solid.Solids)))
            App.closeDocument(doc.Name)
        except Exception:
            print("  %-40s EXC %s" % (os.path.basename(f)[:40], traceback.format_exc().splitlines()[-1]))
    print("\nPROBE: %d/%d sketches -> valid solid" % (ok, len(files)))


def build(split_dir, out_dir, n=0):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(split_dir, "**", "*.FCStd"), recursive=True))
    random.Random(0).shuffle(files)
    if n > 0:
        files = files[:n]
    stats = {"ok": 0, "fail": {}, "total": len(files)}
    manifest = []
    for i, f in enumerate(files):
        try:
            edges, doc = load_sketch_edges(f)
            if edges is None:
                stats["fail"]["sketch_none"] = stats["fail"].get("sketch_none", 0) + 1
                App.closeDocument(doc.Name); continue
            solid, status = sketch_to_solid(edges, seed=i)
            if solid is None:
                stats["fail"][status] = stats["fail"].get(status, 0) + 1
                App.closeDocument(doc.Name); continue
            name = os.path.splitext(os.path.basename(f))[0]
            outp = os.path.join(out_dir, name + ".step")
            solid.exportStep(outp)
            stats["ok"] += 1
            manifest.append({"name": name, "src": f, "step": outp,
                             "volume": round(solid.Volume, 2), "cyls": count_cyls(solid)})
            App.closeDocument(doc.Name)
            if stats["ok"] % 50 == 0:
                print("  built %d ..." % stats["ok"], flush=True)
        except Exception:
            stats["fail"]["exc"] = stats["fail"].get("exc", 0) + 1
    with open(os.path.join(out_dir, "_seed_manifest.json"), "w") as fp:
        json.dump({"stats": stats, "parts": manifest}, fp, indent=1)
    print("BUILD: %d/%d -> STEP solids in %s" % (stats["ok"], stats["total"], out_dir))
    print("  fail breakdown:", json.dumps(stats["fail"]))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "probe"
    if cmd == "probe":
        probe(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 12)
    elif cmd == "build":
        build(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 0)
    else:
        print(__doc__)
