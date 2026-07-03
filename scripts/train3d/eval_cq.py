"""eval_cq.py — execute predicted CadQuery code in isolation and score it against a GT solid.

AMVDG->3D leg evaluator. For each sample we:
  1. exec the predicted CadQuery source in a *separate process* with a wall-clock
     timeout (CadQuery/OCC leak + crash + hang containment, cadrille evaluate.py style),
  2. pull the result solid (`result`, else `r`, else the last Workplane/Shape in globals),
  3. validity gates: OCC `shape.isValid()` + `Volume() > tol` + tessellated-mesh
     watertightness (an `isClosed` proxy),
  4. tessellate to a mesh and load the GT `.step` -> mesh,
  5. **translation-aligned, scale-PRESERVING voxel IoU** (absolute mm): each mesh is
     shifted so its bbox-min sits at the origin (the g2 part-frame convention; the GT
     CadQuery frame differs from it by a translation only), then both are rasterized
     into a shared integer voxel grid at a common pitch and compared as occupancy sets.
     Unlike cadrille/Zero-to-CAD (which rescale every mesh to a unit box, discarding
     absolute size), we keep millimetres so a mis-scaled part scores low — dimension
     grounding is the whole point of this project.
  6. **bbox dimension error in absolute mm** (sorted extents; the headline dimension metric),
     plus a secondary cadrille-style scale-normalized IoU for cross-reference.

Runs in a CadQuery env (conda `drawing2cad`: cadquery + OCP + trimesh + numpy).
No torch/transformers import, so `train_sft.py` (a different env) calls it as a subprocess.

Usage:
  # score a directory of predicted {id}.py against GT {id}.step:
  python eval_cq.py --pred-dir PREDS --gt-dir experiments/stage_z2c --out metrics.json
  # GT-code self-IoU smoke (predictions ARE the GT code -> expect IoU ~1.0):
  python eval_cq.py --pred-dir experiments/stage_z2c --gt-dir experiments/stage_z2c \
                    --ids-file paired_uuids.txt --limit 8 --out self_iou.json
"""
import os
import sys
import json
import math
import time
import argparse
import traceback
from multiprocessing import Process, Queue

import numpy as np

TESS_TOL = 0.1        # OCC linear tessellation tolerance (mm)
VOL_TOL = 1e-6        # min volume to count as a non-degenerate solid


# ---------------------------------------------------------------- solid -> mesh
def _shape_from_globals(g):
    """Recover the result Shape from an exec'd namespace, robustly."""
    import cadquery as cq
    for name in ("result", "r"):
        if name in g and g[name] is not None:
            obj = g[name]
            return obj.val() if hasattr(obj, "val") else obj
    # fallback: last Workplane / Shape bound in the namespace
    cand = None
    for v in g.values():
        if isinstance(v, cq.Workplane):
            cand = v.val()
        elif isinstance(v, cq.Shape):
            cand = v
    return cand


def _mesh_from_shape(shape):
    import trimesh
    verts, faces = shape.tessellate(TESS_TOL)
    # process=True merges the per-face duplicate vertices OCC emits, which is what
    # makes a valid CAD tessellation register as watertight (the isClosed proxy).
    return trimesh.Trimesh([(v.x, v.y, v.z) for v in verts], faces, process=True)


def _load_gt_mesh(step_path):
    import cadquery as cq
    solid = cq.importers.importStep(step_path).val()
    return _mesh_from_shape(solid), solid


# ---------------------------------------------------------------- voxel IoU
def _voxel_set(mesh, pitch, origin):
    """Occupancy as a set of integer voxel coords in a grid anchored at `origin`."""
    vg = mesh.voxelized(pitch)
    try:
        vg = vg.fill()          # fill interior so solids are solid, not shells
    except Exception:
        pass
    pts = vg.points             # world-space centres of occupied cells
    idx = np.floor((pts - origin) / pitch + 1e-6).astype(np.int64)
    return set(map(tuple, idx.tolist()))


def _aligned_origin(mesh, mode):
    if mode == "center":
        return (mesh.bounds[0] + mesh.bounds[1]) / 2.0
    return mesh.bounds[0]       # "min"


def _voxel_iou(m_pred, m_gt, vox_res, align, normalize_scale=False):
    """Translation-aligned voxel IoU. normalize_scale=True reproduces the cadrille
    unit-box IoU (both meshes rescaled to max-extent 1) for cross-reference."""
    p, g = m_pred.copy(), m_gt.copy()
    if normalize_scale:
        for m in (p, g):
            m.apply_translation(-(m.bounds[0] + m.bounds[1]) / 2.0)
            e = float(np.max(m.extents))
            if e > 1e-9:
                m.apply_scale(1.0 / e)
        ext = 1.0
    else:
        ext = float(np.max(g.extents))
    if ext <= 1e-9:
        return None
    pitch = ext / float(vox_res)
    o_p = _aligned_origin(p, "center" if normalize_scale else align)
    o_g = _aligned_origin(g, "center" if normalize_scale else align)
    sp = _voxel_set(p, pitch, o_p)
    sg = _voxel_set(g, pitch, o_g)
    if not sp or not sg:
        return 0.0
    inter = len(sp & sg)
    union = len(sp | sg)
    return inter / union if union else 0.0


# ---------------------------------------------------------------- per-sample
def eval_one(sample_id, pred_code, gt_step, cfg, q):
    """Runs inside a child process; puts a metrics dict on the queue."""
    out = {"id": sample_id, "exec_ok": False, "has_result": False,
           "is_valid": False, "is_watertight": False, "volume": 0.0,
           "valid": False, "iou": None, "iou_norm": None,
           "bbox_pred": None, "bbox_gt": None, "bbox_err_mm": None,
           "error_type": None}
    try:
        ns = {"__name__": "__main__"}
        try:
            import cadquery as cq
            ns["cq"] = cq                    # some model outputs assume `cq` exists
        except Exception:
            pass
        exec(pred_code, ns)
        out["exec_ok"] = True

        shape = _shape_from_globals(ns)
        if shape is None:
            out["error_type"] = "no_result_object"
            q.put(out); return
        out["has_result"] = True

        try:
            out["is_valid"] = bool(shape.isValid())
        except Exception:
            out["is_valid"] = False
        try:
            out["volume"] = float(shape.Volume())
        except Exception:
            out["volume"] = 0.0

        m_pred = _mesh_from_shape(shape)
        if len(m_pred.faces) < 3:
            out["error_type"] = "degenerate_mesh"
            q.put(out); return
        out["is_watertight"] = bool(m_pred.is_watertight)
        out["bbox_pred"] = np.sort(m_pred.extents).round(4).tolist()
        out["valid"] = out["is_valid"] and out["volume"] > VOL_TOL and out["is_watertight"]

        m_gt, _ = _load_gt_mesh(gt_step)
        out["bbox_gt"] = np.sort(m_gt.extents).round(4).tolist()
        out["bbox_err_mm"] = np.abs(
            np.sort(m_pred.extents) - np.sort(m_gt.extents)).round(4).tolist()

        out["iou"] = _voxel_iou(m_pred, m_gt, cfg["vox_res"], cfg["align"], False)
        if cfg["also_norm"]:
            out["iou_norm"] = _voxel_iou(m_pred, m_gt, cfg["vox_res"], cfg["align"], True)
        out["error_type"] = "ok"
    except Exception as e:
        et = type(e).__name__
        msg = str(e)
        for key in ("GC_MakeArcOfCircle", "Standard_ConstructionError",
                    "BRep_API", "StdFail_NotDone"):
            if key in msg:
                et = f"Kernel:{key}"
                break
        out["error_type"] = et
        out.setdefault("trace", traceback.format_exc()[-400:])
    q.put(out)


def eval_sample_isolated(sample_id, pred_code, gt_step, cfg):
    """Spawn a child process, enforce a timeout, terminate hangs."""
    q = Queue()
    p = Process(target=eval_one, args=(sample_id, pred_code, gt_step, cfg, q))
    p.start()
    p.join(cfg["timeout"])
    if p.is_alive():
        p.terminate(); p.join()
        return {"id": sample_id, "exec_ok": False, "has_result": False,
                "is_valid": False, "is_watertight": False, "volume": 0.0,
                "valid": False, "iou": None, "iou_norm": None,
                "bbox_pred": None, "bbox_gt": None, "bbox_err_mm": None,
                "error_type": "timeout"}
    if not q.empty():
        return q.get()
    return {"id": sample_id, "exec_ok": False, "has_result": False, "valid": False,
            "iou": None, "iou_norm": None, "bbox_err_mm": None,
            "error_type": "crash_no_result"}


# ---------------------------------------------------------------- aggregation
def aggregate(rows):
    n = len(rows)
    ious = [r["iou"] for r in rows if r["iou"] is not None]
    errs = [max(r["bbox_err_mm"]) for r in rows if r.get("bbox_err_mm")]
    valid = [r for r in rows if r["valid"]]
    iou_valid = [r["iou"] for r in valid if r["iou"] is not None]

    def _stat(xs, f, d=0.0):
        return round(float(f(xs)), 4) if xs else d
    agg = {
        "n": n,
        "exec_ok_rate": round(sum(r["exec_ok"] for r in rows) / n, 4) if n else 0,
        "has_result_rate": round(sum(r["has_result"] for r in rows) / n, 4) if n else 0,
        "valid_rate": round(len(valid) / n, 4) if n else 0,
        "iou_scored_n": len(ious),
        "mean_iou": _stat(ious, np.mean),
        "median_iou": _stat(ious, np.median),
        "mean_iou_valid_only": _stat(iou_valid, np.mean),
        "iou>=0.5_rate": round(sum(x >= 0.5 for x in ious) / n, 4) if n else 0,
        "iou>=0.9_rate": round(sum(x >= 0.9 for x in ious) / n, 4) if n else 0,
        "mean_max_bbox_err_mm": _stat(errs, np.mean),
        "median_max_bbox_err_mm": _stat(errs, np.median),
    }
    norms = [r["iou_norm"] for r in rows if r.get("iou_norm") is not None]
    if norms:
        agg["mean_iou_normalized"] = _stat(norms, np.mean)
    # error histogram
    hist = {}
    for r in rows:
        hist[r["error_type"]] = hist.get(r["error_type"], 0) + 1
    agg["error_hist"] = dict(sorted(hist.items(), key=lambda x: -x[1]))
    return agg


def read_ids(pred_dir, ids_file, limit):
    if ids_file:
        ids = [l.strip() for l in open(ids_file) if l.strip()]
    else:
        ids = sorted(f[:-3] for f in os.listdir(pred_dir) if f.endswith(".py"))
    if limit:
        ids = ids[:limit]
    return ids


def load_pred(pred_dir, sid):
    # accept "{id}.py" and cadrille-style "{id}+{k}.py"
    for name in (f"{sid}.py", f"{sid}.cadquery.py"):
        p = os.path.join(pred_dir, name)
        if os.path.exists(p):
            return open(p).read()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True, help="dir of predicted {id}.py")
    ap.add_argument("--gt-dir", required=True, help="dir of GT {id}.step")
    ap.add_argument("--ids-file", default=None, help="optional list of ids (one/line)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None, help="write full metrics JSON here")
    ap.add_argument("--vox-res", type=int, default=64, help="voxels across GT max extent")
    ap.add_argument("--align", choices=["min", "center"], default="min")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--no-norm", action="store_true", help="skip normalized-IoU cross-ref")
    args = ap.parse_args()

    cfg = {"vox_res": args.vox_res, "align": args.align,
           "timeout": args.timeout, "also_norm": not args.no_norm}
    ids = read_ids(args.pred_dir, args.ids_file, args.limit)
    rows, t0 = [], time.time()
    for i, sid in enumerate(ids):
        code = load_pred(args.pred_dir, sid)
        gt = os.path.join(args.gt_dir, f"{sid}.step")
        if code is None:
            rows.append({"id": sid, "exec_ok": False, "has_result": False,
                         "valid": False, "iou": None, "bbox_err_mm": None,
                         "error_type": "missing_pred"})
            continue
        if not os.path.exists(gt):
            rows.append({"id": sid, "exec_ok": False, "valid": False, "iou": None,
                         "bbox_err_mm": None, "error_type": "missing_gt_step"})
            continue
        rows.append(eval_sample_isolated(sid, code, gt, cfg))
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(ids)}] {time.time()-t0:.0f}s", file=sys.stderr)

    agg = aggregate(rows)
    print(json.dumps(agg, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"aggregate": agg, "config": vars(args), "rows": rows}, f, indent=2)
        print(f"\nwrote {args.out}", file=sys.stderr)
    return agg


if __name__ == "__main__":
    main()
