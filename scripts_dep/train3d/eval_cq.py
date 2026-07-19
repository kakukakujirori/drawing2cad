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
  7. **Chamfer Distance**: `cd_mm` (absolute mm, bbox-min aligned like the voxel IoU —
     the primary CD) and `cd_norm` (cadrille/CAD-Recode unit-box-normalized CD, for
     cross-paper comparability), both via the exact sampling+formula in cadrille's
     evaluate.py `compute_chamfer_distance` (n=8192 surface points, cKDTree bidirectional
     nearest-neighbour, sum of mean squared distances).

Usage:
  # score a directory of predicted {id}.py against GT {id}.step:
  python eval_cq.py --pred-dir PREDS --gt-dir experiments/stage_z2c --out metrics.json
  # GT-code self-IoU smoke (predictions ARE the GT code -> expect IoU ~1.0):
  python eval_cq.py --pred-dir experiments/stage_z2c --gt-dir experiments/stage_z2c \
                    --ids-file paired_uuids.txt --limit 8 --out self_iou.json
"""

import os

# Pin numeric libs to 1 thread BEFORE numpy loads. Each isolated worker is CPU-bound
# (numpy/trimesh voxelization); with N parallel workers, multi-threaded BLAS oversubscribes
# the cores, slows every sample, and pushes borderline ones past the wall-clock timeout —
# which silently changes valid_rate vs a sequential run. One thread/worker keeps per-sample
# time ≈ standalone, so parallel scoring reproduces the sequential numbers exactly.
for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")

# ruff: noqa: E402
import sys
import json
import time
import argparse
import traceback
from multiprocessing import Process, Queue

import numpy as np

TESS_TOL = 0.1  # OCC linear tessellation tolerance (mm)
VOL_TOL = 1e-6  # min volume to count as a non-degenerate solid


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


def _load_gt_mesh(step_path: str):
    import cadquery as cq

    solid = cq.importers.importStep(step_path).val()
    return _mesh_from_shape(solid), solid


# ---------------------------------------------------------------- voxel IoU
def _voxel_set(mesh, pitch, origin):
    """Occupancy as a set of integer voxel coords in a grid anchored at `origin`."""
    vg = mesh.voxelized(pitch)
    try:
        vg = vg.fill()  # fill interior so solids are solid, not shells
    except Exception:
        pass
    pts = vg.points  # world-space centres of occupied cells
    idx = np.floor((pts - origin) / pitch + 1e-6).astype(np.int64)
    return set(map(tuple, idx.tolist()))


def _aligned_origin(mesh, mode):
    if mode == "center":
        return (mesh.bounds[0] + mesh.bounds[1]) / 2.0
    return mesh.bounds[0]  # "min"


def _align_min(mesh):
    """Copy shifted so bbox-min sits at the origin (absolute-mm alignment convention
    shared with the voxel IoU's align="min" mode)."""
    m = mesh.copy()
    m.apply_translation(-m.bounds[0])
    return m


def _normalize_unit(mesh):
    """Copy centred at the origin and rescaled to max-extent 1 (the cadrille /
    CAD-Recode unit-box convention: evaluate.py's run_cd_single centres+rescales the
    predicted mesh before sampling; eval_Fusion360.py does the same to both meshes)."""
    m = mesh.copy()
    m.apply_translation(-(m.bounds[0] + m.bounds[1]) / 2.0)
    e = float(np.max(m.extents))
    if e > 1e-9:
        m.apply_scale(1.0 / e)
    return m


def _nn_sq_dist(query, ref, chunk=1024):
    """Chunked brute-force squared nearest-neighbour distance (fallback for when scipy
    isn't installed). O(len(query) * len(ref)) but fine for a few thousand points."""
    out = np.empty(len(query))
    ref = np.asarray(ref)
    for i in range(0, len(query), chunk):
        d2 = np.sum((query[i : i + chunk, None, :] - ref[None, :, :]) ** 2, axis=-1)
        out[i : i + chunk] = d2.min(axis=1)
    return out


def _chamfer_distance(m_pred, m_gt, n_points, seed, normalize):
    """cadrille/CAD-Recode Chamfer Distance, computed on the meshes already in hand
    (no re-tessellation). Sampling + formula follow evaluate.py / evaluate_new.py
    `compute_chamfer_distance` and eval_Fusion360.py's inline CD block exactly:
      - trimesh.sample.sample_surface (area-weighted uniform surface sampling),
        n_points per mesh, same fixed integer seed passed to both calls,
      - bidirectional nearest neighbour via a KD-tree (scipy.spatial.cKDTree; falls
        back to a chunked numpy brute-force search if scipy is unavailable),
      - CD = mean(nn_dist_pred_to_gt^2) + mean(nn_dist_gt_to_pred^2)  (sum, not
        average, of the two mean-squared-distance terms — no extra unit scaling;
        cad-recode's eval_Fusion360.py only applies a x1000 factor at the *display*
        layer, the underlying compute_chamfer_distance() returns this raw value).
    `normalize=True` reproduces their unit-box rescale (centre + divide by max
    extent) before sampling; `normalize=False` instead uses this project's
    absolute-mm bbox-min alignment (matching the voxel IoU's "min" mode) so the
    result stays in real millimetres.
    """
    import trimesh

    p = _normalize_unit(m_pred) if normalize else _align_min(m_pred)
    g = _normalize_unit(m_gt) if normalize else _align_min(m_gt)
    pred_pts, _ = trimesh.sample.sample_surface(p, n_points, seed=seed)
    gt_pts, _ = trimesh.sample.sample_surface(g, n_points, seed=seed)
    pred_pts = np.asarray(pred_pts)
    gt_pts = np.asarray(gt_pts)
    try:
        from scipy.spatial import cKDTree

        gt_nn, _ = cKDTree(gt_pts).query(pred_pts, k=1)
        pred_nn, _ = cKDTree(pred_pts).query(gt_pts, k=1)
        gt_nn_sq, pred_nn_sq = gt_nn**2, pred_nn**2
    except ImportError:
        gt_nn_sq = _nn_sq_dist(pred_pts, gt_pts)
        pred_nn_sq = _nn_sq_dist(gt_pts, pred_pts)
    return float(np.mean(gt_nn_sq) + np.mean(pred_nn_sq))


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
    out = {
        "id": sample_id,
        "exec_ok": False,
        "has_result": False,
        "is_valid": False,
        "is_watertight": False,
        "volume": 0.0,
        "valid": False,
        "iou": None,
        "iou_norm": None,
        "cd_mm": None,
        "cd_norm": None,
        "bbox_pred": None,
        "bbox_gt": None,
        "bbox_err_mm": None,
        "error_type": None,
    }
    try:
        ns = {"__name__": "__main__"}
        try:
            import cadquery as cq

            ns["cq"] = cq  # some model outputs assume `cq` exists
        except Exception:
            pass
        exec(pred_code, ns)
        out["exec_ok"] = True

        shape = _shape_from_globals(ns)
        if shape is None:
            out["error_type"] = "no_result_object"
            q.put(out)
            return
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
            q.put(out)
            return
        out["is_watertight"] = bool(m_pred.is_watertight)
        out["bbox_pred"] = np.sort(m_pred.extents).round(4).tolist()
        out["valid"] = (
            out["is_valid"] and out["volume"] > VOL_TOL and out["is_watertight"]
        )

        m_gt, _ = _load_gt_mesh(gt_step)
        out["bbox_gt"] = np.sort(m_gt.extents).round(4).tolist()
        out["bbox_err_mm"] = (
            np.abs(np.sort(m_pred.extents) - np.sort(m_gt.extents)).round(4).tolist()
        )

        out["iou"] = _voxel_iou(m_pred, m_gt, cfg["vox_res"], cfg["align"], False)
        if cfg["also_norm"]:
            out["iou_norm"] = _voxel_iou(
                m_pred, m_gt, cfg["vox_res"], cfg["align"], True
            )

        if cfg.get("also_cd", True):
            # CD failures (e.g. degenerate normalization) must not clobber the metrics
            # already computed above, so they're isolated in their own try/except.
            try:
                out["cd_mm"] = _chamfer_distance(
                    m_pred, m_gt, cfg["cd_points"], cfg["cd_seed"], normalize=False
                )
            except Exception:
                out["cd_mm"] = None
            if cfg["also_norm"]:
                try:
                    out["cd_norm"] = _chamfer_distance(
                        m_pred, m_gt, cfg["cd_points"], cfg["cd_seed"], normalize=True
                    )
                except Exception:
                    out["cd_norm"] = None
        out["error_type"] = "ok"
    except Exception as e:
        et = type(e).__name__
        msg = str(e)
        for key in (
            "GC_MakeArcOfCircle",
            "Standard_ConstructionError",
            "BRep_API",
            "StdFail_NotDone",
        ):
            if key in msg:
                et = f"Kernel:{key}"
                break
        out["error_type"] = et
        out.setdefault("trace", traceback.format_exc()[-400:])
    q.put(out)


def _timeout_row(sample_id):
    return {
        "id": sample_id,
        "exec_ok": False,
        "has_result": False,
        "is_valid": False,
        "is_watertight": False,
        "volume": 0.0,
        "valid": False,
        "iou": None,
        "iou_norm": None,
        "cd_mm": None,
        "cd_norm": None,
        "bbox_pred": None,
        "bbox_gt": None,
        "bbox_err_mm": None,
        "error_type": "timeout",
    }


def eval_sample_isolated(sample_id, pred_code, gt_step, cfg):
    """Spawn a child process, enforce a timeout, terminate hangs."""
    q = Queue()
    p = Process(target=eval_one, args=(sample_id, pred_code, gt_step, cfg, q))
    p.start()
    p.join(cfg["timeout"])
    if p.is_alive():
        p.terminate()
        p.join()
        return _timeout_row(sample_id)
    if not q.empty():
        return q.get()
    return {
        "id": sample_id,
        "exec_ok": False,
        "has_result": False,
        "valid": False,
        "iou": None,
        "iou_norm": None,
        "bbox_err_mm": None,
        "error_type": "crash_no_result",
    }


def imap_isolated(tasks, worker, timeout, workers):
    """Run `worker` in isolated child processes with bounded concurrency + a per-task
    wall-clock timeout. `tasks` is an iterable of (key, args_tuple); each child is
    `Process(target=worker, args=args_tuple + (q,))` and must put exactly one result dict
    on its queue `q`. Yields (key, result_or_None) as each finishes — None means the task
    timed out or crashed without a result (the caller supplies the fallback row).

    We manage the processes ourselves (rather than ProcessPoolExecutor) because CadQuery/OCC
    can hang in C code, and only a parent-side terminate() reliably kills a hung child — the
    same containment `eval_sample_isolated` gives, now `workers`-wide."""
    workers = max(1, workers)
    tasks = list(tasks)
    n, i = len(tasks), 0
    live = []  # [key, proc, queue, start_time]
    while i < n or live:
        while i < n and len(live) < workers:
            key, args = tasks[i]
            i += 1
            q = Queue()
            p = Process(target=worker, args=tuple(args) + (q,))
            p.start()
            live.append([key, p, q, time.time()])
        nxt = []
        for item in live:
            key, p, q, t0 = item
            if not p.is_alive():
                try:
                    res = q.get(timeout=1.0)
                except Exception:
                    res = None
                p.join()
                yield key, res
            elif time.time() - t0 > timeout:
                p.terminate()
                p.join()
                yield key, None
            else:
                nxt.append(item)
        live = nxt
        if live and (len(live) >= workers or i >= n):
            time.sleep(0.02)


# ---------------------------------------------------------------- aggregation
def aggregate(rows):
    # rows can include hand-built early-return dicts (missing_pred / missing_gt_step in
    # main(), below) alongside eval_one's/_ timeout_row's full schema -- .get() throughout
    # so an incomplete row degrades gracefully (counts as false/unscored) instead of
    # KeyError-ing the whole aggregation (which, in the in-training eval, would drop that
    # eval's periodic metrics; see train_sft.py's compute_metrics / GenEvalTrainer).
    n = len(rows)
    ious = [r.get("iou") for r in rows if r.get("iou") is not None]
    errs = [max(r["bbox_err_mm"]) for r in rows if r.get("bbox_err_mm")]
    valid = [r for r in rows if r.get("valid")]
    iou_valid = [r.get("iou") for r in valid if r.get("iou") is not None]
    cds_mm = [r["cd_mm"] for r in rows if r.get("cd_mm") is not None]
    cds_norm = [r["cd_norm"] for r in rows if r.get("cd_norm") is not None]

    def _stat(xs, f, d=0.0):
        return round(float(f(xs)), 4) if xs else d

    agg = {
        "n": n,
        "exec_ok_rate": round(sum(bool(r.get("exec_ok")) for r in rows) / n, 4)
        if n
        else 0,
        "has_result_rate": round(sum(bool(r.get("has_result")) for r in rows) / n, 4)
        if n
        else 0,
        "valid_rate": round(len(valid) / n, 4) if n else 0,
        "iou_scored_n": len(ious),
        "mean_iou": _stat(ious, np.mean),
        "median_iou": _stat(ious, np.median),
        "mean_iou_valid_only": _stat(iou_valid, np.mean),
        "iou>=0.5_rate": round(sum(x >= 0.5 for x in ious) / n, 4) if n else 0,
        "iou>=0.9_rate": round(sum(x >= 0.9 for x in ious) / n, 4) if n else 0,
        "mean_max_bbox_err_mm": _stat(errs, np.mean),
        "median_max_bbox_err_mm": _stat(errs, np.median),
        "cd_mm_scored_n": len(cds_mm),
        "mean_cd_mm": _stat(cds_mm, np.mean),
        "median_cd_mm": _stat(cds_mm, np.median),
    }
    norms = [r["iou_norm"] for r in rows if r.get("iou_norm") is not None]
    if norms:
        agg["mean_iou_normalized"] = _stat(norms, np.mean)
    if cds_norm:
        agg["cd_norm_scored_n"] = len(cds_norm)
        agg["mean_cd_norm"] = _stat(cds_norm, np.mean)
        agg["median_cd_norm"] = _stat(cds_norm, np.median)
    # error histogram
    hist = {}
    for r in rows:
        hist[r["error_type"]] = hist.get(r["error_type"], 0) + 1
    agg["error_hist"] = dict(sorted(hist.items(), key=lambda x: -x[1]))
    return agg


def read_ids(pred_dir, ids_file, limit):
    if ids_file:
        ids = [line.strip() for line in open(ids_file) if line.strip()]
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", required=True, help="dir of predicted {id}.py")
    parser.add_argument("--gt-dir", required=True, help="dir of GT {id}.step")
    parser.add_argument(
        "--ids-file", default=None, help="optional list of ids (one/line)"
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default=None, help="write full metrics JSON here")
    parser.add_argument(
        "--vox-res", type=int, default=64, help="voxels across GT max extent"
    )
    parser.add_argument("--align", choices=["min", "center"], default="min")
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="per-sample wall-clock cap (s). Higher than the old 30 s: workers are "
        "pinned to 1 thread (no BLAS oversubscription), so heavy 64³ voxelization "
        "runs single-threaded and needs headroom — this keeps results invariant "
        "to --workers (a genuine hang is still caught, amortized across workers).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="parallel exec/score workers (0 = auto = min(8, cpu_count))",
    )
    parser.add_argument(
        "--no-norm", action="store_true", help="skip normalized-IoU cross-ref"
    )
    parser.add_argument(
        "--no-cd", action="store_true", help="skip Chamfer Distance scoring"
    )
    parser.add_argument(
        "--cd-points",
        type=int,
        default=8192,
        help="surface points sampled per mesh for CD (cadrille/CAD-Recode default)",
    )
    parser.add_argument(
        "--cd-seed",
        type=int,
        default=0,
        help="fixed seed for CD's trimesh.sample.sample_surface calls",
    )
    args = parser.parse_args()

    workers = args.workers or min(8, os.cpu_count() or 1)
    cfg = {
        "vox_res": args.vox_res,
        "align": args.align,
        "timeout": args.timeout,
        "also_norm": not args.no_norm,
        "also_cd": not args.no_cd,
        "cd_points": args.cd_points,
        "cd_seed": args.cd_seed,
    }
    ids = read_ids(args.pred_dir, args.ids_file, args.limit)

    # split into schedulable tasks (each an isolated exec+score) and pre-resolved error rows
    rows, tasks = [], []
    for sid in ids:
        code = load_pred(args.pred_dir, sid)
        gt = os.path.join(args.gt_dir, f"{sid}.step")
        if code is None:
            rows.append(
                {
                    "id": sid,
                    "exec_ok": False,
                    "has_result": False,
                    "valid": False,
                    "iou": None,
                    "bbox_err_mm": None,
                    "error_type": "missing_pred",
                }
            )
        elif not os.path.exists(gt):
            rows.append(
                {
                    "id": sid,
                    "exec_ok": False,
                    "has_result": False,
                    "valid": False,
                    "iou": None,
                    "bbox_err_mm": None,
                    "error_type": "missing_gt_step",
                }
            )
        else:
            tasks.append((sid, (sid, code, gt, cfg)))

    print(f"scoring {len(tasks)} samples with {workers} workers", file=sys.stderr)
    t0, done = time.time(), 0
    for sid, res in imap_isolated(tasks, eval_one, args.timeout, workers):
        rows.append(res if res is not None else _timeout_row(sid))
        done += 1
        if done % 20 == 0:
            print(f"  [{done}/{len(tasks)}] {time.time() - t0:.0f}s", file=sys.stderr)
    rows.sort(
        key=lambda r: r["id"]
    )  # deterministic order (pool completes out of order)

    agg = aggregate(rows)
    print(json.dumps(agg, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                {"aggregate": agg, "config": vars(args), "rows": rows}, f, indent=2
            )
        print(f"\nwrote {args.out}", file=sys.stderr)
    return agg


if __name__ == "__main__":
    main()
