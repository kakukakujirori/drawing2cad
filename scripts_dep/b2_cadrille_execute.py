#!/usr/bin/env python
"""B2 experiment (execute step): run the CadQuery code produced by cadrille,
export STEP + STL, and report validity proxies. No GT here (CADGenBench GT is
private), so we report: executed_ok / watertight / volume / bbox dims. These
plus the drawing's stated dimensions are the diagnostic for the distribution break.

cadrille convention (from its evaluate.py): exec the code, result is in global `r`,
`r.val()` -> compound. Each exec runs in a separate process with a timeout (CadQuery
leaks/hangs). We mirror that.

  PYTHONPATH unneeded. Run in the `cadrille` env (has cadquery + trimesh):
    python scripts/b2_cadrille_execute.py --dir experiments/b2_cadrille/whole268
"""
import os, json, glob, argparse, traceback
from multiprocessing import Process, Queue


def _worker(py_path, step_path, stl_path, q):
    rec = {"executed_ok": False, "exported_step": False, "n_faces": 0,
           "watertight": False, "volume": None, "bbox": None, "err": None}
    try:
        import cadquery as cq
        import trimesh
        with open(py_path) as f:
            code = f.read()
        g = {}
        exec(code, g)                      # cadrille puts the result in `r`
        compound = g["r"].val()
        rec["executed_ok"] = True
        try:
            cq.exporters.export(compound, step_path)
            rec["exported_step"] = os.path.isfile(step_path)
        except Exception as e:
            rec["err"] = f"step_export: {e}"
        verts, faces = compound.tessellate(0.001, 0.1)
        mesh = trimesh.Trimesh([(v.x, v.y, v.z) for v in verts], faces)
        rec["n_faces"] = int(len(mesh.faces))
        rec["watertight"] = bool(mesh.is_watertight)
        rec["volume"] = float(mesh.volume) if mesh.is_volume else None
        b = mesh.bounds
        rec["bbox"] = [round(float(x), 3) for x in (b[1] - b[0])]
        if rec["n_faces"] > 2:
            mesh.export(stl_path)
    except Exception as e:
        rec["err"] = f"{type(e).__name__}: {e}"
        rec["tb"] = traceback.format_exc()[-500:]
    q.put(rec)


def run_one(py_path, step_path, stl_path, timeout=15):
    q = Queue()
    p = Process(target=_worker, args=(py_path, step_path, stl_path, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate(); p.join()
        return {"executed_ok": False, "err": f"timeout>{timeout}s"}
    return q.get() if not q.empty() else {"executed_ok": False, "err": "no result (crash)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="dir with <id>.py from generation step")
    ap.add_argument("--timeout", type=int, default=15)
    args = ap.parse_args()

    step_dir = os.path.join(args.dir, "step"); os.makedirs(step_dir, exist_ok=True)
    stl_dir = os.path.join(args.dir, "stl"); os.makedirs(stl_dir, exist_ok=True)

    pys = sorted(glob.glob(os.path.join(args.dir, "*.py")))
    out = []
    for py in pys:
        fid = os.path.splitext(os.path.basename(py))[0]
        rec = run_one(py, os.path.join(step_dir, f"{fid}.step"),
                      os.path.join(stl_dir, f"{fid}.stl"), timeout=args.timeout)
        rec["id"] = fid
        out.append(rec)
        print(f"[exec] {fid}: ok={rec.get('executed_ok')} watertight={rec.get('watertight')} "
              f"vol={rec.get('volume')} bbox={rec.get('bbox')} err={rec.get('err')}")

    with open(os.path.join(args.dir, "exec_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    n = len(out)
    ok = sum(r.get("executed_ok") for r in out)
    wt = sum(bool(r.get("watertight")) for r in out)
    print(f"[exec] {ok}/{n} executed, {wt}/{n} watertight -> {args.dir}/exec_summary.json")


if __name__ == "__main__":
    main()
