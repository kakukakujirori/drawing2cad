#!/usr/bin/env python3
"""Our route: GT AMVDG graph -> trained model -> CadQuery -> STEP.

For the manifest ids, symlink the selected {uuid}.graph.json into a scratch
dir, run scripts/train3d/infer.py (drawing2cad conda env, GPU1) once over that
dir, then arrange the produced {uuid}.step into the cadgenbench results layout:

    results/<run_name>/<uuid>/output.step      (+ candidate.py for debugging)

Using the GT AMVDG graph is intentional: it measures the ceiling of our route
"if the 2D leg were perfect". Fixtures whose inference fails to produce a STEP
are left without output.step, which the evaluator scores as status=missing / 0.

Usage:
    python bench/run_ours.py                        # all manifest ids
    python bench/run_ours.py --limit 2              # smoke: first 2 ids
    python bench/run_ours.py --ids <uuid> <uuid>
    python bench/run_ours.py --run-name ours_noblend_v1 --gpu 1
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def select_ids(args) -> list[str]:
    ids = args.ids or C.manifest_ids()
    if args.limit is not None:
        ids = ids[: args.limit]
    return ids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, default=C.DEFAULT_CKPT,
                    help=f"LoRA adapter dir (default {C.DEFAULT_CKPT})")
    ap.add_argument("--run-name", default="ours_noblend_v1",
                    help="results/<run_name>/ subdir (default ours_noblend_v1)")
    ap.add_argument("--ids", nargs="*", default=None,
                    help="explicit uuids (default: all manifest ids)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N manifest ids (smoke tests)")
    ap.add_argument("--gpu", default="1",
                    help="CUDA_VISIBLE_DEVICES for inference (default 1)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--keep-preds", action="store_true",
                    help="keep the raw infer.py output dir (default: temp)")
    args = ap.parse_args()

    ids = select_ids(args)
    if not ids:
        raise SystemExit("No ids selected.")
    if not args.ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {args.ckpt}")

    run_dir = C.RESULTS_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1) scratch dir of symlinks to just the selected graphs
    graph_link_dir = Path(tempfile.mkdtemp(prefix="bench_graphs_"))
    for uuid in ids:
        src = C.SRC_GRAPH_DIR / f"{uuid}.graph.json"
        if not src.exists():
            print(f"[warn] missing graph for {uuid}, skipping", file=sys.stderr)
            continue
        (graph_link_dir / f"{uuid}.graph.json").symlink_to(src)

    # 2) run infer.py in the drawing2cad env on GPU <gpu>
    preds_dir = (run_dir / "_preds") if args.keep_preds else Path(
        tempfile.mkdtemp(prefix="bench_preds_"))
    preds_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    cmd = [
        str(C.D2C_PY), str(C.INFER_PY),
        "--ckpt", str(args.ckpt),
        "--input", str(graph_link_dir),
        "--out", str(preds_dir),
        "--batch-size", str(args.batch_size),
    ]
    print("[run_ours] $", " ".join(cmd), f"(CUDA_VISIBLE_DEVICES={args.gpu})")
    rc = subprocess.run(cmd, env=env).returncode
    if rc != 0:
        print(f"[run_ours] infer.py exited {rc}", file=sys.stderr)

    # 3) arrange into results/<run_name>/<uuid>/output.step
    n_step = 0
    per_fixture = {}
    for uuid in ids:
        fx = run_dir / uuid
        fx.mkdir(parents=True, exist_ok=True)
        step = preds_dir / f"{uuid}.step"
        py = preds_dir / f"{uuid}.py"
        if step.exists() and step.stat().st_size > 0:
            shutil.copy2(step, fx / "output.step")
            n_step += 1
            per_fixture[uuid] = "step"
        else:
            per_fixture[uuid] = "missing"
        if py.exists():
            shutil.copy2(py, fx / "candidate.py")

    summary_src = preds_dir / "infer_summary.json"
    run_meta = {
        "run_name": args.run_name,
        "system": "ours",
        "ckpt": str(args.ckpt),
        "n_fixtures": len(ids),
        "n_output_step": n_step,
        "per_fixture": per_fixture,
    }
    if summary_src.exists():
        try:
            run_meta["infer_summary"] = json.loads(summary_src.read_text())
        except Exception:
            pass
    (run_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2))

    # cleanup scratch
    shutil.rmtree(graph_link_dir, ignore_errors=True)
    if not args.keep_preds:
        shutil.rmtree(preds_dir, ignore_errors=True)

    print(f"[run_ours] {n_step}/{len(ids)} fixtures produced output.step -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
