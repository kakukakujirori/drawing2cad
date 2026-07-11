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
    parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", type=Path, required=True, help=f"LoRA adapter dir")
    parser.add_argument("--run-name", type=str, required=True, help="results/<run_name>/ subdir")
    parser.add_argument("--quant", type=int, required=True,
                        help="coordinate quantization passed to infer.py; MUST match the "
                            "checkpoint's training data (see the run's stats.json/config.json). "
                            "build_dataset's default is 1024; use 0 only for a pre-quant ckpt. "
                            "Required on purpose — a wrong value silently corrupts coordinates.")
    parser.add_argument("--ids", nargs="*", default=None, help="explicit uuids (default: all manifest ids)")
    parser.add_argument("--limit", type=int, default=None, help="only the first N manifest ids (smoke tests)")
    parser.add_argument("--gpu", type=str, default="0,1",
                    help="CUDA_VISIBLE_DEVICES for inference. Pass e.g. '0,1' to let infer.py "
                         "shard the model across both GPUs for prompts too long for one (the "
                         "few very complex parts); a single GPU records those as oom_generate.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--graph-dir", type=Path, default=None,
                    help="AMVDG graph source dir (default: common.SRC_GRAPH_DIR). "
                         "Override for alternate benchmark sets, e.g. "
                         "experiments/dataset_z2c_bench_new.")
    parser.add_argument("--keep-preds", action="store_true",
                    help="keep the raw infer.py output dir (default: temp)")
    args = parser.parse_args()

    ids = select_ids(args)
    if not ids:
        raise SystemExit("No ids selected.")
    if not args.ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {args.ckpt}")

    run_dir = C.RESULTS_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1) scratch dir of symlinks to just the selected graphs
    graph_src = args.graph_dir or C.SRC_GRAPH_DIR
    if not graph_src.is_absolute():
        graph_src = C.REPO / graph_src
    graph_link_dir = Path(tempfile.mkdtemp(prefix="bench_graphs_"))
    for uuid in ids:
        src = graph_src / f"{uuid}.graph.json"
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
        sys.executable, str(C.INFER_PY),
        "--ckpt", str(args.ckpt),
        "--input", str(graph_link_dir),
        "--out", str(preds_dir),
        "--batch-size", str(args.batch_size),
        "--quant", str(args.quant),
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

    # surface infer.py's error histogram so failures aren't buried in run_meta.json
    infer_hist = (run_meta.get("infer_summary") or {}).get("summary", {}).get("error_hist", {})
    if infer_hist:
        top = ", ".join(f"{k}={v}" for k, v in list(infer_hist.items())[:4])
        print(f"[run_ours] infer error histogram: {top}", file=sys.stderr)

    print(f"[run_ours] {n_step}/{len(ids)} fixtures produced output.step -> {run_dir}")

    if n_step == 0 and ids:
        # A total wipeout is a hard failure, NOT a benign empty result. Exit non-zero
        # and point at the real diagnostics (infer.py already printed the trace above).
        print("=" * 72, file=sys.stderr)
        print(f"[run_ours] 0/{len(ids)} fixtures produced a STEP -> hard failure.",
              file=sys.stderr)
        print(f"[run_ours] real error+trace: infer.py's stderr above; generated code: "
              f"{run_dir}/<uuid>/candidate.py; summary: {run_dir}/run_meta.json",
              file=sys.stderr)
        print("=" * 72, file=sys.stderr)
        return 1
    return rc if rc != 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
