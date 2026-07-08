#!/usr/bin/env python3
"""Score a results dir with CADGenBench's own evaluator against bench/data/gt.

Wraps `cadgenbench evaluate <run_dir>`, which recomputes metrics for every
fixture directory under the run (aligning the candidate STEP to GT, then
shape-similarity + topology + interface), writing per-fixture result.json and a
run-level run_summary.json.

CAD Score (generation) = weighted mean over available axes
    shape 0.4 / interface 0.4 / topology 0.2, renormalized over present axes.
Our fixtures have NO interface jig sub-volumes, so only shape + topology are
present => effective weights shape 2/3, topology 1/3. Invalid/missing => 0.

Runs under xvfb (headless box; the evaluator renders candidate previews).

Usage:
    python bench/evaluate.py results/ours_noblend_v1
    python bench/evaluate.py results/20260705_xxxx_qwen2.5vl:32b --workers 4
    python bench/evaluate.py results/*/                      # several at once
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+", type=Path,
                    help="results/<run_name>/ directories to (re)score")
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel fixtures (default 4; 1 = sequential)")
    ap.add_argument("--force", action="store_true",
                    help="realign + re-render even if cached artefacts are fresh")
    ap.add_argument("--no-xvfb", action="store_true")
    args = ap.parse_args()

    run_dirs = [d if d.is_absolute() else (C.REPO / d) for d in args.run_dirs]
    for d in run_dirs:
        if not d.is_dir():
            raise SystemExit(f"Not a directory: {d}")

    cgb = [
        str(C.CGB_VENV_PY), "-m", "cadgenbench.cli", "evaluate",
        *[str(d) for d in run_dirs],
        "--workers", str(args.workers),
    ]
    if args.force:
        cgb.append("--force")

    cmd = cgb
    if not args.no_xvfb:
        xvfb = shutil.which("xvfb-run")
        if xvfb is not None:
            cmd = [xvfb, "-a"] + cgb

    print("[evaluate] $", " ".join(cmd))
    print(f"[evaluate] CADGENBENCH_DATA_DIR={C.DATA_DIR}")
    rc = subprocess.run(cmd, env=C.cgb_env()).returncode
    print(f"[evaluate] cadgenbench evaluate exited {rc}")
    for d in run_dirs:
        rs = d / "run_summary.json"
        if rs.exists():
            print(f"[evaluate] wrote {rs}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
