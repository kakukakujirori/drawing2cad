#!/usr/bin/env python3
"""LLM baseline: CADGenBench's own reference agent, pointed at a local ollama
vision model, input = the rendered drawing PNG.

Thin wrapper around `cadgenbench baseline run` that:
  - points the fixture loader at bench/data (CADGENBENCH_DATA_DIR),
  - drives a local ollama model via LiteLLM (--model ollama_chat/<name>),
  - forces backend=cadquery so both systems use the same OCC kernel (fairness;
    cadgenbench's default build123d would confound the shape comparison),
  - bounds max-iter / max-duration / max-tokens so a run cannot hang forever,
  - runs under xvfb (the box is headless; VTK 9.3 needs an X server for the
    agent's per-turn iso-render feedback).

Writes results/<timestamp>_<model_slug>/<uuid>/output.step (cadgenbench layout).

Usage:
    python bench/run_baseline.py --limit 1                      # smoke
    python bench/run_baseline.py --ids <uuid>
    python bench/run_baseline.py --all --model ollama_chat/qwen2.5vl:32b
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import subprocess  # noqa: E402

DEFAULT_MODEL = "ollama_chat/qwen2.5vl:32b"
# A private ollama server pinned to the free GPU (see README): the shared
# systemd ollama on :11434 multi-GPU-spreads a 21 GB model onto the busy GPU0
# and offloads to CPU (unusably slow). Pointing LiteLLM here forces single-GPU.
DEFAULT_OLLAMA_BASE = "http://127.0.0.1:11435"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"LiteLLM model string (default {DEFAULT_MODEL})")
    ap.add_argument("--ids", nargs="*", default=None,
                    help="explicit fixture uuids to run")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of fixtures (smoke tests)")
    ap.add_argument("--all", action="store_true",
                    help="run every fixture in data/inputs")
    ap.add_argument("--backend", default="cadquery",
                    choices=["cadquery", "build123d", "openscad"],
                    help="geometry kernel (default cadquery, for fairness vs ours)")
    ap.add_argument("--max-iter", type=int, default=12,
                    help="agent turn cap (default 12)")
    ap.add_argument("--max-duration", type=float, default=900.0,
                    help="per-fixture wall-clock seconds (default 900)")
    ap.add_argument("--max-tokens", type=int, default=200000,
                    help="global token budget per fixture (default 200000)")
    ap.add_argument("--max-tokens-per-call", type=int, default=8192,
                    help="completion tokens per LLM call (default 8192)")
    ap.add_argument("--runner-timeout", type=int, default=120,
                    help="per-script exec timeout, seconds (default 120)")
    ap.add_argument("--llm-timeout", type=float, default=600.0,
                    help="per-LLM-call timeout, seconds (default 600)")
    ap.add_argument("--parallel", type=int, default=1,
                    help="fixtures in parallel (keep 1: single local GPU)")
    ap.add_argument("--no-xvfb", action="store_true",
                    help="do not wrap in xvfb-run (only if a display exists)")
    ap.add_argument("--ollama-base", default=DEFAULT_OLLAMA_BASE,
                    help=f"OLLAMA_API_BASE for LiteLLM (default {DEFAULT_OLLAMA_BASE}, "
                         "the GPU-pinned private server)")
    args = ap.parse_args()

    if not C.INPUTS_DIR.exists():
        raise SystemExit(f"No fixtures at {C.INPUTS_DIR}. Run build_fixtures.py.")

    ids = args.ids
    if ids is None and not args.all:
        ids = C.manifest_ids()
        if args.limit is not None:
            ids = ids[: args.limit]

    cgb = [
        str(C.CGB_VENV_PY), "-m", "cadgenbench.cli", "baseline", "run",
        "--backend", args.backend,
        "--model", args.model,
        "--max-iter", str(args.max_iter),
        "--max-duration", str(args.max_duration),
        "--max-tokens", str(args.max_tokens),
        "--max-tokens-per-call", str(args.max_tokens_per_call),
        "--timeout", str(args.runner_timeout),
        "--llm-timeout", str(args.llm_timeout),
        "--parallel", str(args.parallel),
        "--output-dir", str(C.RESULTS_DIR),
    ]
    if args.all:
        cgb.append("--all")
    else:
        cgb += list(ids)
    if args.limit is not None and args.all:
        cgb += ["--limit", str(args.limit)]

    cmd = cgb
    if not args.no_xvfb:
        xvfb = shutil.which("xvfb-run")
        if xvfb is None:
            print("[warn] xvfb-run not found; rendering feedback may segfault.",
                  file=sys.stderr)
        else:
            cmd = [xvfb, "-a"] + cgb

    env = C.cgb_env()
    if args.ollama_base:
        env["OLLAMA_API_BASE"] = args.ollama_base

    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("[run_baseline] $", " ".join(cmd))
    print(f"[run_baseline] CADGENBENCH_DATA_DIR={C.DATA_DIR} "
          f"OLLAMA_API_BASE={env.get('OLLAMA_API_BASE')}")
    rc = subprocess.run(cmd, env=env).returncode
    print(f"[run_baseline] cadgenbench exited {rc}")
    print(f"[run_baseline] results under {C.RESULTS_DIR} "
          f"(newest <timestamp>_<model> dir is this run)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
