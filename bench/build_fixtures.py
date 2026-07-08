#!/usr/bin/env python3
"""Build the pseudo-CADGenBench fixture tree from a *fixture source set*.

A source set is a pair of dirs (see common.py): a STEP dir with GT
``{uuid}.step`` (+ ``{uuid}.cadquery.py``) and a graph dir with the rendered
``{uuid}.png`` (clean multi-view drawing) + ``{uuid}.graph.json`` AMVDG. This
script selects N parts present in BOTH and emits the cadgenbench fixture layout
under bench/data:

    data/inputs/<uuid>/description.yaml   # synthesized minimal task text
    data/inputs/<uuid>/drawing.png        # clean multi-view drawing (not scan)
    data/gt/<uuid>/ground_truth.step      # our GT STEP
    data/manifest.json                    # ids, seed, sources, faces distribution

Both candidate systems (ours + the LLM/agy baseline) run against this one tree.

The DEFAULT source set is the complexity-matched "hard" set
(experiments/{stage,dataset}_z2c_val_hard: Zero-To-CAD-1m validation, seed 0,
face band [90,400]) whose face-count distribution sits inside the real
CADGenBench IQR. Point at any other source set with --step-dir / --graph-dir.

The manifest records the achieved complexity: per-fixture B-rep face counts and
their distribution, computed via OCC (shelled to the cadgenbench venv). This
documents that the selected set is genuinely complexity-matched.

Usage:
    python bench/build_fixtures.py --clean            # N=50, seed 0, hard set
    python bench/build_fixtures.py -n 20 --seed 7
    python bench/build_fixtures.py --step-dir experiments/stage_z2c_val \\
        --graph-dir experiments/dataset_z2c_val    # legacy (easy) source set
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def build(n: int, seed: int, clean: bool, step_dir: Path, graph_dir: Path,
          with_faces: bool = True) -> dict:
    ids_all = C.eligible_ids(step_dir, graph_dir)
    if not ids_all:
        raise SystemExit(
            f"No eligible parts found under {step_dir} ∩ {graph_dir}"
        )
    if n > len(ids_all):
        print(f"[warn] requested {n} but only {len(ids_all)} eligible; using all.",
              file=sys.stderr)
        n = len(ids_all)

    rng = random.Random(seed)
    chosen = sorted(rng.sample(ids_all, n))

    if clean:
        shutil.rmtree(C.INPUTS_DIR, ignore_errors=True)
        shutil.rmtree(C.GT_DIR, ignore_errors=True)
    C.INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    C.GT_DIR.mkdir(parents=True, exist_ok=True)

    for uuid in chosen:
        in_dir = C.INPUTS_DIR / uuid
        gt_dir = C.GT_DIR / uuid
        in_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(graph_dir / f"{uuid}.png", in_dir / C.DRAWING_NAME)
        C.write_description_yaml(
            in_dir / C.DESC_NAME, C.TASK_DESCRIPTION, [C.DRAWING_NAME],
        )
        shutil.copy2(step_dir / f"{uuid}.step", gt_dir / C.GT_STEP_NAME)

    # Document achieved complexity: OCC face counts over the selected GT STEPs.
    face_counts: dict = {}
    faces_dist: dict = {}
    if with_faces:
        print(f"[build_fixtures] counting B-rep faces for {len(chosen)} parts "
              f"via {C.CGB_VENV_PY.name} ...", file=sys.stderr)
        face_counts = C.count_faces(
            {u: (C.GT_DIR / u / C.GT_STEP_NAME) for u in chosen})
        faces_dist = C.faces_distribution(face_counts)

    manifest = {
        "seed": seed,
        "n_requested": n,
        "n_selected": len(chosen),
        "n_eligible": len(ids_all),
        "ids": chosen,
        "sources": {
            "gt_step_dir": str(step_dir),
            "graph_png_dir": str(graph_dir),
            "dataset": "Zero-To-CAD-1m validation",
            "face_band": [90, 400] if step_dir == C.SRC_STEP_DIR else None,
            "note": ("complexity-matched hard set (faces inside real-CADGenBench "
                     "IQR)" if step_dir == C.SRC_STEP_DIR else "custom source set"),
        },
        "faces": {
            "distribution": faces_dist,
            "per_fixture": face_counts,
            "real_cadgenbench_reference": {"p25": 93, "median": 173, "p75": 373},
        },
        "fixture_layout": {
            "inputs": "data/inputs/<uuid>/{description.yaml, drawing.png}",
            "gt": "data/gt/<uuid>/ground_truth.step",
        },
        "task_description": C.TASK_DESCRIPTION,
    }
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    C.MANIFEST.write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--num", type=int, default=50,
                    help="number of fixtures to select (default 50)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (default 0)")
    ap.add_argument("--clean", action="store_true",
                    help="wipe data/inputs and data/gt before building")
    ap.add_argument("--step-dir", type=Path, default=C.SRC_STEP_DIR,
                    help=f"GT STEP source dir (default {C.SRC_STEP_DIR})")
    ap.add_argument("--graph-dir", type=Path, default=C.SRC_GRAPH_DIR,
                    help=f"drawing PNG + AMVDG graph source dir "
                         f"(default {C.SRC_GRAPH_DIR}); the clean {{uuid}}.png "
                         f"is used as the fixture input")
    ap.add_argument("--no-faces", action="store_true",
                    help="skip OCC face counting (faster; leaves distribution empty)")
    args = ap.parse_args()

    step_dir = args.step_dir if args.step_dir.is_absolute() else C.REPO / args.step_dir
    graph_dir = args.graph_dir if args.graph_dir.is_absolute() else C.REPO / args.graph_dir

    m = build(args.num, args.seed, args.clean, step_dir, graph_dir,
              with_faces=not args.no_faces)
    print(f"Built {m['n_selected']} fixtures (seed={m['seed']}, "
          f"eligible pool={m['n_eligible']}).")
    print(f"  sources: step={step_dir}")
    print(f"           graph={graph_dir}")
    fd = m["faces"]["distribution"]
    if fd.get("n"):
        print(f"  faces (n={fd['n']}): min={fd['min']} p25={fd['p25']} "
              f"median={fd['median']} p75={fd['p75']} max={fd['max']}  "
              f"(real CADGenBench: p25 93 / median 173 / p75 373)")
    print(f"  inputs: {C.INPUTS_DIR}")
    print(f"  gt:     {C.GT_DIR}")
    print(f"  manifest: {C.MANIFEST}")
    print("  first ids:", ", ".join(m["ids"][:3]), "...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
