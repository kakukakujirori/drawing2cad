#!/usr/bin/env python3
"""Shared paths + helpers for the pseudo-CADGenBench harness (`bench/`).

Everything here is stdlib-only so every bench script runs under any Python 3.
The heavy work (inference, cadgenbench eval, the LLM baseline agent) is shelled
out to the *right* interpreter, never imported into these scripts.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# --- Repo / bench layout ---------------------------------------------------
BENCH_DIR = Path(__file__).resolve().parent
REPO = BENCH_DIR.parent

DATA_DIR = BENCH_DIR / "data"
INPUTS_DIR = DATA_DIR / "inputs"
GT_DIR = DATA_DIR / "gt"
RESULTS_DIR = BENCH_DIR / "results"
MANIFEST = DATA_DIR / "manifest.json"

# --- Source data ("fixture source set") ------------------------------------
# A fixture source set is a pair of dirs:
#   step_dir : GT STEP + GT CadQuery, one {uuid}.step (+ {uuid}.cadquery.py) each.
#   graph_dir: rendered drawings + AMVDG graphs, {uuid}.png + {uuid}.graph.json
#              (+ {uuid}.svg / {uuid}.scan.png). The clean {uuid}.png is used as
#              the fixture drawing (NOT the scan-augmented {uuid}.scan.png).
# The PNG dir is the same as the graph dir (they are co-located by the renderer).
#
# DEFAULT = the complexity-matched set (Zero-To-CAD-1m validation, seed 0,
# --stratify + real-quantile --bin-edges, see stage_dir's own _select_params.json
# for the exact params used), built to track the real-CADGenBench face-count
# distribution rather than just "hard" examples. build_fixtures.py exposes
# --step-dir / --graph-dir to point at any source set.
SRC_STEP_DIR = REPO / "experiments" / "stage_z2c_bench"     # {uuid}.step, {uuid}.cadquery.py
SRC_GRAPH_DIR = REPO / "experiments" / "dataset_z2c_bench"  # {uuid}.png, {uuid}.graph.json, ...

# Legacy source set (uniform-random Z2C-val sample, faces median ~29 — too easy;
# superseded by the hard set above). Kept for reference / reproducibility.
SRC_STEP_DIR_LEGACY = REPO / "experiments" / "stage_z2c_val"
SRC_GRAPH_DIR_LEGACY = REPO / "experiments" / "dataset_z2c_val"

# --- Interpreters / entry points -------------------------------------------
INFER_PY = REPO / "scripts" / "train3d" / "infer.py"

# Fixture-contract file names (mirrors cadgenbench).
GT_STEP_NAME = "ground_truth.step"           # data/gt/<uuid>/ground_truth.step
DRAWING_NAME = "drawing.png"                  # data/inputs/<uuid>/drawing.png
DESC_NAME = "description.yaml"               # data/inputs/<uuid>/description.yaml

# Honest, minimal task description handed to the LLM baseline (we have no
# authored natural-language spec; the drawing PNG carries the real signal).
TASK_DESCRIPTION = (
    "Reconstruct the 3D CAD part shown in this multi-view engineering drawing. "
    "The image contains standard orthographic projections (and possibly section "
    "or detail views) with dimensions given in millimetres. Build a single solid "
    "whose geometry and dimensions match the drawing as closely as you can."
)


def eligible_ids(step_dir: Path, graph_dir: Path) -> list[str]:
    """uuids present as a GT STEP *and* a rendered drawing PNG *and* AMVDG graph.

    The drawing PNG is the clean ``{uuid}.png`` (the scan-augmented
    ``{uuid}.scan.png`` is excluded).
    """
    steps = {p.name[: -len(".step")] for p in step_dir.glob("*.step")}
    graphs = {p.name[: -len(".graph.json")] for p in graph_dir.glob("*.graph.json")}
    pngs = {p.name[: -len(".png")]
            for p in graph_dir.glob("*.png") if not p.name.endswith(".scan.png")}
    return sorted(steps & graphs & pngs)


# Snippet run under the cadgenbench venv (which ships OCP/OpenCASCADE) to count
# B-rep faces per STEP file. Kept out-of-process so build_fixtures.py stays
# stdlib-only and works under any python3.
_FACE_COUNT_SNIPPET = r"""
import json, sys
from OCP.STEPControl import STEPControl_Reader
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE

def nfaces(path):
    r = STEPControl_Reader()
    if int(r.ReadFile(path)) != 1:        # IFSelect_RetDone == 1
        return None
    r.TransferRoots()
    shp = r.OneShape()
    exp = TopExp_Explorer(shp, TopAbs_FACE)
    c = 0
    while exp.More():
        c += 1
        exp.Next()
    return c

out = {}
for uuid, path in json.load(sys.stdin).items():
    try:
        out[uuid] = nfaces(path)
    except Exception as e:
        out[uuid] = None
json.dump(out, sys.stdout)
"""


def count_faces(step_paths: dict[str, Path]) -> dict[str, int | None]:
    """Count B-rep faces for each {uuid: step_path} by shelling to the cgb venv.

    Returns {uuid: n_faces or None}. On any failure (missing venv, OCP import
    error) returns all-None so callers can degrade gracefully.
    """
    import subprocess
    payload = json.dumps({u: str(p) for u, p in step_paths.items()})
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _FACE_COUNT_SNIPPET],
            input=payload, capture_output=True, text=True, timeout=1800,
        )
        if proc.returncode != 0:
            return {u: None for u in step_paths}
        return json.loads(proc.stdout)
    except Exception:
        return {u: None for u in step_paths}


def faces_distribution(face_counts: dict[str, int | None]) -> dict:
    """Summary stats (n, min/p25/median/p75/max) over non-None face counts."""
    vals = sorted(v for v in face_counts.values() if isinstance(v, int))
    n = len(vals)
    if n == 0:
        return {"n": 0}

    def q(frac: float) -> int:
        # nearest-rank percentile (matches how the pilot reported quartiles)
        idx = min(n - 1, max(0, int(round(frac * (n - 1)))))
        return vals[idx]

    return {
        "n": n,
        "min": vals[0],
        "p25": q(0.25),
        "median": q(0.50),
        "p75": q(0.75),
        "max": vals[-1],
    }


def load_manifest() -> dict:
    if not MANIFEST.exists():
        raise SystemExit(
            f"No manifest at {MANIFEST}. Run build_fixtures.py first."
        )
    return json.loads(MANIFEST.read_text())


def manifest_ids() -> list[str]:
    return list(load_manifest()["ids"])


def cgb_env() -> dict:
    """Environment for any cadgenbench subprocess: point it at bench/data."""
    env = dict(os.environ)
    env["CADGENBENCH_DATA_DIR"] = str(DATA_DIR)
    # backend=cadquery: the .venv itself has cadquery, so the agent's code
    # runner uses its own interpreter. Pin it explicitly to be safe.
    env["CADGENBENCH_CADQUERY_PYTHON"] = sys.executable
    return env


def write_description_yaml(path: Path, description: str, input_files: list[str],
                           task_type: str = "generation") -> None:
    """Write a cadgenbench fixture description.yaml.

    Uses a YAML block scalar for the description so colons / punctuation in the
    text never need escaping. Parsed by cadgenbench via yaml.safe_load.
    """
    lines = ["description: |"]
    for ln in description.splitlines() or [""]:
        lines.append(f"  {ln}")
    lines.append(f"task_type: {task_type}")
    lines.append("input_files:")
    for f in input_files:
        lines.append(f"  - {f}")
    path.write_text("\n".join(lines) + "\n")
