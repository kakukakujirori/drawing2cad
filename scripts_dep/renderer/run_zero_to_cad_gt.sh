#!/usr/bin/env bash
# Zero-To-CAD -> AMVDG GT, end to end. Two envs: select needs `datasets` (py312),
# batch_dataset needs FreeCAD (drawing2cad). Override paths/N via env vars.
#   N=300 SPLIT=validation scripts/renderer/run_zero_to_cad_gt.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

N="${N:-300}"
SPLIT="${SPLIT:-validation}"
STAGE="${STAGE:-experiments/stage_z2c}"
OUT="${OUT:-experiments/dataset_z2c}"
PY_DATASETS="${PY_DATASETS:-/home/ryotaro/miniforge3/envs/py312/bin/python}"
PY_FREECAD="${PY_FREECAD:-/home/ryotaro/miniforge3/envs/drawing2cad/bin/python}"

echo "[1/2] select $N Zero-To-CAD STEP seeds ($SPLIT) -> $STAGE"
"$PY_DATASETS" scripts/renderer/select_zero_to_cad.py "$N" "$STAGE" --split "$SPLIT"

echo "[2/2] render -> $OUT"
PYTHONPATH="$(pwd)" "$PY_FREECAD" scripts/renderer/batch_dataset.py "$STAGE" "$OUT"

echo "DONE: $(ls "$OUT"/*.graph.json 2>/dev/null | wc -l) graphs in $OUT"
