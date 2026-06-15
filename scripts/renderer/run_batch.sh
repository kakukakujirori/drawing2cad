#!/usr/bin/env bash
# Phase-1 synthetic dimensioned-drawing renderer (batch).
# seed B-rep STEPs -> per-part {graph.json (Tier-1/2), .png (Tier-0 raster), .dxf (Tier-1 2D CAD)}
#
# Usage: scripts/renderer/run_batch.sh <step_dir> <out_dir> [N=0:all]
# Example (Fusion360, already on /disk2):
#   scripts/renderer/run_batch.sh /disk2/Fusion360Gallery/r1.0.1/reconstruction out/f360_drawings 2000
#
# Scale-out: shard <step_dir> across machines/processes, or split by file ranges;
# each process is independent. HLR projection needs B-rep (STEP), NOT STL meshes.
set -eo pipefail
STEP_DIR="${1:?usage: run_batch.sh <step_dir> <out_dir> [N]}"
OUT_DIR="${2:?usage: run_batch.sh <step_dir> <out_dir> [N]}"
N="${3:-0}"
cd "$(dirname "$0")/../.."
FC_LIB=/home/ryotaro/github/FreeCAD/build/release/lib
REND_PY=/home/ryotaro/miniforge3/envs/cadrille/bin/python
mkdir -p "$OUT_DIR"

echo "[stage1] STEP -> drawing-graph JSON (FreeCAD HLR)"
PYTHONPATH=$FC_LIB conda run -n freecad python scripts/renderer/project_views.py \
    --step-dir "$STEP_DIR" --n "$N" --out "$OUT_DIR"

echo "[stage2] JSON -> raster PNG + DXF (matplotlib/ezdxf)"
"$REND_PY" scripts/renderer/render_drawing.py --dir "$OUT_DIR"

echo "[done] $(ls "$OUT_DIR"/*.png 2>/dev/null | wc -l) drawings in $OUT_DIR"
