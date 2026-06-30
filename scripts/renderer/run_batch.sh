#!/usr/bin/env bash
# Phase-1 synthetic dimensioned-drawing renderer (batch).
# seed B-rep STEPs -> per-part {graph.json (Tier-1/2), .png (Tier-0 raster), .dxf (Tier-1 2D CAD)}
#
# Assumes the `drawing2cad` conda env is ALREADY activated (see README.md):
#   conda activate drawing2cad
# Needs conda-forge FreeCAD + ezdxf importable by the active `python` (matplotlib
# ships with FreeCAD). The renderer scripts import FreeCAD via the `freecad` shim.
#
# Usage:   run_batch.sh <step_dir> <out_dir> [N=0:all]
# Example: run_batch.sh /path/to/Fusion360Gallery/r1.0.1/reconstruction out/drawings 2000
#
# Scale-out: shard <step_dir> across machines, or split by file ranges; each
# process is independent. HLR projection needs B-rep (STEP), NOT STL meshes.
set -euo pipefail

STEP_DIR="${1:?usage: run_batch.sh <step_dir> <out_dir> [N]}"
OUT_DIR="${2:?usage: run_batch.sh <step_dir> <out_dir> [N]}"
N="${3:-0}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$OUT_DIR"

echo "[stage1] STEP -> drawing-graph JSON (FreeCAD HLR)"
python "$HERE/project_views.py" --step-dir "$STEP_DIR" --n "$N" --out "$OUT_DIR"

echo "[stage2] JSON -> raster PNG + DXF (matplotlib/ezdxf)"
python "$HERE/render_drawing.py" --dir "$OUT_DIR"

echo "[done] $(ls "$OUT_DIR"/*.png 2>/dev/null | wc -l) drawings in $OUT_DIR"
