#!/usr/bin/env bash
# B2-cadrille end-to-end: generate (drawing -> CadQuery) -> execute -> analyze.
#
# Assumes the `drawing2cad-ml` conda env is ALREADY activated:
#   conda activate drawing2cad-ml
#
# Env vars (all optional, with defaults):
#   CADRILLE_REPO   path to the cadrille model-code clone   (default ~/github/cadrille)
#   FEED            whole268 | whole_native                 (default whole268)
#   IDS             comma-separated fixture ids             (default: use --n)
#   OUT             output root                             (default <repo>/experiments/b2_cadrille)
#   CGB_DATA_DIR    CADGenBench fixtures dir                (default: HF cache via huggingface_hub)
#   CUDA_VISIBLE_DEVICES                                    (e.g. 1)
# Extra args (e.g. --n 49) are forwarded to the generate step.
#
# Usage: scripts/run_b2.sh --n 49      |      IDS=101,103 scripts/run_b2.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

CADRILLE_REPO="${CADRILLE_REPO:-$HOME/github/cadrille}"
export PYTHONPATH="${CADRILLE_REPO}${PYTHONPATH:+:$PYTHONPATH}"

FEED="${FEED:-whole268}"
OUT="${OUT:-$ROOT/experiments/b2_cadrille}"

id_args=()
[ -n "${IDS:-}" ] && id_args=(--ids "$IDS")
data_args=()
[ -n "${CGB_DATA_DIR:-}" ] && data_args=(--data-dir "$CGB_DATA_DIR")

python "$HERE/b2_cadrille_generate.py" --feed "$FEED" --out "$OUT" "${id_args[@]}" "${data_args[@]}" "$@"
python "$HERE/b2_cadrille_execute.py" --dir "$OUT/$FEED"
python "$HERE/b2_analyze.py" "$OUT/$FEED"
