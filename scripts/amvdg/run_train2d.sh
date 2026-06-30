#!/usr/bin/env bash
# Train the 2D leg (drawing image -> AMVDG graph text): build bundle -> train -> infer.
# Mirrors the b2 wrapper style: assumes the `drawing2cad-ml` env is ACTIVE, calls bare
# `python`, hardcodes no machine paths. Run it from this directory.
#
#   conda activate drawing2cad-ml
#   export CADRILLE_REPO=/path/to/cadrille            # the col14m/cadrille checkout
#   DATASET=../../experiments/f360_drawings OUT=../../experiments/2d_v0 \
#     scripts/amvdg/run_train2d.sh
#
# Seed drawings come from the renderer (scripts/renderer/run_batch.sh over STEP parts).
# Fusion360 Gallery reconstruction set (27,958 STEP parts) is a good source:
#   https://fusion-360-gallery-dataset.s3.us-west-2.amazonaws.com/reconstruction/r1.0.1/r1.0.1.zip
set -euo pipefail
cd "$(dirname "$0")"

DATASET="${DATASET:-../../experiments/f360_drawings}"   # holds *.graph.json + *.png (+ *.scan.png)
BUNDLE="${BUNDLE:-../../experiments/data_bundle}"
OUT="${OUT:-../../experiments/2d_v0}"
INIT="${INIT:-cadrille}"                                 # cadrille | base
MAX_STEPS="${MAX_STEPS:-400}"
: "${CADRILLE_REPO:?set CADRILLE_REPO to the cadrille checkout (for 'from cadrille import ...')}"
export PYTHONPATH="$CADRILLE_REPO:$PWD"

echo "== 1/3 build bundle ($DATASET -> $BUNDLE) =="
python make_dataset.py --glob "$DATASET/*.graph.json" --out "$BUNDLE"

echo "== 2/3 train (init=$INIT, steps=$MAX_STEPS -> $OUT) =="
python train.py --data-root "$BUNDLE" --out "$OUT" --init "$INIT" --max-steps "$MAX_STEPS"

# batch_size=1: cadrille's collate can't batch images with differing patch counts
# (real drawings vary in aspect ratio); per-sample avoids the shape mismatch.
echo "== 3/3 infer (held-out val -> $OUT/pred) =="
python infer.py --data-root "$BUNDLE" --split-file val.jsonl \
  --checkpoint "$OUT/final" --out "$OUT/pred" --batch-size 1

echo "== done. score a prediction against GT with: =="
echo "   python score.py $OUT/pred/<part>__clean.pred.json $DATASET/<part>.graph.json"
