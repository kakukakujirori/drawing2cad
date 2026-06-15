#!/usr/bin/env bash
# B2-cadrille end-to-end: generate (2 feed strategies) -> execute -> validity.
set -eo pipefail
cd /home/ryotaro/my_works/drawing2cad
PY=/home/ryotaro/miniforge3/envs/cadrille/bin/python
export PYTHONPATH=/home/ryotaro/github/cadrille
export CUDA_VISIBLE_DEVICES=1          # GPU0 busy; use the free A5000
IDS="${IDS:-101,103,119,120,121}"

echo "===== generate whole268 ====="
$PY scripts/b2_cadrille_generate.py --feed whole268 --ids "$IDS"
echo "===== generate whole_native ====="
$PY scripts/b2_cadrille_generate.py --feed whole_native --ids "$IDS"
echo "===== execute whole268 ====="
$PY scripts/b2_cadrille_execute.py --dir experiments/b2_cadrille/whole268
echo "===== execute whole_native ====="
$PY scripts/b2_cadrille_execute.py --dir experiments/b2_cadrille/whole_native
echo "===== B2 DONE ====="
