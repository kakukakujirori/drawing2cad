#!/usr/bin/env bash
# Minimal cadrille INFERENCE env (no pytorch3d/open3d rendering — we feed our own images).
# Matches cadrille's pins: transformers==4.50.3, qwen-vl-utils==0.0.10. + cadquery for executing generated code.
set -eo pipefail
source /home/ryotaro/miniforge3/etc/profile.d/conda.sh

if ! conda env list | grep -qE '/envs/cadrille$'; then
  echo "[1/4] creating env cadrille (python 3.11)"
  conda create -y -n cadrille python=3.11
fi
conda activate cadrille
python -V

echo "[2/4] torch 2.5.1 cu124"
pip install --no-input torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

echo "[3/4] transformers stack (cadrille pins)"
pip install --no-input "transformers==4.50.3" "qwen-vl-utils==0.0.10" accelerate safetensors einops "numpy<2" pillow

echo "[4/4] cadquery (for executing generated CadQuery -> STEP)"
pip install --no-input cadquery || echo "WARN: cadquery install failed (inference still usable)"

echo "DONE: cadrille env ready"
python -c "import torch,transformers,qwen_vl_utils; print('torch',torch.__version__,'tf',transformers.__version__,'cuda',torch.cuda.is_available())"
