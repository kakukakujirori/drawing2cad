# Drawing2CAD

This is an R&D codebase for **2D engineering drawing → 3D CAD** conversion.

## Environments

NOTE: `CADQuery` requires `cadquery-ocp` whereas `build123d` requires `cadquery-ocp-novtk`, i.e., these two packages conflict with each other under normal installation (for the time being).

As a workaround, we install `build123d` with manually replacing the `cadquery-ocp-novtk` dependency with `cadquery-ocp`.

For that, note that the installed `cadquery-ocp` version in conda satisfies the version requirement of `build123d` in its `pyproject.toml`, and change its version if necessary.

```bash
conda create -y -n drawing2cad python=3.12 cadquery open3d ocp=7.9.3.1 pythonocc-core pyvista -c conda-forge
conda activate drawing2cad

# install build123d (v0.11.1 requires `cadquery-ocp-novtk >= 7.9, < 8.0`, so `ocp=7.9.3.1` satisfies it)
git clone --depth 1 -b v0.11.1 https://github.com/gumyr/build123d.git /tmp/build123d && \
  sed -i '/cadquery-ocp-novtk/d' /tmp/build123d/pyproject.toml && \
  pip install /tmp/build123d

# install other requirements
pip3 install torch torchvision xformers --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# Install CADGenBench
pip install "cadgenbench @ git+https://github.com/huggingface/cadgenbench.git@main" --no-deps --no-build-isolation

# ABI symbol resolution
conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
conda deactivate && conda activate drawing2cad

# Runtime verification
python -c "import cadquery as cq; s=cq.Workplane('XY').box(1,2,3).val(); assert s.isValid() and s.Volume() == 6"
python -c "import build123d as b; assert b.Box(1,2,3).volume == 6"

# install flash-attn
MAX_JOBS=4 pip3 install flash-attn --no-build-isolation
```

## Data Preparation

We use the [Zero-to-CAD](https://huggingface.co/datasets/ADSKAILab/Zero-To-CAD-1m/tree/main) dataset, which will be automatically downloaded when you run `select_zero_to_cad.py` in Step 1.

1. Randomly sample STEP files:
    ```bash
    # train
    python src/render/select_zero_to_cad.py \
      --n 100000 \
      --stage_dir data/z2c_train/target \
      --split train \
      --max-faces 400 \
      --stratify  # ensure uniform diversity in difficulty

    # val
    python src/render/select_zero_to_cad.py \
      --n 300 \
      --stage_dir data/z2c_val/target \
      --split validation \
      --max-faces 400 \
      --stratify
    ```

2. Generate 2D drawings and isomentric renderings:
    ```bash
    # train
    python src/render/render_dataset.py \
        --input_dir data/z2c_train/target \  # flat dir of {stem}.step ({stem}.cadquery.py ignored)
        --output_dir data/z2c_train/ \
        [--workers N] [--timeout 120] [--limit N] [--no-render3d] [--no-techdraw]

    # val
    python src/render/render_dataset.py \
        --input_dir data/z2c_val/target \
        --output_dir data/z2c_val/ \
        [--workers N] [--timeout 120] [--limit N] [--no-render3d] [--no-techdraw]
    ```

    The script outputs `techdraw/{svg,dxf,pdf}/{stem}.*`, `render_3d/<style>/{stem}.png`, and `manifest.jsonl`.

    Resume: re-running with the same OUTPUT_DIR skips finished parts; each part runs in a killable subprocess because OCC HLR can hang in native code.

    Calibration/verification harnesses: `src/render/calibrate_techdraw.py`, `src/render/calibrate_render3d.py`.

    Successful techdraw rows store layout metadata under `extra.techdraw`. View and cluster `bbox` values use `[xmin, ymin, xmax, ymax]` (`bbox_format="xyxy"`) in sheet-mm with a bottom-left origin. Manifests created before this metadata was added are refreshed on the next techdraw run; pass `--no-render3d` to avoid re-rendering the perspective images during that one-time migration.

## DXF + raster -> CadQuery SFT

```bash
python src/train_sft.py
```

Runs are written to `logs/train_sft/<yyyy-mm-dd_hh-mm-ss>/`.

Resume an interrupted run into the same directory with its original planned
step limit:

```bash
conda run -n drawing2cad python src/train_sft.py \
  training.resume_from_latest=true training.max_steps=<ORIGINAL_MAX_STEPS> \
  hydra.run.dir=logs/train_sft/<RUN_DIRECTORY>
```

Resume restores the saved scheduler as well as model/optimizer state.
Extending a run that already reached its original `max_steps` therefore requires an explicit new scheduler policy; merely increasing `max_steps` does not restart or stretch the completed schedule.

Enable W&B with `logger.wandb.enabled=true`.
