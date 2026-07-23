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
    python src/data/render/select_zero_to_cad.py \
      --n 100000 \
      --stage_dir data/z2c_train/target \
      --split train \
      --max-faces 400 \
      --stratify  # ensure uniform diversity in difficulty

    # val
    python src/data/render/select_zero_to_cad.py \
      --n 300 \
      --stage_dir data/z2c_val/target \
      --split validation \
      --max-faces 400 \
      --stratify
    ```

2. Audit GT data:
    ```bash
    # train
    python src/data/audit/gt_audit.py \
        --stage-dir data/z2c_train/target \
        --out-dir data/z2c_train/target_audit \
        --workers 30 \
        --task-timeout-s 45

    # val
    python src/data/audit/gt_audit.py \
        --stage-dir data/z2c_val/target \
        --out-dir data/z2c_val/target_audit \
        --workers 30 \
        --task-timeout-s 45
    ```

    This generates `data/z2c_{train,val}/target_audit/results.json`. Based on this audit result and `configs/data/z2c.yaml` audit config, `Drawing2CADDataset` sieves training data samples.

3. Generate 2D drawings and isomentric renderings:
    ```bash
    # train
    python src/data/render/render_dataset.py \
        --input_dir data/z2c_train/target \
        --output_dir data/z2c_train/ \
        [--workers N] [--timeout 120] [--limit N] [--no-render3d] [--no-techdraw]

    # val
    python src/data/render/render_dataset.py \
        --input_dir data/z2c_val/target \
        --output_dir data/z2c_val/ \
        [--workers N] [--timeout 120] [--limit N] [--no-render3d] [--no-techdraw]
    ```

    The script outputs `techdraw/{svg,dxf,pdf}/{stem}.*`, `render_3d/<style>/{stem}.png`, and `manifest.jsonl`.

    Resume: re-running with the same OUTPUT_DIR skips finished parts; each part runs in a killable subprocess because OCC HLR can hang in native code.

    Calibration/verification harnesses: `src/data/render/calibrate_techdraw.py`, `src/data/render/calibrate_render3d.py`.

    Successful techdraw rows store layout metadata under `extra.techdraw`. View and cluster `bbox` values use `[xmin, ymin, xmax, ymax]` (`bbox_format="xyxy"`) in sheet-mm with a bottom-left origin. Manifests created before this metadata was added are refreshed on the next techdraw run; pass `--no-render3d` to avoid re-rendering the perspective images during that one-time migration.

### When using other datasets

- For a GT split that ships only `{uuid}.step` with no generating `{uuid}.cadquery.py` (e.g. a STEP-only eval set), add `--no-cadquery` to run the STEP-only subset of checks — see `src/data/audit/README.md` for details. Without this flag, `.step` files with zero matching `.cadquery.py` raise instead of silently auditing nothing.
    ```bash
    python src/data/audit/gt_audit.py \
        --stage-dir data/[my_dataset]/target \
        --out-dir data/[my_dataset]/target_audit \
        --workers 30 \
        --task-timeout-s 45 \
        --no-cadquery
    ```
- `src/data/render/render_dataset.py` generates `manifest.jsonl`, which specifies the bboxes of front/top/right view positions. If your dataset already prepares three-view techdraws, then you need to generate `manifest.jsonl` by the following command:
    ```bash
    python src/data/render/manifest_from_techdraw.py --data_dir data/[my_dataset]/
    ```

## DXF + raster -> CadQuery SFT

Specify the number of GPUs at `--num_processes`:

```bash
accelerate launch --num_processes 2 src/train_sft.py
```

Runs are written to `logs/train_sft/<yyyy-mm-dd_hh-mm-ss>/`.

Resume an interrupted run into the same directory with its original planned
step limit:

```bash
python src/train_sft.py training.resume_from_latest=true \
    hydra.run.dir=logs/train_sft/<RUN_DIRECTORY>
```

Resume restores the saved scheduler as well as model/optimizer state.
Extending a run that already reached its original `max_steps` therefore requires an explicit new scheduler policy; merely increasing `max_steps` does not restart or stretch the completed schedule.

## Inference

```bash
python src/test.py \
    --ckpt logs/train_sft/[yyyy-mm-dd_hh-mm-ss]/checkpoints/latest \
    --test_dir data/z2c_val \
    --out_dir outputs/z2c_val/[yyyy-mm-dd_hh-mm-ss]
```