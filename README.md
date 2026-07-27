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
conda env config vars set LD_PRELOAD=$CONDA_PREFIX/lib/libjpeg.so
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

    The script outputs `techdraw/{svg,dxf,pdf}/{stem}.*` and `render_3d/<style>/{stem}.png`.

    Resume: re-running with the same OUTPUT_DIR skips finished parts; each part runs in a killable subprocess because OCC HLR can hang in native code.

    Calibration/verification harnesses: `src/data/render/calibrate_techdraw.py`, `src/data/render/calibrate_render3d.py`.

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
- `src/data/render/render_dataset.py` writes each view's geometry onto its own DXF layer (`front`/`top`/`right`); that layer is how the loader assigns every primitive to a view. If your dataset already ships three-view techdraw DXFs whose geometry sits on a single layer (e.g. real SolidWorks drawings on layer `0`), stamp them into the layered format with:
    ```bash
    python src/data/render/reformat_techdraw.py --techdraw_dir data/[my_dataset]/techdraw
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

## Evaluation metrics

Validation scoring is a list of metric families in `configs/train_sft.yaml`:

```yaml
evaluation:
  metrics:
    - CadExecutionMetric
    - VoxelIoUMetric
    - BoundingBoxMetric
    # - ChamferMetric
    # - CADGenBenchScoreMetric
    # - ECCVChallengeMetric
```

Each entry names a class in `src/metrics/` (registered by class name; a mapping
with `name` plus that family's parameters configures it). Every family declares
which artifacts it reads, and only those are produced, so an unused family costs
nothing. Adding a metric means adding one file, not editing the evaluator.

| Family | Reports |
| --- | --- |
| `CadExecutionMetric` | execution, result and validity rates |
| `VoxelIoUMetric` | shape-only voxel IoU (the default checkpoint monitor) |
| `BoundingBoxMetric` | absolute and target-relative bounding-box errors |
| `ChamferMetric` | surface Chamfer distance, normalized and in mm² |
| `CADGenBenchScoreMetric` | [CADGenBench](https://huggingface.co/spaces/HuggingAI4Engineering/CADGenBench) CAD Score, surface distance F1, volume IoU, topology match |
| `ECCVChallengeMetric` | [ECCV 2026 CAD Challenge](https://huggingface.co/spaces/jingwei-xu-00/eccv2026-cad-challenge) valid ratio, surface/edge/vertex F1, topology F1, summary |

The last two read B-Rep (STEP) rather than meshes and cost seconds to tens of
seconds per sample, so they are off during training and on by default in
`src/test.py`. Both are shape-only by default: `normalize_to_gt_bbox` rescales
the prediction onto the target's bounding box because the drawings carry no
dimensions. Set it to `false` for numbers directly comparable to the published
leaderboards. All scoring runs in an isolated subprocess, so a CAD kernel fault
or hang cannot take down training.

## Inference

```bash
python src/test.py \
    --ckpt logs/train_sft/[yyyy-mm-dd_hh-mm-ss]/checkpoints/latest \
    --test_dir data/z2c_val \
    --out_dir outputs/z2c_val/[yyyy-mm-dd_hh-mm-ss]
```

Scores with every metric family; `--metrics VoxelIoUMetric BoundingBoxMetric`
restricts that, and `--no_metrics` skips scoring entirely.