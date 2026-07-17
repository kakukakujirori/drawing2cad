# Drawing2CAD

This is an R&D codebase for **2D engineering drawing → 3D CAD** conversion.

To developers: Research notes and the
running log live in [`research/`](research/) (start with `research/research-log_3d.md`).

## Environments

```bash
conda create -y -n drawing2cad python=3.12 pythonocc-core pyvista open3d -c conda-forge
conda activate drawing2cad

# install requirements
pip3 install torch torchvision xformers --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# ABI symbol resolution
conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
conda deactivate && conda activate drawing2cad

# Install CADGenBench
pip install "cadgenbench @ git+https://github.com/huggingface/cadgenbench.git@main" --no-deps --no-build-isolation

# install flash-attn
MAX_JOBS=4 pip3 install flash-attn --no-build-isolation
```

## Data Preparation

End-to-end, this turns existing 3D CAD (STEP) into the `(AMVDG graph JSON, 2D drawing PNG)` pairs, then bundles them into the SFT jsonl for `AMVDG -> 3D CAD` training.

We use the *Zero-to-CAD* dataset, which will be automatically downloaded when you run `select_zero_to_cad.py` in Step 1.

1. Randomly sample STEP files:
    ```bash
    # train
    python scripts/renderer/select_zero_to_cad.py \
      --n 100000 \
      --stage_dir experiments/stage_z2c_train \
      --split train \
      --max-faces 400 \
      --stratify  # ensure uniform diversity in difficulty

    # val
    python scripts/renderer/select_zero_to_cad.py \
      --n 300 \
      --stage_dir experiments/stage_z2c_val \
      --split validation \
      --max-faces 400 \
      --stratify
    ```

2. Generate AMVDG json and 2D drawings:
    ```bash
    # train
    python scripts/renderer/batch_dataset.py \
      --step_dir experiments/stage_z2c_train \
      --out_dir experiments/dataset_z2c_train

    # val
    python scripts/renderer/batch_dataset.py \
      --step_dir experiments/stage_z2c_val \
      --out_dir experiments/dataset_z2c_val
    ```
    For the AMVDG format, refer to [`spec/AMVDG_v0.3.md`](spec/AMVDG_v0.3.md).

3. Strip edge blends (fillet/chamfer) from the GT CadQuery (Note: AMVDG doesn't encode blends by default):
    ```bash
    # train
    python scripts/train3d/strip_blends.py \
      --in-dir experiments/stage_z2c_train \
      --out-dir experiments/stage_z2c_train_noblend

    # val
    python scripts/train3d/strip_blends.py \
      --in-dir experiments/stage_z2c_val \
      --out-dir experiments/stage_z2c_val_noblend
    ```

4. Bundle each `(AMVDG graph, GT CadQuery)` pair into an SFT jsonl. Coordinates use a signed canonical grid with 1024 magnitude bins per sign (`-1023..1023`) by default:
    ```bash
    # train
    python scripts/train3d/build_dataset.py \
      --graph-dir experiments/dataset_z2c_train \
      --code-dir  experiments/stage_z2c_train_noblend \
      --out experiments/data_z2c_train_noblend \
      --hid-dropout 0.95  # drop redundant hidden lines visible from other views

    # val
    python scripts/train3d/build_dataset.py \
      --graph-dir experiments/dataset_z2c_val \
      --code-dir  experiments/stage_z2c_val_noblend \
      --out experiments/data_z2c_val_noblend \
      --hid-dropout 0.95  # MAYBE BETTER TO SET 0?
    ```
    > **Consistency:** `--quant` sets the coordinate encoding, so `infer.py --quant` must equal what the training data used (both default 1024 → consistent by default; pass `--quant 0` on both to ablate). `stats.json` records the transforms and reports token coverage up to 16384.
    Each bundle is `all.jsonl` (one record per valid part: `input_text` = serialized graph, `target_code` = GT CadQuery) + `invalid_extent.jsonl` + `stats.json`. The builder keeps `extent_ratio <= 5`, warns for `2 < extent_ratio <= 5`, and quarantines larger values so one pathological HLR primitive cannot collapse the shared quantization grid; counts, IDs, thresholds, and the ratio distribution are recorded in `stats.json`. `build_dataset.py` calls the graph→text serializer [`scripts/train3d/serialize.py`](scripts/train3d/serialize.py); to eyeball / round-trip-check the exact text the model reads:
    ```bash
    # one graph -> the model-input text (pass any <uuid>.graph.json from Step 2's output):
    python scripts/train3d/serialize.py \
      --quant 1024 --drop-covered \
      "$(ls experiments/dataset_z2c_val/*.graph.json | head -1)"
    # whole dir -> round-trip + cross-view consistency check (quote the glob):
    python scripts/train3d/serialize.py \
      --quant 1024 --drop-covered --check \
      'experiments/dataset_z2c_val/*.graph.json'
    ```
    Existing bundles can be audited without rebuilding or loading the tokenizer. Add `--graph-dir` to identify the maximum-contributing source primitive and classify the failure mode:
    ```bash
    python scripts/train3d/audit_extent.py \
      --bundle experiments/data_z2c_train_noblend/all.jsonl \
      --graph-dir experiments/dataset_z2c_train \
      --warn-ratio 2 --fail-ratio 5 \
      --out experiments/data_z2c_train_noblend/extent_audit.json
    ```
    Training + eval then consume these two bundles — see [`scripts/train3d/README.md`](scripts/train3d/README.md).

**DEPRECATED**: You can also use the *Fusion360Gallery* `reconstruction` subset — `select_fusion360_recon.py` is specific to that
set's `{partid}_{hash}_{seq}_{substep}` naming (takes each part's final state, samples across parts):
```bash
python scripts/renderer/select_fusion360_recon.py /path/to/Fusion360Gallery/r1.0.1/reconstruction 2500 experiments/stage_recon

python scripts/renderer/batch_dataset.py experiments/stage_recon experiments/dataset_recon
```
A graph can be exported to DXF 2D-CAD with `python scripts/amvdg/graph_to_dxf.py GRAPH.json OUT.dxf`.

## AMVDG→3D leg (training — the current focus)

**Input = serialized-graph text only** (no drawing PNG is fed), so a working leg is direct evidence the AMVDG IR is *sufficient* to recover 3D. The training set (serialized graph → GT CadQuery jsonl) is built in **Data Preparation Step 3**.

**SFT harness** (`scripts/train3d/`) — SFT fine-tune (init:
`ADSKAILab/Zero-To-CAD-Qwen3-VL-2B`, text-only input) + isolated CadQuery execution eval (validity + translation-aligned voxel IoU + absolute-mm bbox error). See [`scripts/train3d/README.md`](scripts/train3d/README.md) for the train/eval **procedure**, and `research/research-log.md` for **measured results**.

**DEPRECATED**: **B2-cadrille baseline** (historical, 2026-06-15 — motivates the above; drawings → CadQuery → validity/scale metrics):
```bash
scripts/run_b2.sh --n 49          # or  IDS=101,103 scripts/run_b2.sh
```

## 2D→2D leg (drawing → AMVDG graph) — DE-PRIORITIZED (2026-07-02)

> **Status**: this stage (inferring AMVDG *from* a raster drawing) is **paused**. The plan
> is to first build the **AMVDG→3D leg** on clean synthetic-GT graphs (prove the IR is
> sufficient / measure the accuracy the 2D leg must hit), then return here. The OCR +
> line/arc-detection + OrthoSolve construction below is kept for reference but is **not** the
> current focus — and the cross-view correspondence (`topo_origins`/`features`) it must also
> predict needs deeper spatial reasoning than pixel detection. See `research/research-log.md`
> (2026-07-02 entry) for the decision and the open design questions. The **renderer**
> (synthetic-GT generator, above under Usage) stays active — it feeds both legs.

<!-- DE-PRIORITIZED — 2D→AMVDG inference stage (see the note above; kept for reference)

The raster-drawing → AMVDG-graph stage (design, methods and measured results in
`research/research-log.md`, 2026-06-20 entry):

- `spec/` — the AMVDG DSL: schema, validator, worked example. Start here.
- `scripts/train2d/` — the 2D→AMVDG-graph (cadrille) leg, mirroring `scripts/train3d/`:
  - `serialize.py` (`g1`, graph ⇔ compact-JSON training target, intra-view — drops `features[]`),
    `score.py` (predicted vs GT graph).
  - pipeline: `dataset.py` → `make_dataset.py` (build bundle) → `train.py` (fine-tune) →
    `infer.py`; `tile.py` tiles a high-res real drawing for inference; `run_train2d.sh`
    wraps bundle→train→infer (needs the `drawing2cad-ml` env + `CADRILLE_REPO`).
- `scripts/amvdg/` — general AMVDG format tooling: `graph_to_dxf.py` (graph → 2D-CAD DXF).
- `scripts/orthosolve/orthosolve_spike.py` — **OrthoSolve** proof-of-concept: instead of
  regressing pixels, recover exact metric for the dimensioned subset from topology + OCR'd
  dims via a constraint solve (pure-numpy, zero training).
- `scripts/detector/` — geometry detection: `circlenet.py` (center-heatmap + **log-radius** +
  **dilated big-RF backbone** that detects circles at all radii incl. large bores; train/eval on
  GPU below); `detect_hough.py` / `detect_contour.py` (classical-CV probes showing why
  thresholding fails on real drawings).
  ```bash
  CIRCLENET_SEED=0 python scripts/detector/circlenet.py train experiments/dataset_recon out_dir \
      --steps 6000 --tile 640 --bs 24
  # per-drawing recall/precision/radius-MAE + stratified-by-radius recall on the held-out split
  # (--scan for scan-aug; --thr 0.4 operating point; --dimunion unions diameter-dimension circles):
  python scripts/detector/circlenet.py evalfull out_dir/circlenet.pt experiments/dataset_recon \
      --split out_dir/split.json --thr 0.4 [--dimunion]
  ```
- `scripts/renderer/render_dataset.py` + `pipeline.py` — the synthetic-drawing renderer
  (ISO dims, title block, isometric, cross-view correspondences) with scan-noise augmentation;
  driven by `batch_dataset.py` and consumed by `circlenet.py`.
  **Phase 4 — Oracle Matching**: pure mathematical projection (`projector/` handlers →
  `CADProjector` + `GraphBuilder`) is matched against OCC TechDraw's HLR output, preserving
  exact 3D topological lineage (`prov.topo_origins`, `{dim,id,role}` objects) on visually
  occluded/segmented 2D lines; features then bind primitives ACROSS views by shared B-rep
  face ids. Projector local frames are verified against `projectEx` per view
  (`scripts/renderer/test_integration.py` + the per-module `__main__` unit tests).
- `research/{werk24-recon,orthosolve-method,foundation-model-survey,amvdg-v0.3-roadmap}.md`
  — the supporting surveys: Werk24 reverse-engineering, the OrthoSolve method note, the
  "does a big model solve geometry?" survey, and the v0.3 representation roadmap.

DE-PRIORITIZED section ends here -->

### Renderer output (active — feeds both legs)

> The renderer emits **v0.3** graphs at `profile: vectorized` (see the AMVDG section above
> for the v0.3 delta), and `python spec/validate_amvdg.py <graph>` passes all 7 gates on its
> output (schema accepts 0.2 and 0.3). The tooling
> (`serialize.py`/`score.py`/`graph_to_dxf.py`/`circlenet.py`/`tile.py`/`pipeline.py`) reads the
> latest v0.3/v0.2 names with a v0 fallback, so pre-existing v0 graphs still load until they are
> regenerated. NOTE: graphs rendered before 2026-07-02 predate the oracle-projector fixes
> (right-view frame, duplicate HLR edges, single-member features) — regenerate with
> `batch_dataset.py` before training on `topo_origins`/`features`.

Scripts that need fixtures not shipped here (the real drawing, GT graphs) take a `*_DATA`
env var pointing at a local data dir; the spikes print what they expect.


## The drawing DSL — AMVDG (`spec/`)

The 2D intermediate representation a drawing is parsed into before 3D reasoning is the
**Annotated Multi-View Drawing Graph (AMVDG)** — a 4-layer typed graph (geometry · annotation ·
cross-view correspondence · 3D provenance), mechanically validated by a 7-gate checker.
The current wire format is **v0.3** = the v0.2 schema plus:

- `prov.topo_origins` — per-primitive list of originating B-rep entities as
  `{dim, id, role}` objects (dim 2=Face / 1=Edge; role = edge · silhouette ·
  edge-on · parent_face · axis), computed by the oracle projector, so occlusion-split
  segments and edge-on-degenerate faces keep exact 3D lineage;
- `features[].members` spanning **multiple views** (derived from shared `topo_origins`
  face ids — the cross-view correspondence layer is provenance-exact, not radius-guessed);
- arc `start_angle`/`end_angle` (px frame, y-down, arc = start→end with increasing
  atan2 angle mod 360 — removes the major/minor ambiguity of endpoint-only arcs);
- centerline primitives (`line_role: center`) so every inked line is explained by the graph;
- `frame.axis_remap` per view (px axes ↔ signed model axes, e.g. front `px_x:+X, px_y:-Z`),
  and `projection_dir` = unit vector from part toward viewer (front `(0,-1,0)`,
  top `(0,0,1)`, right `(1,0,0)`, third-angle).

Files: [`spec/AMVDG_v0.3.md`](spec/AMVDG_v0.3.md) (prose spec — start here) ·
`spec/AMVDG_v0.3.schema.json` (accepts 0.2/0.3) · `spec/validate_amvdg.py` (7 gates) ·
[`spec/AMVDG_TUTORIAL.md`](spec/AMVDG_TUTORIAL.md) (日本語チュートリアル: JSONの読み方と2D復元).
Both the worked example and every renderer-emitted graph validate:
```bash
python spec/validate_amvdg.py spec/example_flange_v0.3.json          # all 7 gates PASS
python spec/validate_amvdg.py experiments/dataset_recon/XXX.graph.json
```
