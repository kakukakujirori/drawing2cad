# Drawing2CAD

R&D for **2D engineering drawing → 3D CAD** conversion. Research notes and the
running log live in [`research/`](research/) (start with `research/research-log.md`).

## The drawing DSL — AMVDG (`spec/`)

The 2D intermediate representation a drawing is parsed into before 3D reasoning is the
**Annotated Multi-View Drawing Graph (AMVDG)** — a 4-layer typed graph (geometry · annotation ·
cross-view correspondence · 3D provenance), mechanically validated by a 7-gate checker.
See [`spec/README.md`](spec/README.md). It validates out of the box:
```bash
python spec/validate_amvdg.py spec/example_flange_v0.2.json   # all 7 gates PASS
```

## Environments

```bash
conda create -y -n drawing2cad python=3.11 freecad cadquery -c conda-forge
conda activate drawing2cad

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

**Note**: conda-forge ships `FreeCAD.so` under `$CONDA_PREFIX/lib`, which is not on `sys.path`, so a bare `import FreeCAD` fails — the renderer scripts therefore `import freecad` first (a conda-forge shim that adds that lib dir), then `import FreeCAD`. No `PYTHONPATH` or `freecadcmd` is needed.

## Usage

**Synthetic dimensioned-drawing renderer** (seed B-rep STEP → PNG + graph JSON).
`batch_dataset.py` runs the FreeCAD HLR render + cairosvg rasterize + scan-aug in one env
and emits the `(PNG, graph.json)` pairs `scripts/detector/circlenet.py` consumes. From the
Fusion360Gallery **`reconstruction`** subset (B-rep STEP solids), stage a diverse sample
(one final state per part, sampled across parts) and render (`select_fusion360_recon.py`
is specific to that subset's `{partid}_{hash}_{seq}_{substep}` naming):
```bash
# Sample Fusion360 STEP files
python scripts/renderer/select_fusion360_recon.py /path/to/Fusion360Gallery/r1.0.1/reconstruction 2500 experiments/stage_recon

# Generate orthographic views
python scripts/renderer/batch_dataset.py experiments/stage_recon experiments/dataset_recon
```
Seed 3D data must be **B-rep (STEP)** — STL meshes are unsuitable for HLR. Good sources:
Fusion360Gallery (STEP), Zero-To-CAD (`ADSKAILab/Zero-To-CAD-1m`, Apache-2.0; CadQuery code
→ convert to STEP first). A graph can be exported to DXF 2D-CAD with
`python scripts/amvdg/graph_to_dxf.py GRAPH.json OUT.dxf`.

**B2-cadrille baseline** (CADGenBench drawings → CadQuery → validity/scale metrics):
```bash
scripts/run_b2.sh --n 49          # or  IDS=101,103 scripts/run_b2.sh
```
Output under `experiments/` (git-ignored). See `research/research-log.md` for results.

## 2D→2D leg (drawing → AMVDG graph)

The raster-drawing → AMVDG-graph stage (design, methods and measured results in
`research/research-log.md`, 2026-06-20 entry):

- `spec/` — the AMVDG DSL: schema, validator, worked example. Start here.
- `scripts/amvdg/` — the graph format + extractor pipeline:
  - format tooling: `serialize.py` (graph ⇔ compact-JSON training target), `graph_to_dxf.py`
    (graph → 2D-CAD DXF), `score.py` (predicted vs GT graph).
  - cadrille 2D→graph pipeline: `dataset.py` → `make_dataset.py` (build bundle) → `train.py`
    (fine-tune) → `infer.py`; `tile.py` tiles a high-res real drawing for inference;
    `run_train2d.sh` wraps bundle→train→infer (needs the `drawing2cad-ml` env + `CADRILLE_REPO`).
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
  **New in Phase 4**: The renderer now implements an "Oracle Matching Strategy", fusing pure mathematical projection (`CADProjector` + `GraphBuilder`) with OCC TechDraw's HLR. This preserves exact 3D topological lineage (`topo_origins`) on visually occluded/segmented 2D lines.
- `research/{werk24-recon,orthosolve-method,foundation-model-survey,amvdg-v0.3-roadmap}.md`
  — the supporting surveys: Werk24 reverse-engineering, the OrthoSolve method note, the
  "does a big model solve geometry?" survey, and the v0.3 representation roadmap.

> The renderer now emits the **v0.3** schema (`topo_origins` for precise 3D provenance, `line_role`/`feature_tag`/`annotations`/`features`,
> plus `profile`/`source`/`world`/`dof` etc.) at `profile: vectorized`, and `python
> spec/validate_amvdg.py <graph>` passes all 7 gates on its output. The tooling
> (`serialize.py`/`score.py`/`graph_to_dxf.py`/`circlenet.py`/`tile.py`/`pipeline.py`) reads the
> latest v0.3/v0.2 names with a v0 fallback, so pre-existing v0 graphs still load until they are regenerated.

Scripts that need fixtures not shipped here (the real drawing, GT graphs) take a `*_DATA`
env var pointing at a local data dir; the spikes print what they expect.
