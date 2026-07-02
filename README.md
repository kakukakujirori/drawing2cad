# Drawing2CAD

R&D for **2D engineering drawing → 3D CAD** conversion. Research notes and the
running log live in [`research/`](research/) (start with `research/research-log.md`).

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

Files: [`spec/README.md`](spec/README.md) (overview) · `spec/AMVDG_v0.3.md` (prose spec) ·
`spec/AMVDG_v0.3.schema.json` (accepts 0.2/0.3) · `spec/validate_amvdg.py` (7 gates) ·
[`spec/AMVDG_TUTORIAL.md`](spec/AMVDG_TUTORIAL.md) (日本語チュートリアル: JSONの読み方と2D復元).
Both the worked example and every renderer-emitted graph validate:
```bash
python spec/validate_amvdg.py spec/example_flange_v0.3.json          # all 7 gates PASS
python spec/validate_amvdg.py experiments/dataset_recon/XXX.graph.json
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

**Synthetic dimensioned-drawing renderer** (seed B-rep STEP → PNG + AMVDG graph JSON).
`batch_dataset.py` runs the FreeCAD HLR render + cairosvg rasterize + scan-aug in one env
and emits the `(PNG, graph.json)` pairs. It needs the repo root on `PYTHONPATH` (the renderer
imports `scripts.renderer.*`); `batch_dataset.py` sets that for its stage-1 subprocess.

Two seed sources are wired up (seed 3D data must be **B-rep STEP** — STL is unsuitable for HLR):

*Zero-To-CAD* (`ADSKAILab/Zero-To-CAD-1m`, Apache-2.0) — **preferred**. Ships each part as a
binary `step_file` (**directly usable, no STL→STEP conversion**) plus its GT `cadquery_file`.
`select_zero_to_cad.py` needs the `datasets` lib (an env like `py312`, *not* the FreeCAD env);
it also drops the GT `{uuid}.cadquery.py` next to each STEP for the future AMVDG→3D leg.
```bash
# one-shot driver (select in py312 + render in drawing2cad); override N/SPLIT/STAGE/OUT via env:
N=300 SPLIT=validation scripts/renderer/run_zero_to_cad_gt.sh    # -> experiments/dataset_z2c
# or the two steps by hand:
/path/to/py312/bin/python scripts/renderer/select_zero_to_cad.py 300 experiments/stage_z2c --split validation
python scripts/renderer/batch_dataset.py experiments/stage_z2c experiments/dataset_z2c
```

*Fusion360Gallery* `reconstruction` subset — `select_fusion360_recon.py` is specific to that
set's `{partid}_{hash}_{seq}_{substep}` naming (takes each part's final state, samples across parts):
```bash
python scripts/renderer/select_fusion360_recon.py /path/to/Fusion360Gallery/r1.0.1/reconstruction 2500 experiments/stage_recon
python scripts/renderer/batch_dataset.py experiments/stage_recon experiments/dataset_recon
```
A graph can be exported to DXF 2D-CAD with `python scripts/amvdg/graph_to_dxf.py GRAPH.json OUT.dxf`.

**B2-cadrille baseline** (CADGenBench drawings → CadQuery → validity/scale metrics):
```bash
scripts/run_b2.sh --n 49          # or  IDS=101,103 scripts/run_b2.sh
```
Output under `experiments/` (git-ignored). See `research/research-log.md` for results.

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
