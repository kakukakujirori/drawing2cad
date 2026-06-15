# drawing2cad

R&D for **2D engineering drawing → 3D CAD** conversion. Research notes and the
running log live in [`research/`](research/) (start with `research/research-log.md`).

## Environments

Two conda environments (scripts assume the relevant one is **already activated** —
they call bare `python`, never `conda run`, and hardcode no machine paths):

### 1. `drawing2cad` — renderer / CAD (Phase-1 synthetic drawings)
```bash
conda env create -f environment.yml      # conda-forge FreeCAD + matplotlib + ezdxf
conda activate drawing2cad
```
FreeCAD comes from conda-forge, so `import FreeCAD` works with no PYTHONPATH.
If instead you have a **source-built** FreeCAD, omit `freecad` from the env and
export its lib dir before running:
```bash
export FREECAD_LIBDIR=/path/to/FreeCAD/build/release/lib
```

### 2. `drawing2cad-ml` — B2 / cadrille inference + CadQuery execution
```bash
conda create -n drawing2cad-ml python=3.11
conda activate drawing2cad-ml
pip install -r requirements-ml.txt
export CADRILLE_REPO=/path/to/cadrille          # the cadrille model-code clone
```

## Usage

**Synthetic dimensioned-drawing renderer** (seed B-rep STEP → PNG + DXF + graph JSON):
```bash
conda activate drawing2cad
# (FREECAD_LIBDIR only if using a source-built FreeCAD)
scripts/renderer/run_batch.sh <step_dir> <out_dir> [N=0:all]
# e.g.  scripts/renderer/run_batch.sh /path/to/Fusion360Gallery/.../reconstruction out/drawings 2000
```
Seed 3D data must be **B-rep (STEP)** — STL meshes are unsuitable for HLR. Good
sources: Fusion360Gallery (STEP), Zero-To-CAD (`ADSKAILab/Zero-To-CAD-1m`,
Apache-2.0; CadQuery code → convert to STEP first).

**B2-cadrille baseline** (CADGenBench drawings → CadQuery → validity/scale metrics):
```bash
conda activate drawing2cad-ml
scripts/run_b2.sh --n 49          # or  IDS=101,103 scripts/run_b2.sh
```
Output under `experiments/` (git-ignored). See `research/research-log.md` for results.
