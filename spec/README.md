# AMVDG — the drawing-to-CAD intermediate representation (DSL)

**AMVDG = Annotated Multi-View Drawing Graph.** This is the project's own DSL for the
`raster drawing → [AMVDG] → 3D CAD` pipeline: the structured form a 2D mechanical drawing
is parsed into *before* a model reasons about 3D. It is the concrete realization of the
"annotated multi-view drawing graph" decided in `research/research-log.md` (2026-06-19),
then built out and mechanically validated (see the 2026-06-20 log entry).

## Why a graph (not raw DXF / sketch-constraints / CAD-command list)
3D recovery needs the *binding* information a drawing carries: **which dimension constrains
which geometry, which views show the same feature, what a hidden line means.** Raw DXF (dumb
geometry), single-sketch constraint DSLs (intra-view only), and procedural command lists
(DeepCAD/cadrille — these are 3D *generators*, not a 2D IR) each drop one of those. A typed
graph keeps them as edges. The cross-view layer is the novel part: orthographic projection is
rank-2 (depth is the kernel), so a single view cannot fix depth — only inter-view feature
identity can. No standard sketch format expresses that.

## The 4 authored layers
1. **geometry** — per-view typed primitives (`line/circle/arc/ellipse/bezier/bspline/polyline/point`)
   + `line_role` (visible/hidden/center/phantom/section_cut/break).
2. **annotation** — dims/GD&T/thread/CBORE/CSINK/chamfer bound to geometry via `refs`, carrying
   the **exact value + tolerance** (the absolute-scale spine — the opposite of cadrille's
   scale collapse).
3. **correspondence** — `features[]`, each a shared `feature_id` binding the SAME 3D feature
   across views via `members[]{view, primitive_id, projection_role}`.
4. **provenance** — every element → its originating 3D entity+param (synthetic-GT only).

Constraints (parallel/concentric/…) are **derived** (FreeCAD autoconstrain), never authored —
the model emits enums + OCR'd values + id pointers only, so it cannot hallucinate the
downstream 3D decision.

## Who produces what
| owner | writes |
|---|---|
| **model** (the DSL subset) | view meta, annotation kind/value/refs, feature id/kind/members. No raw coords, no constraints. |
| **CV vectorizer** | primitive coordinates (`p1/p2/center/r_px/…`) |
| **derived** | `constraints[]`, `dof`, `validity` (the validator recomputes & cross-checks) |
| **renderer (GT)** | `prov` on every element → correspondence is provenance-exact, not radius-guessed |

## Four validation profiles (so "schema-valid" means something at each stage)
`model` (coords may be null) ⊂ `vectorized` (coords required) ⊂ `derived` (constraints present)
⊂ `gt_executable` (prov + a complete kernel `build` on every determined feature).

## Seven mechanical gates — `validate_amvdg.py`
structure (jsonschema) · ref-integrity · census (abstain ≠ absent) · DoF recompute · abstain-disjoint
· profile gate · round-trip (DSL ↔ JSON is identity). "Sufficiency" is reported as
**correctness-on-determined × coverage**, never parse-rate.

```bash
python spec/validate_amvdg.py spec/example_flange_v0.2.json   # all 7 gates PASS
```

The flange worked example: 22 top circles (12 visible / 10 hidden), 3 dims → required **16**
params, determined **3** (plate dx,dz + boss dia), 1 by geometry (bolt count), **coverage 0.1875**,
12 params honestly abstained, `kernel_executable=false` reported (not hidden).

## Files
| file | role |
|---|---|
| `AMVDG_v0.2.md` | the human spec (authoritative prose) |
| `AMVDG_v0.2.schema.json` | JSON Schema — structure, `additionalProperties:false` |
| `validate_amvdg.py` | the 7-gate validator + the canonical pipe-delimited DSL `lower`/`lift` |
| `example_flange_v0.2.json` | a valid `gt_executable` instance (breaking any ref/DoF/round-trip fails it) |

## Two serialization forms — and an open migration gap
- The **canonical model-facing DSL** is the pipe-delimited one-record-per-line form in
  `validate_amvdg.py` (`V|… A|… F|…`), grammar-constrainable so a malformed *line* drops
  instead of the whole document.
- The **training serializer** `scripts/amvdg/serialize.py` emits a compact-JSON `g1` target
  for cadrille — but it (and `graph_to_dxf.py`, `score.py`, and the renderer `render_dataset.py`)
  still use the **v0 field names** (`visibility`, `feature`, `dimensions`) whereas this v0.2
  schema uses (`line_role`, `feature_tag`, `annotations`). Running `serialize.py` on a v0.2
  instance therefore drops dims and reads every line as visible. **Reconciling the v0 toolchain
  to the v0.2 names is the open landing task** (spec §8); do it across renderer + serializer +
  scorer together, not one side at a time.
