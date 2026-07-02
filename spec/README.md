# AMVDG — the drawing-to-CAD intermediate representation (DSL)

**AMVDG = Annotated Multi-View Drawing Graph (v0.3).** This is the project's own DSL for the
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
   In JSON this layer is the per-element **`prov`** key (there is no top-level `provenance`
   object); v0.3's `prov.topo_origins` lives here. This is distinct from layer 3 `features`:
   `topo_origins` is the per-primitive B-rep lineage the correspondence `members[]` are *derived
   from* — see `AMVDG_TUTORIAL.md` §5.1.

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
python spec/build_flange.py                                   # rebuild spec/flange.step (cadquery)
python spec/validate_amvdg.py spec/example_flange_v0.3.json   # all 7 gates PASS
```

The flange worked example is the **renderer's actual output** on `spec/flange.step` (rebuild it
with `build_flange.py`): a round bolt flange (plate ⌀120 × hub ⌀76 × bore ⌀40, 6 bolt holes ⌀11
on PCD 95, plate+hub 38 tall). It is `profile: vectorized` — 9 cylinder `features` (concentric
plate/hub/bore + the 6 bolt holes, each bound across views by shared `topo_origins`) and 12
`annotations` (bbox 120×38×120 + the 6 diameters). DoF/coverage is emitted as zero (the
`gt_executable` build-recipe accounting is deferred to the 3D leg — see the bottom of this file).

## Files
| file | role |
|---|---|
| `AMVDG_v0.3.md` | the human spec (authoritative prose; §9 = v0.3 delta) |
| `AMVDG_v0.3.schema.json` | JSON Schema — structure, `additionalProperties:false` (accepts `amvdg_version` 0.2/0.3) |
| `validate_amvdg.py` | the 7-gate validator + the canonical pipe-delimited DSL `lower`/`lift` |
| `example_flange_v0.3.json` | renderer's actual `vectorized` output on `flange.step` (breaking any ref/census/round-trip fails it) |
| `flange.step` · `build_flange.py` | the example's seed solid + the cadquery script that rebuilds it |
| `AMVDG_TUTORIAL.md` | 日本語チュートリアル — JSONの読み方、2D図面の復元手順、featuresの意味 |

## Two serialization forms
- The **canonical model-facing DSL** is the pipe-delimited one-record-per-line form in
  `validate_amvdg.py` (`V|… A|… F|…`), grammar-constrainable so a malformed *line* drops
  instead of the whole document.
- The **training serializer** `scripts/amvdg/serialize.py` emits a compact-JSON `g1` target
  for cadrille.

## v0.2 → v0.3 (current renderer output; 2026-07-02)
Motivated by `research/2026-07-01-AMVDG-v0.3.md` (degeneracy + occlusion-split provenance):
- **`prov.topo_origins`** on every geometry primitive: `[{dim, id, role}]` — dim 2=Face/1=Edge,
  role ∈ {edge, silhouette, boundary, edge-on, parent_face, axis}. Produced by the analytic
  oracle (`scripts/renderer/projector/` + `graph_builder.py`) matched against HLR output.
- **Multi-view `features[].members`** derived from shared origin face-ids (top circle +
  side silhouettes + centerlines of the same cylinder bind under one `feature_id`).
- **Arc `start_angle`/`end_angle`** (px frame y-down, arc = start→end with increasing
  atan2 angle mod 360) — endpoint-only arcs were major/minor-ambiguous.
- **Centerline primitives** (`line_role: center`) emitted with the drawn centerline crosses.
- **`frame.axis_remap` + normalized `projection_dir`** (part→viewer unit vector; front
  `(0,-1,0)` · top `(0,0,1)` · right `(1,0,0)`, third-angle) so px→model inversion is recorded.
- Schema: `amvdg_version` enum {0.2, 0.3}; provenance def gained `topo_origins`.

The renderer (`render_dataset.py`) emits this schema directly at `profile: vectorized`, and the
whole toolchain — `serialize.py`, `graph_to_dxf.py`, `score.py`, `circlenet.py`, `tile.py`,
`pipeline.py` — reads the current names with a **v0 fallback** so pre-existing graphs still load.
`python validate_amvdg.py <renderer-output>` passes all 7 gates.

Remaining for a later pass: richer `dof`/coverage (currently emitted as zero — the OrthoSolve
accounting) and the `gt_executable` profile (kernel `build` recipes), to be filled when the
2D→3D leg starts.
