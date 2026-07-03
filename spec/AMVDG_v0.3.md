# AMVDG v0.3 — Annotated Multi-View Drawing Graph (spec)

2D→2D leg intermediate representation for `raster drawing → [AMVDG] → 3D CAD`.
Why a typed graph (not raw DXF / single-sketch constraints / a CAD-command list)? 3D recovery needs
the *binding* a drawing carries — which dimension constrains which geometry, which views show the
same feature, what a hidden line means. The novel part is the cross-view layer: orthographic
projection is rank-2 (depth is the kernel), so no single view fixes depth — only inter-view feature
identity can, and no standard sketch format expresses that.
**v0.3 is mechanically validated**: `validate_amvdg.py example_flange_v0.3.json` passes all 7 gates;
breaking structure/ref-integrity/round-trip makes it fail. (The census and DoF gates only bite once
`validity`/per-feature `dof` are emitted — see the status note in §1; on current `vectorized` output
they pass vacuously.) This file is the human spec; the authoritative artifacts are
`AMVDG_v0.3.schema.json` (structure), `validate_amvdg.py` (enforcement), `example_flange_v0.3.json` (a valid instance).

## 1. The 4 authored layers
1. **geometry** — per-view typed primitives (`line/circle/arc/ellipse/bezier/bspline/polyline/point/generic`) + `line_role` (visible/hidden/center/phantom/section_cut/break).
2. **annotation** — dims/GD&T/thread/CBORE/CSINK/chamfer bound to geometry via `refs`, with exact value + `value_source` + tolerance. (dimension-spine; absolute scale lives here.)
3. **correspondence** — `features[]`, each a shared `feature_id` binding the SAME 3D feature across views via `members[]{view,primitive_id,projection_role}`.
4. **provenance** — every element → originating 3D entity+param (synthetic GT only). In JSON this is the per-element `prov` key (v0.3's `prov.topo_origins` lives here); there is no top-level `provenance` object.

> **Implementation status (2026-07-02).** Only layers 1–4 above are actually **emitted** by the renderer, all at `profile: vectorized`. The *derived* additions promised below — `constraints[]` (autoconstrain), a populated `dof`/coverage, `validity`/`census` — are **schema-reserved but NOT produced** by any current code path (renderer, spec example, or Zero-To-CAD batch). `dof`/coverage is on the roadmap (research-log TODO #6, filled with the 3D leg); `constraints[]`/`validity`/the `derived` profile are **not** currently planned (they were tied to OrthoSolve, which is off the critical path). Read §2/§3 as the *target* schema, not the current output.

## 2. Who produces what (4-way)
| owner | fields |
|---|---|
| **MODEL** (DSL subset) | V-records (view meta + signed-axis remap), A-records (kind/param_role/refs/value/tolerance/feature payloads), F-records (feature_id/kind/members/axis/extrude_dir/parent/build-hint), sparse G/X/C/D. **No raw coords, no constraints.** *(conceptual role — the synthetic renderer authors these; a real 2D→AMVDG model does not exist yet.)* |
| **CV vectorizer** | `primitives[].{p1,p2,pts,center,r_px,bbox_px,angles}`; `coords_source`,`state`. *(conceptual role — in the synthetic pipeline `coords_source` is always `gt`; there is no separate vectorizer.)* |
| **derived** | `constraints[]` (autoconstrain), `dof`, `validity`, `contradictions`. ⚠️ **NOT emitted by any current code** — `constraints[]`/`validity` unplanned; `dof` emitted as an all-zero block (populated later, TODO #6). `validate_amvdg.py` has **no `derived` branch**, so its census/DoF gates run but are vacuous on current output. |
| **renderer (GT)** | `prov` on every element (feature_id assigned 3D-side → correspondence is provenance-exact, not radius-matched). *(the only producer that actually runs today.)* |
| **prior/library** | may only write `value_source=prior` / `dof.supplied_by_prior`; scorer excludes these from determined-DoF. *(design placeholder — no prior/library stage implemented.)* |

## 3. Validation profiles (the key v0.2 addition — codex's fix)
One schema, four profiles (`profile` field); `validate_amvdg.py` enforces per-profile requirements so **"schema-valid" means something at each stage**. **Today the renderer only ever emits `vectorized`** — `derived`/`gt_executable` are defined in the schema but nothing produces them (see the status note in §1):
| profile | requires | produced? |
|---|---|---|
| `model` | structure + refs + census + DoF + round-trip; **coords may be null** | not a distinct output (renderer always fills coords) |
| `vectorized` | + every `state=known` primitive has non-null coords and `coords_source≠unknown` | ✅ **the only profile emitted** (renderer, spec example, Zero-To-CAD) |
| `derived` | + `constraints[]` present, DoF recomputed | ❌ **not implemented** — `check_profile` has no `derived` branch; `constraints[]`/`validity` unplanned |
| `gt_executable` | + `prov` on every feature; every **determined** feature has complete `build` (op+datum_plane) and every `required_param` has a non-null `driven_by` → a `value_state=known` annotation | ❌ **not yet** — roadmap TODO #6 (with the 3D leg's build recipes) |

## 4. Validity / DoF self-check (mechanical — see validate_amvdg.py)
1. **structure** — jsonschema (`additionalProperties:false`).
2. **ref-integrity** — refs/members/feature_id/parent resolve; no parent cycle.
3. **census** — `validity.census` declared counts == emitted (abstain ≠ absent).
4. **DoF recompute** — `required/determined/determined_by_geometry/missing/coverage` recomputed == `dof` == `self_declared`. determined = params backed by an annotation with `value_source∈{ocr,gt}`; `count`-type evidence → `determined_by_geometry` (NOT determined).
5. **abstain-disjoint** — `dof.undetermined` never overlaps a determined param (cannot hallucinate the under-determined).
6. **profile gate** — §3.
7. **round-trip** — model subset `lower→DSL→lift` is identity (DSL↔JSON well-defined; numbers canonicalized).
**"Sufficient" = correctness-on-determined × coverage.** Reported, never "parse-rate" or whole-blob equality.

Flange worked example (v0.3): the **renderer's actual output** on `spec/flange.step` (`build_flange.py`),
`profile: vectorized` — a round bolt flange (plate ⌀120 × hub ⌀76 × bore ⌀40, 6 bolt holes ⌀11 on PCD 95).
9 cylinder `features` bound across views by shared `topo_origins`; 12 `annotations` (bbox 120×38×120 + 6 diameters).
DoF/coverage emitted as zero (the `gt_executable` build-recipe accounting from the old hand-authored example is
deferred to the 3D leg; see §8/§9). Historic hand-authored numbers (16/3, coverage 0.1875) are retained below as the DoF-arithmetic worked example only.

## 5. DSL serialization (EXPLORATORY — tokenization not settled)
> **Status.** This belongs to the **de-prioritized 2D→AMVDG leg** (the current plan is 3D-leg-first with CadQuery output — research-log 2026-07-02). The right tokenization for a graph target — how to encode primitives, cross-view `members`, and whether to carry `prov` — is an **open design question**, not a locked decision. Two *different, non-interchangeable* forms exist in code, and they cover **disjoint** layers:
> - **pipe-DSL** (`validate_amvdg.py` `lower`/`lift`, below): view/annotation/**feature** records. Encodes cross-view correspondence (`members` as `view:primitive_id:projection_role`) and is round-trip-tested (gate 7) — but carries **no coordinates and no `prov`**, and its `F|` members reference primitive-ids that a separate vectorizer stage must assign.
> - **`scripts/train2d/serialize.py`** (`g1`, the cadrille training target): per-view primitives **with** coords + dims, but **drops `features[]` entirely** (it is explicitly intra-view). So the current training target expresses **no** cross-view correspondence.
>
> Neither is a full-graph serialization; settling one is a prerequisite before resuming the 2D leg.

The pipe-DSL: model emits flat one-record-per-line; tokens are the **same enum strings as the schema** (no `cbore`/`boltcirc` aliases — they were the v0.1 mismatch). Lift/lower are defined in `validate_amvdg.py` and round-trip-tested. Pipe-delimited canonical form:
```
V|front|front|0,-1,0|-
A|D1|linear|width|plate|100|known|F13.F0
F|boss_d100|cylinder|Z|+Z|known|-|top:T0:visible
```
Grammar is grammar-constrainable (one record/line) → the v0.1 4/12 whole-doc JSON-parse failures become 0 (a malformed *line* drops, not the document). Report `lines_dropped/lines_total`.

## 6. ①–⑫ coverage (unchanged from v0.1 intent; now enforced by profile)
geometry+line_role①, view-meta/projection/align②, annotation-binding+value+tol/GD&T/thread/CBORE/CSINK/chamfer③, cross-view feature-id④, provenance⑤, section/aux⑥ **(schema-reserved, NOT emitted v0.1/0.2)**, symmetry/centerline⑦, units/coord-sys⑧, DoF self-check⑨, round-trip DXF & 3D⑩, LLM-text+GNN-graph⑪, noise/confidence/contradiction⑫.

## 7. Changelog v0.1 → v0.2 (cross-analysis fixes: Claude + Codex)
- **Removed** leaked agent process-text that had contaminated the v0.1 `.md` (codex).
- **Reconciled** DSL grammar enums with schema enums — schema is the single source (codex #2).
- **Fixed** GBNF ambiguity (`prole` duplicate; `status`/`axintent` holes) → round-trip now defined & TESTED (both).
- **Worked example is now valid JSON** in its own file `example_flange_v0.3.json`, validates against the schema; **DoF arithmetic corrected** (v0.1 said required 17/determined 4 — inconsistent; now 16/3/coverage 0.1875) (codex).
- **Added validation profiles** (model/vectorized/derived/gt_executable) so optional ≠ unconstrained; `gt_executable` requires complete `build`+`prov` (codex "too permissive").
- **Added `validate_amvdg.py`** — the schema is now machine-checkable, not asserted.

## 8. Still open (not blockers; tracked for v0.3)
- **serializer robustness when the annotation vocabulary grows**: `train3d/serialize.py` TypeErrors on
  `value: null` annotations (note/datum/gdt) and drops thread/counterbore/tolerance payloads —
  harmless today (the renderer only emits numeric linear/diameter dims) but must be handled
  before richer dimensioning GT lands (2026-07-03 review, A-3).
- inference-time `feature_id` authorship on real drawings (no 3D) — deterministic matcher proposes, scored separately.
- `feature.build` authorship on real — fusion pass vs model hint.
- `param_role` ontology + per-kind required-DoF must be locked in a versioned file shared by renderer/3D-builder/scorer.
- section/hatch (⑥), GD&T depth, bezier/bspline sub-schema → v0.2 deferred.
- **curved-edge provenance (found 2026-07-03)**: the renderer never actually emits
  `ellipse`/`bspline` primitives — `classify_edge` discretizes every non-circular HLR edge to
  `polyline` (Z2C 274: 1657 rows in 95 graphs, ALL without `topo_origins` → invisible to
  features/dims, and the main driver of long-tail g2 token counts). TODO: extend
  `scripts/renderer/projector/` — ellipse first (a tilted cylinder/cone projects to an exact
  ellipse, analytic), then bspline silhouettes (at minimum hashCode-bind the originating 3D
  edge); discretization stays as fallback.
- DoF/Sketcher harness is **referenced but not yet built as code** (separate from this schema).
- Landing: bundle v0→v0.2 migration (visibility→line_role, feature→feature_tag, dimensions→annotations, correspondences→features) with the `projectEx`+geometry-refs port into `drawing2cad/scripts/renderer/`. `graph_to_dxf.py` needs: key on `line_role`, add CENTER/PHANTOM layers, skip null-coord primitives.

## 9. v0.3 delta (what the renderer emits now — schema accepts both versions)
Design note: `research/2026-07-01-AMVDG-v0.3.md`. All additions are BACKWARD-compatible fields:
- **`prov.topo_origins`**: `[{dim, id, role}]` per geometry primitive. `dim` 2=Face / 1=Edge /
  0=Vertex; `id` = stable B-rep index (`Face_i`/`Edge_j` in `shape.Faces/Edges` order);
  `role` ∈ {`edge` (3D edge projects to this primitive), `silhouette` (cylinder limb line),
  `boundary` (cylinder seen head-on), `edge-on` (planar face degenerated to a line),
  `parent_face` (face adjacent to a directly-projected edge — Type-2 correspondence),
  `axis` (centerline of that face's cylinder)}. Occlusion-split segments carry the SAME
  origin ids; edge-on faces list both the face and the coincident edges.
- **conventions locked**: `projection_dir` = unit vector part→viewer (front `(0,-1,0)`,
  top `(0,0,1)`, right `(1,0,0)`; third-angle). `frame.axis_remap` records px axes as signed
  model axes (front `px_x:+X, px_y:-Z` · top `+X,-Y` · right `+Y,-Z`). Arc
  `start_angle`/`end_angle`: degrees in the px frame (y down), θ=atan2(y−cy, x−cx), arc runs
  start→end with increasing θ mod 360.
- **centerline primitives**: the drawn centerline crosses are real primitives
  (`line_role: center`, `feature_id` set) — the graph explains every inked line.
- **features**: members span views (bound by shared `topo_origins` face ids);
  `prov.occ_face_ids` lists the owning B-rep faces.

**v0.3.1 delta (2026-07-03)**: `frame.model_origin` — the model-frame mm coordinate (source STEP
frame) at `origin_px`, per px axis (`m = model_origin + sign(axis_remap)·(px − origin_px)/px_per_mm`).
Makes px→model exact without the per-view shift discovery `train3d/serialize.py` documents. **Synthetic
GT diagnostic only**: a real 2D pipeline cannot know the authoring origin, so model-input
serializations (g2) must keep a drawing-derivable frame (content-min-0) — otherwise the 3D leg
trains on coordinates that literally match the GT program's frame, an information leak that a real
2D front-end cannot reproduce. `amvdg_version: "0.3.1"`; schema accepts 0.2/0.3/0.3.1.
**Profile gate hardened (same rev)**: `vectorized`/`gt_executable` now REQUIRE per-view `frame`
(origin_px, px_per_mm, axis_remap.px_x/px_y) — the 3D-leg consumer (`train3d/serialize.py`) reads them
unconditionally, so "schema-valid vectorized" now implies "consumable" — and `kind: linear`
annotations must carry `subtype` horizontal|vertical (otherwise their measurement span is
undefined). Verified: renderer output (0.3 and 0.3.1) passes; stripping `frame` or the subtype
fails the gate.

---
Artifacts: `AMVDG_v0.3.schema.json` · `example_flange_v0.3.json` · `validate_amvdg.py` (run it: `python validate_amvdg.py example_flange_v0.3.json`).
