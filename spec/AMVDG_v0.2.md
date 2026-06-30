# AMVDG v0.2 — Annotated Multi-View Drawing Graph (spec)

2D→2D leg intermediate representation for `raster drawing → [AMVDG] → 3D CAD`.
**v0.2 is mechanically validated**: `validate_amvdg.py example_flange_v0.2.json` passes all 7 gates;
breaking any ref/census/DoF/round-trip makes it fail. This file is the human spec; the authoritative
artifacts are `AMVDG_v0.2.schema.json` (structure), `validate_amvdg.py` (enforcement), `example_flange_v0.2.json` (a valid instance).

## 1. The 4 authored layers
1. **geometry** — per-view typed primitives (`line/circle/arc/ellipse/bezier/bspline/polyline/point/generic`) + `line_role` (visible/hidden/center/phantom/section_cut/break).
2. **annotation** — dims/GD&T/thread/CBORE/CSINK/chamfer bound to geometry via `refs`, with exact value + `value_source` + tolerance. (dimension-spine; absolute scale lives here.)
3. **correspondence** — `features[]`, each a shared `feature_id` binding the SAME 3D feature across views via `members[]{view,primitive_id,projection_role}`.
4. **provenance** — every element → originating 3D entity+param (synthetic GT only).
Constraints are **derived** (autoconstrain), not authored. Uncertainty is a per-element attribute.

## 2. Who produces what (4-way)
| owner | fields |
|---|---|
| **MODEL** (DSL subset) | V-records (view meta + signed-axis remap), A-records (kind/param_role/refs/value/tolerance/feature payloads), F-records (feature_id/kind/members/axis/extrude_dir/parent/build-hint), sparse G/X/C/D. **No raw coords, no constraints.** |
| **CV vectorizer** | `primitives[].{p1,p2,pts,center,r_px,bbox_px,angles}`; `coords_source`,`state` |
| **derived** | `constraints[]` (autoconstrain), `dof`, `validity`, `contradictions` (validator recomputes & cross-checks) |
| **renderer (GT)** | `prov` on every element (feature_id assigned 3D-side → correspondence is provenance-exact, not radius-matched) |
| **prior/library** | may only write `value_source=prior` / `dof.supplied_by_prior`; scorer excludes these from determined-DoF |

## 3. Validation profiles (the key v0.2 addition — codex's fix)
One schema, four profiles (`profile` field); `validate_amvdg.py` enforces per-profile requirements so **"schema-valid" means something at each stage**:
| profile | requires |
|---|---|
| `model` | structure + refs + census + DoF + round-trip; **coords may be null** |
| `vectorized` | + every `state=known` primitive has non-null coords and `coords_source≠unknown` |
| `derived` | + `constraints[]` present, DoF recomputed |
| `gt_executable` | + `prov` on every feature; every **determined** feature has complete `build` (op+datum_plane) and every `required_param` has a non-null `driven_by` → a `value_state=known` annotation |

## 4. Validity / DoF self-check (mechanical — see validate_amvdg.py)
1. **structure** — jsonschema (`additionalProperties:false`).
2. **ref-integrity** — refs/members/feature_id/parent resolve; no parent cycle.
3. **census** — `validity.census` declared counts == emitted (abstain ≠ absent).
4. **DoF recompute** — `required/determined/determined_by_geometry/missing/coverage` recomputed == `dof` == `self_declared`. determined = params backed by an annotation with `value_source∈{ocr,gt}`; `count`-type evidence → `determined_by_geometry` (NOT determined).
5. **abstain-disjoint** — `dof.undetermined` never overlaps a determined param (cannot hallucinate the under-determined).
6. **profile gate** — §3.
7. **round-trip** — model subset `lower→DSL→lift` is identity (DSL↔JSON well-defined; numbers canonicalized).
**"Sufficient" = correctness-on-determined × coverage.** Reported, never "parse-rate" or whole-blob equality.

Flange worked example: 22 top circles (12 visible/10 hidden), PCD 70mm, 3 dims → required **16**, determined **3** (plate dx,dz + boss dia), determined_by_geometry **1** (bolt count), missing **12**, **coverage 0.1875**. 12 params correctly abstained; `kernel_executable=false` honestly reported.

## 5. DSL serialization (enums are SCHEMA-EXACT — codex fix #2)
The model emits flat one-record-per-line; tokens are the **same enum strings as the schema** (no `cbore`/`boltcirc` aliases — they were the v0.1 mismatch). Lift/lower are defined in `validate_amvdg.py` and round-trip-tested. Pipe-delimited canonical form:
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
- **Worked example is now valid JSON** in its own file `example_flange_v0.2.json`, validates against the schema; **DoF arithmetic corrected** (v0.1 said required 17/determined 4 — inconsistent; now 16/3/coverage 0.1875) (codex).
- **Added validation profiles** (model/vectorized/derived/gt_executable) so optional ≠ unconstrained; `gt_executable` requires complete `build`+`prov` (codex "too permissive").
- **Added `validate_amvdg.py`** — the schema is now machine-checkable, not asserted.

## 8. Still open (not blockers; tracked for v0.3)
- inference-time `feature_id` authorship on real drawings (no 3D) — deterministic matcher proposes, scored separately.
- `feature.build` authorship on real — fusion pass vs model hint.
- `param_role` ontology + per-kind required-DoF must be locked in a versioned file shared by renderer/3D-builder/scorer.
- section/hatch (⑥), GD&T depth, bezier/bspline sub-schema → v0.2 deferred.
- DoF/Sketcher harness is **referenced but not yet built as code** (separate from this schema).
- Landing: bundle v0→v0.2 migration (visibility→line_role, feature→feature_tag, dimensions→annotations, correspondences→features) with the `projectEx`+geometry-refs port into `drawing2cad/scripts/renderer/`. `graph_to_dxf.py` needs: key on `line_role`, add CENTER/PHANTOM layers, skip null-coord primitives.

---
Artifacts: `AMVDG_v0.2.schema.json` · `example_flange_v0.2.json` · `validate_amvdg.py` (run it: `python validate_amvdg.py example_flange_v0.2.json`).
