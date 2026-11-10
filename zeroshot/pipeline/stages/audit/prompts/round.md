## Audit Round

### Task and inputs

Coding and verification for reconstruction round $current_round are complete. Audit that snapshot against the input drawing.

Read this round's `open_tickets`, including their subjects and stage responses, `stage_reports` and `interpretation` from `.snapshots[-1]` in `$reconstruction_path`.

The program is at `$coding_output_path`. Built artifacts are in $attempt_dir:
- `output.step`: the built solid.
- `projection/<view>.dxf` and `projection/<view>.png`: projections for the orthographic views identified in the interpretation.
- `render_3d/*.png`: perspective renders. List the directory for their names.

The pipeline's verification report describes this build.

### Artifact contract

Write the complete `AuditReport` to `$audit_output_path`. Do not modify the program, reconstruction history, input files, verification report or generated artifacts.

Give one `ticket_reviews` entry per open ticket and one `concern_reviews` entry per stage-report concern, keyed `<reporting_stage>.<concern_id>`.

$drawing_diff_reviews

Each finding is one defect:
- `observation`: the shape the drawing shows, the shape the built solid or its projection has, and where. Quantify the difference when the source supports a measurement; do not invent a number.
- `evidence`: boxes measured in each file's own frame: millimetres for DXF, integer pixels for raster images. At least one region must cite a `.dxf` under this build's `projection/` directory.
- `cause`: `interpretation` when the interpretation misreads or omits what the drawing shows and the program builds that error faithfully. `coding` when the interpretation is right but the program builds something else.
- `targets`: the interpretation members the defect concerns. For an interpretation cause, name the wrong members: `sem_...` for a feature or its placement, `dim_...` for a misread printed figure, `view_...` for a wrong crop, role or calibration, `datum` for the shared frame. For something the interpretation omits, name the `view_...` where the drawing shows it. For a coding cause, name the members the program builds wrongly, or give `[]`.
- `revision_request`: what is wrong, not how to fix it. The responsible stage decides the correction; a fix you prescribe may mislead it.
- `related_ticket_ids`: the unsolved tickets this finding continues.

Report all material defects. Give defects with different causes separate findings. Cover every unsolved ticket with the current findings' `related_ticket_ids`. Merge overlapping defects and judge their cause from the current artifacts: it may differ from the old ticket's. Do not repeat a finding inside its ticket review.

AuditReport JSON schema:
```json
$audit_schema
```

### Work cycle

Check in two steps.

1. Check the interpretation against the input drawing.
   - Views: each crop holds its whole view, the role and axes match the projection, and the scale agrees with the printed dimensions.
   - Dimensions: each printed dimension is read correctly and measures the extent it claims. If a printed dimension and the build differ beyond measuring error, one of them is wrong: the number may be misread or belong to another extent, or the build may be off. Recheck both, and report a defect only if the build uses the less likely value. Consistency with the other dimensions and the overall size is the best evidence.
   - Features: each feature's shape, `parameters` and termination agree with its `evidence` regions, its `dimension_refs` and the shared `datum`. Check which side of each boundary is material or void, not just where its projected line lies. A region uses the file and pixel frame of its `view_`; it is not a model position.
   - Look for features, views and figures the interpretation omits.
2. Check the build against the interpretation and the drawing. Inspect its construction as well as its projections and perspective renders.
   - Compare the projections and perspective renders with every input view using load_image: silhouettes, visible and hidden edges, dimensions, feature positions and omissions. Align images on geometry that already matches; do not mistake an image origin or scale difference for a model defect.
   - Check that each straight segment and circular arc in the source is preserved, including in a closed non-circular outline. Inspect the corresponding edge and face types in `$attempt_dir/output.step` via CadQuery (`run_shell`) when the program or geometry warnings indicate spline/loft approximations. Request an analytic reconstruction when a spline replaces those segments, even if the projected error is within stroke width. Splines are appropriate for individual source-supported freeform segments.
   - For constant-section bodies, check for single, clean extrusions rather than stitched primitives or overlapping cuts that introduce extra face boundaries or steps. Inspect the built STEP when face counts or geometry warnings suggest unintended splits. Before dismissing an extra line as an optional tangent boundary, confirm that the built surfaces actually join tangentially there.
   - Adding a thin sheet, narrow land, shallow groove or tiny offset solely to make a projection line appear or disappear is an incorrect reconstruction. Such cases tend to occur when a feature structure is wrongly understood. Check if the area is solid or hollow, cavities or openings, and whether an opening passes through or stops at a floor. Report any unsupported patch and identify whether it comes from the recorded solid/void reading or from the program. Fillets, chamfers and thin features are valid, including small or undimensioned ones, when the source views support their extent and connections.
   - For a defect concealed by projection, use the required DXF evidence box to locate the affected feature, and give the code or built STEP measurement in `observation`. Treat a geometry warning as a location to investigate; establish the defect from the source drawing and the built geometry before reporting it.
   - For a widespread mismatch, check axes, mirroring, crop, scale and alignment before choosing a cause.
   - A defect that comes from a wrong reading found in step 1 has cause `interpretation`. Otherwise it has cause `coding`.
   - If no valid solid was produced, decide whether the failure comes from the program or from the interpretation.

Ticket responses and `concerns` are claims, not proof. Check each against the artifacts: compare the input view with the build's projection of that view where the claim applies. Check whether each earlier defect is resolved, not merely whether an edit was attempted.

After a turn that changes the file, the pipeline validates it and returns paths to evidence images. Each image shows the file named in an evidence entry with that entry's `box` outlined in red. Open each one with `load_image`. Check that the box covers the feature and the dimensions or edges the claim needs, and that the stated mismatch is real. Correct misplaced boxes and unsupported findings, then check the new images. With no findings, review the report itself.

### Submission

Accept only when the build was verified, its solid structure and projections agree with the drawing in all material respects, and no stage output needs correction. A successful STEP export alone does not establish geometric correctness.

Once the file validates and you have checked the evidence, finish with one `AuditSubmission`: `accepted` is true exactly when findings is empty. The decision belongs only in the final response; do not put it in the report or repeat the report in your answer. Stay within the announced turn budget.
