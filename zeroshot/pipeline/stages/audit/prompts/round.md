## Audit Round

### Task and inputs

Coding and verification for reconstruction round $current_round are complete. Audit that immutable snapshot against the input drawing.

Read this round's `open_tickets`, including their subjects and stage responses, and `stage_reports` from `.snapshots[-1]` in `$reconstruction_path`.

The program is at `$coding_output_path`. Built artifacts are in $attempt_dir:
- `output.step`: the built solid.
- `projection/<view>.dxf` and `projection/<view>.png`: projections for the orthographic views identified in the interpretation.
- `render_3d/*.png`: perspective renders. List the directory for their names.

#### Automatic drawing comparison

$drawing_diff_summary

#### Intermediate operation outputs

Recorded directory: $intermediate_returns_dir

Expected paths within this directory: `<ret_name>/output.step`, `<ret_name>/projection/` and `<ret_name>/render_3d/`.

### Artifact contract

Write the complete `AuditReport` to `$audit_output_path`. Do not modify the program, reconstruction history, input files, verification report or generated artifacts.

Review every ticket and every stage-report concern: give one `ticket_reviews` entry per ticket and one `concern_reviews` entry per concern, keyed `<reporting_stage>.<concern_id>`.

- Report all material defects. Quantify the discrepancy when the source supports a measurement; do not invent a number when it does not.
- Cover every unsolved ticket with the current findings' `related_ticket_ids`. Merge overlapping defects and recompute their backtraces from current artifacts: the root may have changed since the old ticket. Do not repeat the backtrace or revision request inside the ticket review.
- Each finding describes one defect with one revision root stage. Separate roots in different stages; several members of one stage sharing the same defect may be requested together.
- State the shape expected from the input, the shape observed in the output, the missing/extra geometry or position/dimension difference, and its location and measurement basis. Then follow the declared links to identify which stage introduced it.
- Measure each evidence `box` in its referenced file's own frame: millimetres for DXF, integer pixels for raster images, including input drawings, projection PNGs and perspective images. At least one region per finding must cite a `.dxf` under this build's `projection/` directory; `intermediate_returns/<ret_>/projection/` counts.
- If a feature visible in the drawing has no corresponding `sem_...` member, leave the `backtrace` empty and request `add` on the whole interpretation stage (`name: null`), proposing one or more stable `sem_...` names. Use the same whole-stage add for a missing view or printed figure, with new `view_...` or `dim_...` names. Correct existing members with `modify` on their stable names.
- Coding backtrace members are stable `ret_...` outputs. `result` is the terminal export, not a causal member; request `modify` on the whole coding stage with `name: null` when its final assignment is defective.
- Coding revisions use only `modify`, including changes that add or remove code. Changes to operation identities or structure belong to operations, which owns the corresponding `ret_...` identities. For other stages, choose the action according to the schema; use rename only when identity must change.

AuditReport JSON schema:
```json
$audit_schema
```

### Work cycle

- Check the doubts in ticket summaries, `concerns` and `unticketed_changes` against the artifacts; their placement does not determine whether a defect is new or which stage caused it. These are claims, not proof of correctness. Check whether each previously observed defect is resolved, not merely whether an edit was attempted. If no valid solid was produced, identify whether the failure comes from coding or an upstream artifact.
- Compare the generated projections and perspective renders with every input view using load_image. Check silhouettes, visible/hidden edges, dimensions, feature positions and omissions. When aligning images, use already matching geometry; do not confuse image origins or scale differences with a model defect.
- For widespread mismatch, inspect axes, mirroring, crop, scale and alignment before assigning a cause.
- Check each interpretation feature against its `evidence` regions and `dimension_refs`. A Region uses the file and pixel/UV frame of its `view_` reference; it is not a model position. Check `parameters`, the described shape and termination, and the shared `datum` against the drawing. Also inspect the original drawing for features omitted from the interpretation.
- Compare the plan with the interpretation and the program/solid. Inspect relevant intermediate `ret_...` outputs to determine whether an operation built what the plan meant it to. Failed exports or renders may be absent; for resumed runs, confirm that recorded files still exist.
- Check coding's `stage_reports.coding.dimension_checks` against the printed dimensions and final geometry. Coverage validation only ensures every ID has an explanation; it does not prove geometric correctness. Investigate unverified claims and doubtful evidence. Absent checks in an old snapshot mean no checks were recorded.
- Trace each defect upstream to the output that introduced it. An output that faithfully implements incorrect upstream information is not the revision target. Use `sem_...` for an incorrect feature or placement, `dim_...` for a misread printed figure, and `view_...` for an incorrect view, crop or calibration. If the defect is established directly in that output, leave the `backtrace` empty.
- Follow declared links upstream: `ret_x -> op_x -> sem_...`, with `op_x.semantics` identifying the feature. Within interpretation, follow cited features, dimensions and views. Do not invent links through unrelated outputs.

After a turn that changes the file, the pipeline validates it and returns paths to evidence images. Each image displays the file named in an evidence entry, with that entry's `box` outlined in red. Open each evidence image with `load_image`: check that the red box covers the intended feature and any dimensions or edges needed to support the claim, and compare the input drawing with the built projections or renders to confirm the stated mismatch. Correct misplaced or insufficient boxes and unsupported findings in the report, then inspect the regenerated images. With no findings, review the report itself.

### Submission

- Accept only when the solid was verified, matches the drawing in all material respects, and no stage output requires correction. Successful STEP export alone does not establish geometric correctness.

Once the file validates and you have checked the evidence, finish with one `AuditSubmission`: `accepted` is true exactly when findings is empty. The decision belongs only in the final response; do not put it in the report or repeat the report in your answer. Stay within the announced turn budget.
