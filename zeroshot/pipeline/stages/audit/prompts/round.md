## Audit Round
Coding and verification for reconstruction round $current_round are complete. Audit that immutable snapshot against the input drawing.

Read this round's `open_tickets`, including their subjects and stage responses, and `stage_reports` from `.snapshots[-1]` in `$reconstruction_path`. Review every ticket and every stage-report concern: give one `ticket_reviews` entry per ticket and one `concern_reviews` entry per concern, keyed `<reporting_stage>.<concern_id>`.

The program is at `$coding_output_path`. Built artifacts are in $attempt_dir:
- `output.step`: the built solid.
- `projection/<view>.dxf` and `projection/<view>.png`: projections for the orthographic views identified in the interpretation.
- `render_3d/*.png`: perspective renders. List the directory for their names.

## Intermediate operation outputs

Recorded directory: $intermediate_returns_dir

Expected paths within this directory: `<ret_name>/output.step`, `<ret_name>/projection/` and `<ret_name>/render_3d/`.

## Submission

Write the complete `AuditReport` to `$audit_output_path`. Do not modify the program, reconstruction history, input files, verification report or generated artifacts.

After a turn that changes the file, the pipeline validates it and generates evidence crops. Use `load_image` to check that each crop supports your findings. Correct the report and inspect the new crops if necessary. With no findings, review the report itself.

Once the file validates and you have checked the evidence, finish with one `AuditSubmission`: `accepted` is true exactly when findings is empty. The decision belongs only in the final response; do not put it in the report or repeat the report in your answer. Stay within the announced turn budget.

AuditReport JSON schema:
```json
$audit_schema
```
