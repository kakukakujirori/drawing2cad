## Audit Round
Coding and verification for reconstruction round $current_round are complete. Audit that immutable snapshot against the input drawing.

The program is at `$coding_output_path`.

Built artifacts are in $attempt_dir:
- `output.step`: the built solid.
- `projection/<view>.dxf` and `projection/<view>.png`: projections for the orthographic views identified in the interpretation.
- `render_3d/*.png`: perspective renders. List the directory for their names.

## Intermediate operation outputs

Recorded directory: $intermediate_returns_dir

Expected paths within this directory: `<ret_name>/output.step`, `<ret_name>/projection/` and `<ret_name>/render_3d/`.

## Tasks

Read this round's `open_tickets`, including their subjects and stage responses,
and `stage_reports` from `.snapshots[-1]` in `$reconstruction_path`.

Review every entry of each stage report's `concerns` exactly once: one
`concern_reviews` entry each, keyed `<reporting_stage>.<concern_id>`.
