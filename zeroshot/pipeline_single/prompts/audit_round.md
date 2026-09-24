## Audit Round
Coding and verification for reconstruction round $current_round are complete. Audit that result against the input drawing.

The program is at `$coding_output_path`.

Built artifacts are in $attempt_dir:
- `output.step`: the built solid.
- `projection/<view>.dxf` and `.png`: projections in the six views of the Orthographic view directions table: front, back, top, bottom, left and right.
- `render_3d/*.png`: perspective renders. List the directory for their names.

Compare each view in the input drawing with the projection from the same direction.

Verification status: $verification_status

The coder's report:
$coding_report
