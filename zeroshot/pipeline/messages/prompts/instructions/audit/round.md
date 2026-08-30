Coding and verification for reconstruction round $current_round are complete. Audit that immutable snapshot against the input drawing.

The complete reconstruction history is stored at:

$reconstruction_path

Inspect it with `jq`, `rg`, or another read-only shell command. The final item in `snapshots` is the round being audited and contains its open tickets, semantic hypothesis, operation plan, complete `model.py` source, and verification result.

The current workspace source is at:

$output_path

Numbers in semantic evidence are in the sheet coordinates of the view each reading names: $view_frame.

The built artifacts are in $attempt_dir, laid out as:
- `output.step` — the solid that was built
- `techdraw.dxf` — its three-view projection, for comparison against the input DXF
- `render_3d/<style>.png` — its perspective renders, named as the inputs are

If the report shows the program did not produce a solid, the directory holds no projection or renders.

Do not edit the reconstruction history, source program, or generated artifacts. Return one `AuditReport`; the pipeline will validate every named reference and causal link against the immutable snapshot.
