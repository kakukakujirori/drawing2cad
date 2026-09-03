Coding and verification for reconstruction round $current_round are complete. Audit that immutable snapshot against the input drawing.

The current workspace source is at:

$output_path

Numbers in semantic evidence are in the sheet coordinates of the view each reading names: $view_frame.

The built artifacts are in $attempt_dir, laid out as:
- `output.step` — the solid that was built
- `techdraw.dxf` — its three-view projection, for comparison against the input DXF
- `render_3d/<style>.png` — its perspective renders, named as the inputs are
$intermediate_returns

If the report shows the program did not produce a solid, the directory holds no projection or renders.

If a feature visible in the input drawing has no corresponding `sem_...` member, there is no existing causal chain to traverse. Do not invent evidence-free hops through unrelated outputs. For the missing feature, leave the `backtrace` empty and return an `add` revision request targeting the whole semantics stage (`name: null`), and propose one or more stable `sem_...` names.

Do not edit the reconstruction history, source program, or generated artifacts. Return one `AuditReport`; the pipeline will validate every named reference and causal link against the immutable snapshot.
