The source program identified below has been executed and its output verified. Audit the result against the input drawing, and either accept it or name the stage to go back to.

Current semantic hypothesis:

$semantic_hypothesis

Current operation plan:

$operation_plan

Current source program location:

$output_path

Verification report:

$report

The built artifacts are in $attempt_dir, laid out as:
- `output.step` — the solid that was built
- `techdraw/` — its three-view projection, for comparison against the input DXF
- `render_3d/` — its perspective renders

If the report shows the program did not produce a solid, the directory holds no projection or renders, and the fault is in the code.
