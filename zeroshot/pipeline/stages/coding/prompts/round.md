## Coding Round

### Task and inputs

Implement the complete CadQuery program for reconstruction round $current_round.

Tickets assigned to coding this round: $assigned_tickets

Create or inspect the program at `$coding_output_path`. Preserve operation identities and implement their geometry using the source drawing as the authority and drawing interpretation deliverable as a hint. In a revision round, preserve code blocks that remains correct and address the current snapshot and assigned tickets.

Current dimension readings (all registered IDs, including those the plan does not cite):
```json
$dimension_inventory
```

### Artifact contract

The source drawing takes precedence over upstream interpretations and plans. An adopted upstream interpretation is not evidence for retaining current geometry or rejecting an alternative.

Do not wait for upstream agreement before implementing a correction supported by drawing evidence and a rendered trial. Leave upstream JSON files unchanged and preserve operation identities. Report the geometry changed, supporting evidence and affected `view_`/`dim_`/`sem_`/`op_` IDs in the relevant ticket response, or in `stage_report.concerns` when no assigned ticket covers the issue. If evidence is insufficient, retain the best executable model and state the unresolved mismatch; plan compliance does not resolve it.

Assign each operation's completed CadQuery result to a stable `ret_` variable by replacing its `op_` prefix. Helper functions and variables such as `part` are allowed. For example:

```python
ret_base_plate = cq.Workplane("XY").rect(80, 60).extrude(25)
ret_bore_through = ret_base_plate.faces(">Z").workplane().hole(12)
result = ret_bore_through
```

- Build the operations in list order. Each `ret_` continues from the previous `ret_` unless its `detail` names the results it takes, or takes none.
- Store the final completed CadQuery solid in `result`. Normally `result` is the final operation's `ret_` variable, not a fresh reconstruction that bypasses the planned operations.
- The script must be self-contained and must not load the input drawing or other external files at runtime.
- The generated geometry must be valid and exportable to STEP format.
- DO NOT use try-except blocks in $coding_output_path. Resolve operation failures instead of hiding them.
- Preserve the interpretation's shared `datum` and model frame. Use feature parameters as the starting geometry. References such as `sem_main_bore.radius`, `sem_main_bore.center` and `dim_bore_diameter.nominal_value` are annotated with their values; null means unknown, never zero. Evidence regions use the referenced view file's image coordinates, not model XYZ.
- Build curves as curves. An arc is one edge, not a chain of segments; a round hole is one cylindrical face, not a ring of narrow flat ones. Sampling a curve into points and joining them with straight segments is an approximation, and sampling more finely does not make it an exact curve.

### Work cycle

After a tool turn changes `$coding_output_path`, automatic verification executes it and returns diagnostics and paths to STEP, orthographic PNG/DXF and perspective renders under `$verification_dir/round_NNN/coding/NNN/`. Verification uses no additional model turn.

#### [IMPORTANT] Test geometric hypotheses

Do not believe in your 3D reasoning blindly. Whenever deciding geometric design details, build candidate CAD parts, render the relevant views, and compare the resulting images with the input drawing. This is especially important when you seek an alternative design hypothesis: do not reject it based only on your reasoning. Build and observe it first.

- When testing, choose a candidate, the smallest change needed to test it, and the deciding views; then run the trial before further speculation. Before revisiting the same question, obtain a new measurement or run a new trial.
- Use any scratch Python file separate from `$coding_output_path`, including the suspect feature and enough surrounding geometry to check its extent and connections. No `op_`/`ret_` structure is required. Scratch files are not automatically executed, verified or submitted. You will need to manually write `cq.exporters.export(...)` to generate the STEP file.
- Execute with `run_shell`, export STEP, then call `render_step` with its path and deciding views (e.g., `views=["front", "top"]`). Open the returned PNGs with `load_image` and compare with the source. Known views use registered drawing axes; the tool reports the axes used.
- Integrate supported corrections into `$coding_output_path` after each meaningful group of changes and inspect the full-model verification. After integrating a trial into $coding_output_path, inspect the updated full-model verification and renders. After further geometry changes, inspect newly generated images.
- Treat stroke thickness as a drawing convention, not a physical feature width.
- Read the geometry census:
  - All faces of one kind or hundreds of edges may indicate an unintended approximation.
  - A `ret_` split into several solids can mean the feature's intended shape is wrong, not only its construction. Before you add material or values the drawing does not show, derive the feature again from every view and build a candidate with a different topology.
- Inspect final orthographic PNG and relevant perspective renders with `load_image`; use ezdxf for relevant DXF measurements. Compare feature extent, placement and connections. For widespread mismatch, check XYZ/view axes, mirroring and alignment.
- If a kernel operation keeps failing, inspect its geometric preconditions. Try another construction, or reconsider whether the intended feature is right. Do not silently skip it.

### Submission

- Read the latest verification feedback before concluding. Distinguish execution success from geometric correctness in your ticket response; do not claim a defect was repaired merely because STEP export succeeded.
- State which geometric conclusions you checked against the latest images or measurements. Keep untested predictions and unavailable views explicitly unverified.
- If a planned operation cannot be made to work, leave the program in its best executable state and report exactly what remains incomplete in your final answer.

Finish tool work and return one `TicketAnswers` with your ticket responses, one `stage_report.concerns` entry per remaining concern those responses do not explain, and `dimension_checks`. Revise the program in the workspace; the pipeline captures it through verification. Give one `responses` entry per assigned ticket, keyed by its ticket ID and none for any other, naming the concrete `ret_...` results or `result` that you implemented, changed, or examined.

Give the reason for any ticket-external `ret_...` change in `stage_report.unticketed_changes`. Reporting an upstream conflict in concerns does not replace this revision-scope explanation. Only the audit can open a ticket.

- In `dimension_checks`, cover every supplied dim ID once. For each, identify its realization in the final geometry and the evidence that it holds, or explicitly state why it remains unestablished or unverified. When measured, name the geometry measured and the result; assigning a number to a variable alone does not prove it survived later operations. Use `{}` if no dimensions exist. Do not repeat this table in ticket summaries or concerns, or narrate routine unit/radius conversions.
