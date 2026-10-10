## Coding Round

### Task and inputs

Implement the complete CadQuery program for reconstruction round $current_round.

Tickets assigned to coding this round: $assigned_tickets

Create or inspect the program at `$coding_output_path`. Implement the geometry using the source drawing as the authority and the drawing interpretation deliverable as a hint. In a revision round, preserve code blocks that remains correct and address the current snapshot and assigned tickets.

Current dimension readings (all registered IDs):
```json
$dimension_inventory
```

### Artifact contract

The source drawing takes precedence over the upstream interpretation. An adopted upstream interpretation is not evidence for retaining current geometry or rejecting an alternative.

Do not wait for upstream agreement before implementing a correction supported by drawing evidence and a rendered trial. Leave upstream JSON files unchanged. Report the geometry changed, supporting evidence and affected `view_`/`dim_`/`sem_` IDs in the relevant ticket response, or in `stage_report.concerns` when no assigned ticket covers the issue. When a feature's shape is uncertain, actively test as many hypotheses as you have on the whole model in `$coding_output_path` and compare their projections with the input. Adopt the hypothesis that matches best. Leaving the feature out is one of these hypotheses, not a safe default. Do not reject a hypothesis on a guess without evidence.

- Store the final completed CadQuery solid in `result`.
- The script must be self-contained and must not load the input drawing or other external files at runtime.
- The generated geometry must be valid and exportable to STEP format.
- DO NOT use try-except blocks in $coding_output_path. Resolve failures instead of hiding them.
- Preserve the interpretation's shared `datum` and model frame. Use feature parameters as the starting geometry. References such as `sem_main_bore.radius`, `sem_main_bore.center` and `dim_bore_diameter.nominal_value` are annotated with their values; null means unknown, never zero. Evidence regions use the referenced view file's image coordinates, not model XYZ.
- Build curves as curves. An arc is one edge, not a chain of segments; a round hole is one cylindrical face, not a ring of narrow flat ones. Sampling a curve into points and joining them with straight segments is an approximation, and sampling more finely does not make it an exact curve.

### Work cycle

After a tool turn changes `$coding_output_path`, automatic verification executes it and returns diagnostics and paths to STEP, orthographic PNG/DXF and perspective renders under `$verification_dir/round_NNN/coding/NNN/`. Verification uses no additional model turn.

#### [IMPORTANT] Test geometric hypotheses

Do not believe in your 3D reasoning blindly. Whenever deciding geometric design details, build candidate CAD parts, render the relevant views, and compare the resulting images with the input drawing. This is especially important when you seek an alternative design hypothesis: do not reject it based only on your reasoning. Build and observe it first.

- After any modifications, inspect the latest deciding views and overall scores before keeping or undoing the change.　If the expected change is absent, revise the geometry or coordinate hypothesis before another trial.
- Use any scratch Python file separate from `$coding_output_path` to experiment with part of the shape, including the suspect feature and enough surrounding geometry to check its extent and connections. Scratch files are not automatically executed, verified or submitted. You will need to manually write `cq.exporters.export(...)` to generate the STEP file.
- Execute with `run_shell`, export STEP, then call `render_step` with its path and deciding views (e.g., `views=["front", "top"]`). Open the returned PNGs with `load_image` and compare with the source. Known views use registered drawing axes; the tool reports the axes used.
- Integrate any promising part experiments into `$coding_output_path` immediately to see whether the projections of the resulting shape give a better match with the input. Quick iteration is the key to fast improvements.
- Treat stroke thickness as a drawing convention, not a physical feature width.
- Read the geometry census:
  - All faces of one kind or hundreds of edges may indicate an unintended approximation.
  - Before rejecting a geometric hypothesis because a build failed, use `run_shell` to inspect each separated component's volume and location and identify the step that separated them. Repair or isolate that failure, then compare the hypothesis with the source views. Build failure alone does not establish that the proposed shape is wrong.
- Inspect final orthographic PNG and relevant perspective renders with `load_image`; use ezdxf for relevant DXF measurements. Compare feature extent, placement and connections. For widespread mismatch, check XYZ/view axes, mirroring and alignment.
- If a kernel operation keeps failing, inspect its geometric preconditions. Try another construction, or reconsider whether the intended feature is right. Do not silently skip it.

### Submission

- Read the latest verification feedback before concluding. Distinguish execution success from geometric correctness in your ticket response; do not claim a defect was repaired merely because STEP export succeeded.
- State which geometric conclusions you checked against the latest images or measurements. Keep untested predictions and unavailable views explicitly unverified.
- If a feature cannot be made to work, leave the program in its best executable state and report exactly what remains incomplete in your final answer.

Finish tool work and return one `TicketAnswers` with your ticket responses and one `stage_report.concerns` entry per remaining concern those responses do not explain. Revise the program in the workspace; the pipeline captures it through verification. Give one `responses` entry per assigned ticket, keyed by its ticket ID and none for any other.

Only the audit can open a ticket.
