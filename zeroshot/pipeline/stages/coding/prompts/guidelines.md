Assign each operation's completed CadQuery result to a stable `ret_` variable by replacing its `op_` prefix. Helper functions and variables such as `part` are allowed. For example:

```python
ret_base_plate = cq.Workplane("XY").rect(80, 60).extrude(25)
ret_bore_through = ret_base_plate.faces(">Z").workplane().hole(12)
result = ret_bore_through
```

Requirements:
- Build the operations in list order. Each `ret_` continues from the previous `ret_` unless its `detail` names the results it takes, or takes none.
- Store the final completed CadQuery solid in `result`. Normally `result` is the final operation's `ret_` variable, not a fresh reconstruction that bypasses the planned operations.
- The script must be self-contained and must not load the input drawing or other external files at runtime.
- The generated geometry must be valid and exportable to STEP format.
- DO NOT use try-except blocks in $coding_output_path. Resolve operation failures instead of hiding them.
- If a planned operation cannot be made to work, leave the program in its best executable state and report exactly what remains incomplete in your final answer.

Verification:
Every turn you edit $coding_output_path, it is automatically executed and the final solid is exported to a STEP file. The feedback includes the execution status, return code, stdout, stderr, any executor error, a count of faces and edges by kind, and paths to the generated orthographic PNGs, DXF and perspective renders under `$verification_dir/round_NNN/coding/NNN/`. This costs you no turn.

Guidelines:
- Implement the operation plan using the interpretation's shared `datum` and feature parameters for shapes, sizes, positions and directions. Preserve that model frame. References such as `sem_main_bore.radius`, `sem_main_bore.center` and `dim_bore_diameter.nominal_value` are annotated with their values; null means unknown, never zero. Evidence regions use the referenced view file's image coordinates, not model XYZ.
- Build curves as curves. An arc is one edge, not a chain of segments; a round hole is one cylindrical face, not a ring of narrow flat ones. Sampling a curve into points and joining them with straight segments is an approximation, and sampling more finely does not make it an exact curve.
- Check each feature's shape, extent, placement and connections in the final solid against the input drawing. Matching parameter values or using the named operation does not establish that its geometry is correct.
- Read the geometry census in verification feedback. A part whose faces are all one kind, or whose edge count runs into the hundreds, may contain an unintended approximation.
- Inspect every generated orthographic PNG, plus relevant DXF and perspective renders under this round's `coding/` directory using `run_shell` and `load_image` to compare the result with the source drawing.
- For widespread mismatch, check XYZ/view axes, mirroring and alignment. Correct model geometry; do not flip images to hide discrepancies. If interpreter/planner instructions contradict the drawing, record concerns with the suspect view_/sem_/op_ IDs and evidence.
- Iteratively refine missing or incorrect features such as cutouts, hole patterns, fillets, and chamfers, writing after each meaningful group of edits so the next verification covers it.
- Read the latest verification feedback before concluding. Distinguish execution success from geometric correctness in your ticket response; do not claim a defect was repaired merely because STEP export succeeded.
- In `dimension_checks`, cover every supplied dim ID once. For each, identify its realization in the final geometry and the evidence that it holds, or explicitly state why it remains unestablished or unverified. When measured, name the geometry measured and the result; assigning a number to a variable alone does not prove it survived later operations. Use `{}` if no dimensions exist. Do not repeat this table in ticket summaries or concerns, or narrate routine unit/radius conversions.
- If drawing evidence contradicts the plan or interpretation, correct the geometry in $coding_output_path while preserving operation identities. Leave upstream JSON files unchanged; report the affected `op_` or `sem_` member, evidence and what you changed. If the evidence does not establish a correction, state the unresolved mismatch rather than claiming plan compliance resolves it. Only the audit can open a ticket.
- Address every applicable point from review or audit feedback in the transcript.
- Your turn budget is announced in the transcript as `[turn n/N]`. Turns increment when you use tools.
