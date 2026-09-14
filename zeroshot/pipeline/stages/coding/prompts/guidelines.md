Assign each operation's completed CadQuery result to a stable `ret_` variable by replacing its `op_` prefix. Helper functions and variables such as `part` are allowed. For example:

```python
ret_base_plate = cq.Workplane("XY").rect(80, 60).extrude(25)
ret_bore_through = ret_base_plate.faces(">Z").workplane().hole(12)
result = ret_bore_through
```

Requirements:
- The plan is a DAG: build each operation from the `ret_` results named by its `depends_on`, implementing dependencies first. The JSON list order is not the build order.
- Store the final completed CadQuery solid in `result`. Normally `result` is the final operation's `ret_` variable, not a fresh reconstruction that bypasses the planned operations.
- The script must be self-contained and must not load the input drawing or other external files at runtime.
- The generated geometry must be valid and exportable to STEP format.
- DO NOT use try-except blocks in $coding_output_path. Resolve operation failures instead of hiding them.
- If a planned operation cannot be made to work, leave the program in its best executable state and report exactly what remains incomplete in your final answer.

Verification:
Every turn you edit $coding_output_path, it is automatically executed and the final solid is exported to a STEP file. The feedback includes the execution status, return code, stdout, stderr, any executor error, a count of faces and edges by kind, and paths to the generated DXF and perspective renders under `$verification_dir/round_NNN/coding/NNN/`. This costs you no turn.

Guidelines:
- Implement the operation plan using the interpretation's shared `datum` and feature parameters for shapes, sizes, positions and directions. Preserve that model frame. References such as `sem_main_bore.radius`, `sem_main_bore.center` and `dim_bore_diameter.nominal_value` are annotated with their values; null means unknown, never zero. Evidence regions use the referenced view file's image coordinates, not model XYZ.
- Build curves as curves. An arc is one edge, not a chain of segments; a round hole is one cylindrical face, not a ring of narrow flat ones. Sampling a curve into points and joining them with straight segments is an approximation, and sampling more finely does not make it an exact curve.
- A fillet replaces a corner with a smooth transition tangent to the adjoining faces. Matching its radius alone is insufficient: attached sectors or ribs with sharp joins do not implement the fillet. An alternative construction must preserve the intended silhouette and tangency, not merely produce a valid solid.
- Read the geometry census in verification feedback. A part whose faces are all one kind, or whose edge count runs into the hundreds, may contain an unintended approximation.
- Inspect the generated DXF and perspective renders under this round's `coding/` directory using `run_shell` and `load_image` to compare the result with the source drawing.
- Iteratively refine missing or incorrect features such as cutouts, hole patterns, fillets, and chamfers, writing after each meaningful group of edits so the next verification covers it.
- Read the latest verification feedback before concluding. Distinguish execution success from geometric correctness in your ticket response; do not claim a defect was repaired merely because STEP export succeeded.
- If the plan or the interpretation looks wrong, or is missing a number you need, build your best reading, then name the `op_` or `sem_` member you doubt and what you did in your ticket response. Only the audit can open a ticket, so that response is the one place a doubt reaches it.
- Address every applicable point from review or audit feedback in the transcript.
- Your turn budget is announced in the transcript as `[turn n/N]`. Turns increment when you use tools.
