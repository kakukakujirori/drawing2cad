The program to write and maintain is:

$output_path

Keep this as one complete, self-contained CadQuery program. On a revision, inspect the existing file before editing it and preserve any implementation that remains correct.

Each planned operation has a stable `op_` name. Record the completed CadQuery result of every operation in the corresponding `ret_` variable by replacing the `op_` prefix:

```python
ret_base_plate = cq.Workplane("XY").rect(80, 60).extrude(25)
ret_bore_through = ret_base_plate.faces(">Z").workplane().hole(12)
result = ret_bore_through
```

Here `ret_base_plate` is the result of `op_base_plate`, and `ret_bore_through` is the result of `op_bore_through`. These names are the durable connection between the operation plan and the code, and let later verification inspect the model after a specific operation.

Requirements:
- Assign every planned operation's completed result to its corresponding `ret_<operation name without op_>` variable. Helper functions and variables such as `part` are allowed, but the completed result of each planned operation must still be assigned to its corresponding ret_ variable.
- Build an operation from the `ret_` results of the operations it depends on. Keep the data flow consistent with the plan rather than rebuilding an unrelated solid inside a later operation.
- Store the final completed CadQuery solid in `result`. Normally `result` is the final operation's `ret_` variable, not a fresh reconstruction that bypasses the planned operations.
- The script must be self-contained and must not load the input DXF or other external files at runtime.
- The generated geometry must be valid and exportable to STEP format.
- DO NOT use try-except blocks in $output_path. Resolve operation failures instead of hiding them.
- Do not silently omit a planned operation. If one cannot be made to work, leave the program in its best executable state and report exactly what remains incomplete in your final answer.

Verification:
Every turn you edit $output_path, it is automatically executed and the final solid is exported to a STEP file. The feedback includes the execution status, return code, stdout, stderr, any executor error, a count of faces and edges by kind, and paths to the generated DXF and perspective renders under $verification_dir. This costs you no turn, so write early enough to use the feedback before you stop.

Guidelines:
- The operation plan is a DAG. Read each entry's `depends_on` and implement dependencies before the operations that consume them; the JSON list order is not the build order. Each entry also names the semantic features it implements by their stable `sem_` names.
- Take positions from the plan and shapes and sizes from the hypothesis. A number read from the hypothesis's `evidence` is in the sheet coordinates of the view the reading names: $view_frame. Where the model's origin sits is yours to choose; choose it once and keep it consistent.
- Build curves as curves. An arc is one edge, not a chain of segments; a round hole is one cylindrical face, not a ring of narrow flat ones. Sampling a curve into points and joining them with straight segments is an approximation, and sampling more finely does not make it an exact curve.
- Read the geometry census in verification feedback. A part whose faces are all one kind, or whose edge count runs into the hundreds, may contain an unintended approximation.
- Inspect the generated DXF and perspective renders under $verification_dir using `run_shell` and `load_image` to compare the result with the source drawing.
- Iteratively refine missing or incorrect features such as cutouts, hole patterns, fillets, and chamfers, writing after each meaningful group of edits so the next verification covers it.
- Ensure that $output_path is executable before concluding your final answer.
- If the plan or the hypothesis looks wrong, or is missing a number you need, build your best reading, then name the `op_` or `sem_` member you doubt and what you did in your ticket response. Only the audit can open a ticket, so that response is the one place a doubt reaches it.
- Address every applicable point from review or audit feedback in the transcript.
- Your turn budget is announced in the transcript as `[turn n/N]`. Turns increment when you use tools.

When you stop, return only the structured `CodingSubmission` requested by the current instruction.
