Requirements:
- Store the final completed CadQuery solid in `result`.
- The script must be self-contained and must not load the input drawing or other external files at runtime.
- The generated geometry must be valid and exportable to STEP format.
- DO NOT use try-except blocks in $coding_output_path. Resolve failures instead of hiding them.
- If a feature cannot be made to work, leave the program in its best executable state and report exactly what remains incomplete in your final answer.

Verification:
Every turn you edit $coding_output_path, it is automatically executed and the final solid is exported to a STEP file. The feedback includes the execution status, return code, stdout, stderr, any executor error, a count of faces and edges by kind, and paths to the generated DXF and perspective renders under `$verification_dir/round_NNN/coding/NNN/`. This costs you no turn.

Guidelines:
- Take shapes, sizes, positions and directions from the printed dimensions. Keep one model frame that follows the coordinate frames above.
- Build curves as curves. An arc is one edge, not a chain of segments; a round hole is one cylindrical face, not a ring of narrow flat ones. Sampling a curve into points and joining them with straight segments is an approximation, and sampling more finely does not make it an exact curve.
- A fillet replaces a corner with a smooth transition tangent to the adjoining faces. Matching its radius alone is insufficient: attached sectors or ribs with sharp joins do not implement the fillet. An alternative construction must preserve the intended silhouette and tangency, not merely produce a valid solid.
- Read the geometry census in verification feedback. A part whose faces are all one kind, or whose edge count runs into the hundreds, may contain an unintended approximation.
- Inspect the generated DXF and perspective renders under this round's `coding/` directory using `run_shell` and `load_image` to compare the result with the source drawing.
- Iteratively refine missing or incorrect features such as cutouts, hole patterns, fillets, and chamfers, writing after each meaningful group of edits so the next verification covers it.
- Read the latest verification feedback before concluding. Distinguish execution success from geometric correctness in your report; do not claim a defect was repaired merely because STEP export succeeded.
- Your turn budget is announced in the transcript as `[turn n/N]`. Turns increment when you use tools.
