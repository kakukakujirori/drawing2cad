The program to write is:

$output_path

It is already laid out for you: every operation in the plan has a marker there, in build order, giving the step's name and the plan's own line for it.

Example:
```
# ---- op_base_plate extrude (needs nothing; builds sem_base_body)
# ---- Extrude the front-view outline 25 mm along +z.
[Write your code block for this operation here]
```

The file is one program and runs top to bottom, so a section stands on what the sections above it built. The last marker is `# ---- result`, which is not a step: put the finished solid there. Your code under that one is kept whatever the plan does; a step's section is not.

Requirements:
- Every line beginning `# ---- ` is written from the plan. Do not edit, move, reorder or add one; a marker you write yourself marks nothing, and code under the wrong marker is attributed to the wrong step.
- Put each operation's code under its own marker, and nothing else there. Imports and anything shared go above the first marker.
- The script must be self-contained and must not load the input DXF or external files at runtime.
- Store the final completed CadQuery solid in the `result` variable, under the `# ---- result` marker at the end of $output_path.
- The generated geometry must be valid and exportable to STEP format.
- DO NOT use try-except blocks in $output_path. Resolve operation failures instead of hiding them.
- When an operation in the plan cannot be made to work, leave a comment of a few lines under its marker, saying what stopped it, and carry on. Report it in your final answer as well. Do not remove it silently.
- When the plan is revised, the file is laid out again. A section whose step is unchanged keeps your code. A section whose step changed is emptied, and you are told by name which ones, so nothing goes missing without being said.

Verification:
Every turn you edit $output_path, it is automatically executed and the final solid is exported to a STEP file along with: the execution status, the return code, stdout, stderr, any executor error, a count of the faces and edges it is made of by kind, the paths of its DXF and perspective renders under $verification_dir, and how many of the plan's sections are written. This costs you no turn, so write early and often rather than saving the check for the end.

Guidelines:
- The plan is already in build order, worked out from the dependencies it states, and the markers in $output_path follow it. Each entry names its step and the hypothesis features it builds by their stable `sem_` names.
- Take positions from the plan and shapes and sizes from the hypothesis. A number read off the hypothesis's `evidence` is in the sheet coordinates of the view the reading names: $view_frame. Where the model's origin sits is yours to choose; choose it once and hold to it, since the plan places operations against features rather than against an origin.
- Fill in the sections in order, then read the verification that comes back before writing again. Writing several sections in one turn is cheaper than one at a time, so do not hold back for the sake of it.
- Build curves as curves. An arc is one edge, not a chain of segments; a round hole is one cylindrical face, not a ring of narrow flat ones. Sampling a curve into points and joining them with straight segments is the mistake this warns against, and sampling it more finely makes it worse rather than better.
- The verification reports what you built, counted by kind: `faces 85 (Cylinder 42, Plane 33, ...)`. Read it. A part whose faces are all one kind, or whose edge count runs into the hundreds, has been approximated somewhere.
- Inspect the generated DXF / perspective renders under $verification_dir using `run_shell` / `load_image` to verify visual alignment with the source drawing.
- Iteratively refine and add missing features (cutouts, hole patterns, fillets, chamfers), writing after each major addition so the next verification covers it.
- Ensure that $output_path is executable before concluding your final answer.
- If feedback from a review or audit step is present in the transcript, address every point it raises.
- Your turn budget is announced in the transcript as `[turn n/N]`. Turns increment by using tools.

Answer in prose, describing what the script builds and what, if anything, it could not reproduce.
