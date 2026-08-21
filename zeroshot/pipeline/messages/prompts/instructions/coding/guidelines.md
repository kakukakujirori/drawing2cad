Write the program to:

$output_path

Requirements:
- The script must be self-contained and must not load the input DXF or external files at runtime.
- Store the final completed CadQuery solid in the `result` variable in $output_path.
- The generated geometry must be valid and exportable to STEP format.
- DO NOT use try-except blocks in $output_path. Resolve operation failures instead of hiding them.
- When an operation in the plan cannot be made to work, drop it from the script and report it in your final answer.

Verification:
Every turn you edit $output_path, it is automatically executed and the final solid is exported to a STEP file along with: the execution status, the return code, stdout, stderr, any executor error, and the paths of its DXF and perspective renders under $verification_dir. This costs you no turn, so write early and often rather than saving the check for the end.

Guidelines:
- Write an initial script implementing the operations in plan order, then read the verification that comes back before writing again.
- Inspect the generated DXF / perspective renders under $verification_dir using `run_shell` / `load_image` to verify visual alignment with the source drawing.
- Iteratively refine and add missing features (cutouts, hole patterns, fillets, chamfers), writing after each major addition so the next verification covers it.
- Ensure that $output_path is executable before concluding your final answer.
- If feedback from a review or audit step is present in the transcript, address every point it raises.
- Your turn budget is announced in the transcript as `[turn n/N]`. Turns increment by using tools.

Answer in prose, describing what the script builds and what, if anything, it could not reproduce.
