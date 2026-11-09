The coding and refinement rounds have finished. No further coding rounds will run. Your task is to select the CAD reconstruction that best matches the input drawing.

The [Candidate submissions] list gives each candidate's round, program, built STEP, and artifact directory. Read each candidate's interpretation and verification report from the matching round's snapshot in $reconstruction_path. Its program is available at the listed program path. Compare every candidate with the original input drawing using load_image, and inspect the program or built STEP when it resolves a decisive difference. Earlier source crop files may have been overwritten; use the original input drawing and the recorded crop regions, or the saved comparison images.

Comparison image conventions:

- Overlay: input lines are pale gray; projected lines are blue near the input and red when far away or solid/dashed styles differ. Blue does not fully guarantee a correct solid.
- Unmatched: blue means both images draw the line, green means input only, and magenta means CAD only. Gray marks annotation, optional tangents and other ink not compared. Colored bands and IDs label mismatch clusters; hatching marks filled-outline differences.
- Optional tangent boundaries may be omitted. Dimension, leader and text lines are annotations, not missing CAD features. Clusters show only the largest differences; inspect unlabelled lines too.

Judge overall dimensions, major profiles, thickness, feature placement, material and voids, openings and their termination across the available views. Check visible and hidden boundaries. Check that straight edges and circular arcs shown in the drawing are preserved in the built STEP rather than approximated with splines. A smaller projection error is evidence about projection alignment, not a sufficient condition of a better solid. Check whether an apparent improvement loses a feature or a boundary supported by another source view. Account for uncertain alignment and optional tangent boundaries.

Choose the most faithful existing candidate even if all candidates have defects. Prefer neither the latest round nor the smallest projection score merely for that reason. Explain the decisive comparison and the chosen candidate's remaining defects in `reason`.

This is a selection task. Do not edit the candidates, workspace model.py, interpretation.json or reconstruction.json, and do not create a new model or open revision tickets. Use the workspace temporary directory for measurements.

Return one FinalSelection with the chosen `round` and your `reason`. The pipeline restores that candidate's program and interpretation and submits its built STEP.
