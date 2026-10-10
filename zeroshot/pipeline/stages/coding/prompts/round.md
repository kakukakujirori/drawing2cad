## Coding Round

### Task and inputs

Reconstruct the drawing for round $current_round.

Tickets assigned to coding this round: $assigned_tickets

You own two files:

- `$interpretation_output_path`: your reading of the drawing. It is always seeded in full: in round 0 with the input files already registered as views and `datum` left as `"???"`, and in later rounds with the interpretation accepted last round.
- `$coding_output_path`: the CadQuery program. In later rounds it holds last round's program.

Write the interpretation first. The pipeline builds the program only after the interpretation validates, because it projects the solid into the views the interpretation registers. In a revision round, keep unaffected entries, stable names and correct code while resolving the tickets.

### Artifact contract

#### Interpretation

- Keep the original input view's name, file, role and self-referencing full-file Region.
- A full_page input is an unsplit page in most cases. Register each orthographic view you identify on it as a separate DrawingView; at least one is required. Different roles require separate files: save each crop outside the read-only input directory. Include that view's printed dimensions and their dimension/extension/leader lines in its crop. Its Region locates the crop in the parent file; setting Region alone does not crop the file. For a single-view drawing, e.g., front-only, register that view on the same file with the full-file Region; no file copy is needed. Feature evidence may cite either file through Region.view.
- Raster crops only: exactly match the parent image cropped at Region.box_px, pixel for pixel at 1:1 scale, without resizing, rotation or preprocessing. Save observation-only crops and zooms to separate files, never over a registered view. If you change a registered crop, update its Region and all affected coordinates and measurements.
- Use box_uv in native drawing coordinates for DXF sources. For raster sources, measure box_px in the referenced file's native pixels, not a resized display. Leave image_size, scale and raster box_uv null when writing new or changed measurements; validation derives them. On revisions clear stale derived values for affected files before the next validation.
- Define only the shared model origin in datum. The coordinate-frame table fixes the axes; do not restate or redefine them.
- Record each relevant printed dimension once in `dimensions` of the view on which it is drawn (front, top, right, section, detail, etc.). For a single-view drawing reusing the input file, still register dimensions under the orthographic view. Standalone dimension-bearing text annotations elsewhere on the page, such as ALL FILLETS R2, may be registered under full_page with a Region in page coordinates. Measure the pixel length of every readable linear dimension, and store it to `measured_length`. Radius and diameter measurements are optional; any supplied measurement joins the calibration, so measure radius for radius and diameter for diameter. Leave unreadable values null; if their pixel lengths can be measured, the fitted scale can estimate their lengths. Angles do not calibrate image scale.
- Work from the base body through major features to relevant local details. Describe each finished shape, material/void meaning, parameter anchors, directions and termination. An overall bounding box plus words such as "stepped" or "contoured" does not define a body: give its few shape-defining profile coordinates/rounds or constituent extents, and locate changes in thickness. Flat numeric lists can describe an ordered profile, with the ordering and coordinate plane explained in description. This is a compact description of the 3D feature, not a trace of every drawing primitive.
- Put numeric sizes and model positions in parameters, with lengths in mm, angles in degrees and unit direction vectors. Store numbers once; do not narrate routine arithmetic or CAD operation sequences.
- For undimensioned geometry, measure its position and size in native pixels on the views that show it. Convert the pixels to millimetres with the scale that `calculate_drawing_scale` or validation reports for that view (`mm = px * scale`). Locate positions from datum/axes; DXF is already mm. When the shape or line correspondence is ambiguous, compare the supported hypotheses across views, adopt the best-supported shape, and take the most likely value from your measurements. Provide your best numeric estimate for every required geometric parameter of that shape, including quantities that depend on an assumption. Report measurement uncertainty and assumptions in concerns, naming the affected parameters and supporting views. Omit irrelevant parameters.

DrawingInterpretation JSON schema:
```json
$interpretation_schema
```

#### Program

The source drawing is the authority. The interpretation is your record of it, not evidence for keeping current geometry or rejecting an alternative. When a trial build shows that a reading is wrong, correct the interpretation as well as the program.

When a feature's shape is uncertain, test each hypothesis you have on the whole model in `$coding_output_path` and compare its projections with the input. Only builds of `$coding_output_path` are compared with the drawing and scored. Adopt the hypothesis that matches best. Leaving the feature out is one of these hypotheses, not a safe default. Do not reject a hypothesis on a guess without evidence.

- Store the final completed CadQuery solid in `result`.
- The script must be self-contained and must not load the input drawing or other external files at runtime.
- The generated geometry must be valid and exportable to STEP format.
- DO NOT use try-except blocks in $coding_output_path. Resolve failures instead of hiding them.
- Use the interpretation's shared `datum` and model frame, and its feature parameters as the starting geometry. In the reconstruction record, references such as `sem_main_bore.radius` and `dim_bore_diameter.nominal_value` are annotated with their values; null means unknown, never zero. Evidence regions use the referenced view file's image coordinates, not model XYZ.
- Build curves as curves. An arc is one edge, not a chain of segments; a round hole is one cylindrical face, not a ring of narrow flat ones. Sampling a curve into points and joining them with straight segments is an approximation, and sampling more finely does not make it an exact curve.

### Work cycle

Use `load_image` to inspect drawings, `run_shell` to create crops and edit files, `calculate_drawing_scale` for raster scale checks, and `render_step` to render trial STEP files.

After a tool turn changes either file, automatic verification checks the interpretation. Once it is valid, verification also executes the program and returns diagnostics and paths to STEP, orthographic PNG/DXF and perspective renders under `$verification_dir/round_NNN/coding/NNN/`. Each report lists the requirements your answer still has to meet. Verification uses no additional model turn.

#### Read the drawing

- Base each view role on explicit labels, projection symbols and agreement between views, as the coordinate-frame section describes. If a role or the arrangement stays uncertain, adopt the best-supported roles and report the doubt.
- Cite only source Regions and dim_ names supporting each feature. Inspect hidden lines and matching projections when a silhouette permits several shapes. Reconcile projections of the same physical feature before creating separate features; a circle in one view and a rounded outline in another may describe one protrusion. Keep one mutually consistent adopted model in features; report the alternatives you rejected, naming the affected parameters.
- After inspecting all views once, save the views, dimensions and current feature descriptions with numeric parameters to `$interpretation_output_path`. Use further image inspections to review the described shapes and parameters against all views and update the JSON iteratively. Reserve turns to read the automatic validation and calibration feedback and correct errors. A valid file confirms the contract and scale checks; it does not confirm the 3D interpretation.
- Successful verification fills the derived fields in that same file. Continue editing it and use the automatic feedback to correct errors and inconsistent measurements. Inspect scale status, inlier count/total and outlier dim_ names. Fix incorrect readings or mismatched measurement extents. A single measurement or no consensus leaves scale unknown; do not invent extra measurements to force agreement. Consensus checks consistency, not whether a printed number was read correctly.

#### [IMPORTANT] Test geometric hypotheses

Do not believe in your 3D reasoning blindly. Whenever deciding geometric design details, build candidate CAD parts, render the relevant views, and compare the resulting images with the input drawing. This is especially important when you seek an alternative design hypothesis: do not reject it based only on your reasoning. Build and observe it first.

- After any modifications, inspect the latest deciding views and overall scores before keeping or undoing the change. If the expected change is absent, revise the geometry or coordinate hypothesis before another trial.
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

- Read the latest verification feedback before concluding. Distinguish execution success from geometric correctness in your ticket responses; do not claim a defect was repaired merely because STEP export succeeded.
- Check that each dimension constrains the correct feature and anchor; nearby holes need not share a row or centerline. Check the combined outer envelope, including rounded ends and protrusions, against overall dimensions. Use the other views to check the major profiles and thickness changes.
- State which geometric conclusions you checked against the latest images or measurements. Keep untested predictions and unavailable views explicitly unverified.
- If a feature cannot be made to work, leave the program in its best executable state and report exactly what remains incomplete.

Finish tool work and return one `TicketAnswers` with your ticket responses and one `stage_report.concerns` entry per remaining concern those responses do not explain. Give one `responses` entry per ticket, keyed by its ticket ID and none for any other. Report the geometry changed, its supporting evidence and the affected `view_`/`dim_`/`sem_` names in the relevant ticket response. The pipeline captures both files through verification; claim only changes present in them.

Only the audit can open a ticket.
