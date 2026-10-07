## Interpretation Round

### Task and inputs

Interpret the drawing for round $current_round. Assigned tickets: $assigned_tickets.

Your artifact is `$interpretation_output_path`. It is always seeded in full: on the first round with the input files already registered as views and `datum` left as `"???"`, and on revisions with the interpretation accepted last round. Settle the datum, read the views, and add what the drawing states. Retain unaffected entries and stable names while resolving assigned tickets. Edit the file; its contents when you answer become this round's interpretation.

### Artifact contract

- Keep the original input view's name, file, role and self-referencing full-file Region.
- A full_page input is an unsplit page in most cases. Register each orthographic view you identify on it as a separate DrawingView; at least one is required. Different roles require separate files: save each crop outside the read-only input directory. Include that view's printed dimensions and their dimension/extension/leader lines in its crop. Its Region locates the crop in the parent file; setting Region alone does not crop the file. For a single-view drawing, e.g., front-only, register that view on the same file with the full-file Region; no file copy is needed. Feature evidence may cite either file through Region.view.
- Raster crops only: exactly match the parent image cropped at Region.box_px, pixel for pixel at 1:1 scale, without resizing, rotation or preprocessing. Save observation-only crops and zooms to separate files, never over a registered view. If you change a registered crop, update its Region and all affected coordinates and measurements.
- Use box_uv in native drawing coordinates for DXF sources. For raster sources, measure box_px in the referenced file's native pixels, not a resized display. Leave image_size, scale and raster box_uv null when writing new or changed measurements; validation derives them. On revisions clear stale derived values for affected files before resubmitting.
- Define only the shared model origin in datum. The coordinate-frame table fixes the axes; do not restate or redefine them.
- Record each relevant printed dimension once in `dimensions` of the view on which it is drawn (front, top, right, section, detail, etc.). For a single-view drawing reusing the input file, still register dimensions under the orthographic view. Standalone dimension-bearing text annotations elsewhere on the page, such as ALL FILLETS R2, may be registered under full_page with a Region in page coordinates. Measure the pixel length of every readable linear dimension, and store it to `measured_length`. Radius and diameter measurements are optional; any supplied measurement joins the calibration, so make sure to measure radius for radius and diameter for diameter. Leave unreadable values null; if their pixel lengths can be measured, the fitted scale can estimate their lengths. Angles do not calibrate image scale.
- Work from the base body through major features to relevant local details. Describe each finished shape, material/void meaning, parameter anchors, directions and termination. An overall bounding box plus words such as "stepped" or "contoured" does not define a body: give its few shape-defining profile coordinates/rounds or constituent extents, and locate changes in thickness. Flat numeric lists can describe an ordered profile, with the ordering and coordinate plane explained in description. This is a compact description of the 3D feature, not a trace of every drawing primitive.
- Put numeric sizes and model positions in parameters, with lengths in mm, angles in degrees and unit direction vectors. Store numbers once; do not narrate routine arithmetic or CAD operation sequences.
- For undimensioned geometry, measure its position and size in native pixels on the views that show it. Convert the pixels to millimetres with the scale that `calculate_drawing_scale` or validation reports for that view (`mm = px * scale`). Locate positions from datum/axes; DXF is already mm. When the shape or line correspondence is ambiguous, compare the supported hypotheses across views, adopt the best-supported shape, and take the most likely value from your measurements. Provide your best numeric estimate for every required geometric parameter of that shape, including quantities that depend on an assumption. Report measurement uncertainty and assumptions in concerns, naming the affected parameters and supporting views. Omit irrelevant parameters.

DrawingInterpretation JSON schema:
```json
$interpretation_schema
```

### Work cycle

Use `load_image` to inspect drawings and `run_shell` to create crops and edit the interpretation. Use `calculate_drawing_scale` for raster scale checks.

- Base each view role on explicit labels, projection symbols and agreement between views, as the coordinate-frame section describes. If a role or the arrangement stays uncertain, adopt the best-supported roles and report the doubt.
- Cite only source Regions and dim_ names supporting each feature. Inspect hidden lines and matching projections when a silhouette permits several shapes. Reconcile projections of the same physical feature before creating separate features; a circle in one view and a rounded outline in another may describe one protrusion. Keep one mutually consistent adopted model in features; report the alternatives you rejected, naming the affected parameters.

After inspecting all views once, save the current feature descriptions and numeric parameters to `$interpretation_output_path`. Use further image inspections to review the described shapes and parameters against all views and update the JSON iteratively. Reserve turns to read the automatic validation and calibration feedback and correct errors. A valid file confirms the contract and scale checks; it does not confirm the 3D interpretation.

Successful verification fills the derived fields in that same file. Continue editing it and use the automatic feedback to correct errors and inconsistent measurements.

- Inspect scale status, inlier count/total and outlier dim_ names. Fix incorrect readings or mismatched measurement extents. A single measurement or no consensus leaves scale unknown; do not invent extra measurements to force agreement. Consensus checks consistency, not whether a printed number was read correctly.
- Before submitting, check that each dimension constrains the correct feature and anchor; nearby holes need not share a row or centerline. Check the combined outer envelope, including rounded ends and protrusions, against overall dimensions. Use the other views to check the major profiles and thickness changes.

### Submission

After the major geometry is settled, its required geometric parameters have numeric estimates, and the current artifact validates, finish tool work and return one TicketAnswers with one `responses` entry per assigned ticket, keyed by its ticket ID and none for other tickets, plus one `stage_report.concerns` entry per remaining concern those responses do not explain. Claim only changes present in that artifact.

Give the reason for any change outside your tickets in `stage_report.unticketed_changes`.
