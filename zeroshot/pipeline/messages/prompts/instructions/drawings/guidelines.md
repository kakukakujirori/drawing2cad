Guidelines:

Reading the input
- The input message names each sheet you were given and says what it is; those sheets are the source of truth for the transcription.
- Keep supplied perspective sheets with their original name, role and file for qualitative comparison downstream. A perspective render has no uniform orthographic scale: do not calibrate it or invent millimetre primitives or printed dimensions for it. Its `evidence` and `dimensions` may remain empty.
- Where a sheet arrives undivided, separate the views yourself by where the linework sits: they are not separated by layer or by file. Every entity in the DXF sheets seen so far is on layer `0`, so the layer name tells you nothing.
- Measure with code, never by eye. `run_shell` gives you the same tools for a raster sheet that `ezdxf` gives you for a vector one: read the image with OpenCV or numpy, find the linework, and compute the numbers. A coordinate guessed off a picture is worse than one you admit you do not have.

What a sheet is
- A sheet is a page you were handed or a view you cut out of one, and both remain in the complete `sheets` list in $drawing_output_path. A page carrying three views gives you four sheets: the page as it arrived, and three views taken from it.
- A view you cut out names the sheet it came from in `crop_of`, with the region it covers in that sheet's own coordinates. The input message announces each sheet as `name: path`, so the name is what stands before the colon and never the file.
- Write every view you cut out to a file of its own and name that file in `file`. A sheet you were handed keeps the file it arrived as.
- `label` is the caption printed on the view -- `SECTION A-A`, `FRONT SIDE VIEW` -- or null when it carries none. Read it rather than guess `role` around it.
- `role` is what the view shows. Give a page you have not separated `full_page`, and a view you could not identify `unknown`. Two sheets must not claim the same orthographic role.

Each sheet's own coordinates
- **Every sheet keeps its own UV coordinates**: u runs right and v runs up, both in millimetres. A view you cut out is measured in its own UV frame, not the page's. Keep this 2D UV frame even when its axes happen to coincide with model XYZ; left and bottom views have different signed projections. Pixels belong nowhere in your answer.
- For a raster, put `(u, v) = (0, 0)` at the **lower-left corner of the bottom-left pixel**, not at that pixel's centre. Before applying scale, an image `w` columns by `h` rows therefore spans `0 <= u <= w` and `0 <= v <= h`.
- OpenCV/numpy index pixels from the top-left. The lower-left corner of zero-based pixel `(row=r, column=c)` is `(u_px, v_px) = (c, h - r - 1)`, and its centre is `(c + 0.5, h - r - 0.5)`. Multiply either pixel position by `millimetres_per_pixel` after choosing the corner or centre that the measured line location represents.
- `scale` is how many millimetres one unit of what you measured is worth: the factor you converted by, or 1.0 when you were already measuring millimetres. A vector sheet is already in the drawing's units.
- On a raster sheet, pair each printed dimension with the pixel length of the linework it measures and give those pairs to `calculate_drawing_scale`. It fits one factor and names the pairs that disagree with it. Take its answer as `scale`, and do not average the ratios by hand.
- The tool refuses to choose when the pairs do not agree. A refusal means a misread figure, a mismatched pair, or two views at different scales -- fix that rather than picking one of the alternatives it lists.
- The axes each view fixes are not yours to choose. $view_frame.

Transcribing the linework
- `evidence` is the sheet transcribed: one entry per entity the view draws, with the entity type **as drawn** rather than what you think it is in 3D -- in an orthographic view a circle seen obliquely draws as an ellipse. A page you have separated into views draws nothing itself, so its own `evidence` is empty.
- Whether an edge is visible or hidden is how depth is read -- whether a hole is through or blind, where a pocket stops. In a vector sheet the linetype says so (`Continuous` seen, `HIDDEN` behind material); in a raster sheet the dashes do. A centre line and a phantom line have styles of their own: record them under those rather than as an edge of the part.
- Give an entry **every** number its entity takes. Your answer's schema lists them. A half-transcribed entity is not evidence, and neither is one that states no numbers at all.
- `source` names the printed figures the entry rests on, by their `dim_` names on this same sheet: the diameter beside a circle, the length between two edges. Leave it empty when nothing is printed and you measured the numbers or read them out of a vector definition.

Transcribing the annotations
- `dimensions` is every figure printed on the sheet: the text exactly as printed, the size as a number, how many features it covers, and whatever it says in words beyond the size. What `kind` and `quantity` can carry belongs there rather than in `note`.
- A figure you could read but could not tie to any linework is still worth having. Record it, and cite it from nothing.
- **A printed figure gives a size, never a place.** Keep the printed nominal in `Dimension`; for an entity's radius, use a printed radius directly or divide a printed diameter by two. In scale calibration, pair a printed diameter with a measured pixel diameter, never a pixel radius. A length between two features says nothing about where either one sits: leave those points where you measured them. Moving a point until the printed number comes out makes "the figure equals the distance I measured" true by construction, and that identity is the only thing that tells a wrong scale from a wrong target from a misreading.

Naming
- Give every sheet a stable `name` beginning `sheet_`, every entry one beginning `ev_`, and every figure one beginning `dim_`, all in lower_snake_case and all unique across the whole drawing: `sheet_front`, `ev_front_outer_circle`, `dim_bore_diameter`. Later stages cite `ev_front_outer_circle.center`, and the audit names a whole sheet as `sheet_front`.
- A name is an identity, not a display label. Keep it when you re-read the thing it names, and give something new a new name.

Working
- If feedback from a review or audit step is present in the transcript, address every point it raises.
- $drawing_output_path is the complete, authoritative working `DrawingSource`. Edit it with code so it always remains valid JSON; do not edit `$reconstruction_path`.
- After any tool-using turn that changes the file, it is validated and its evidence is exported as DXF and PNG under `$verification_dir/round_NNN/drawing/NNN/`. The automatic visual feedback is then returned. Open the rendered views with `load_image` and compare them with the input before deciding the reading is correct.
- A submission is accepted only after the current file has been rendered and that feedback has appeared in the transcript. If your inspection identifies any remaining defect or edit, do not submit in that turn: revise the JSON and inspect the next rendering first. The highest-numbered drawing attempt in this round is the version that is submitted.
- Your turn budget is announced in the transcript as `[turn n/N]`. Turns increment by using tools.

When the latest rendered views are correct, stop calling tools and return only the structured `DrawingSubmission`.
