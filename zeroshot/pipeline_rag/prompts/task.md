## Annotating the drawing

Read the drawing into `$interpretation_output_path`. It is seeded with the input files already registered as views and `datum` left as `"???"`. Nothing is built from it. Once it validates and describes every feature, answer with one `AnnotationReport`. An answer is refused until the file validates.

### Interpretation contract

- Keep the original input view's name, file, role and self-referencing full-file Region.
- A full_page input is an unsplit page in most cases. Register each orthographic view you identify on it as a separate DrawingView; at least one is required. Different roles require separate files: save each crop outside the read-only input directory. Its Region locates the crop in the parent file; setting Region alone does not crop the file. For a single-view drawing, register that view on the same file with the full-file Region. Feature evidence may cite either file through Region.view.
- Raster crops only: exactly match the parent image cropped at Region.box_px, pixel for pixel at 1:1 scale, without resizing, rotation or preprocessing. Save observation-only crops and zooms to separate files, never over a registered view.
- Measure box_px in the referenced file's native pixels, not a resized display. Leave image_size, scale and box_uv null; validation derives them.
- Set datum to one sentence naming the model origin. The coordinate-frame table fixes the axes.
- Leave every view's `dimensions` empty, and every feature's `parameters` and `dimension_refs` empty.

DrawingInterpretation JSON schema:
```json
$interpretation_schema
```

### Features as references

Each feature is stored for later lookup. Crops of its evidence boxes are the key. Its description is what a reader of another drawing gets back, for a part of other sizes that may sit in another pose.

- Make one feature per local shape, such as the base body, a hole, a boss, a slot, a rib or a step. For a pattern of identical features, describe and box one instance.
- Cite one Region per orthographic view that shows the feature, in solid or hidden lines. Box tightly around that feature's own lines in that view. Leave out dimension text and other features' lines where the box allows.
- In the description, state the 3D shape, whether it is material or a void, and where it ends: through, blind, or on which kind of face. Then state how the views draw it: which outlines are solid, which are hidden, and what tells it apart from a shape that looks alike in one view, such as a hole from a boss or a slot from a rib. Name a view by its direction to the feature, such as along its axis or across it.
- Write no numbers, model axes (X, Y, Z), view names (front view, top view, view_...) or member names (sem_..., dim_...) in a description; validation refuses them. They do not carry over to another part.$pictorial
- Keep one mutually consistent reading in features. Report the readings you rejected, and the views that decided between them, in your answer's concerns.

### Work cycle

Use `load_image` to inspect drawings and `run_shell` to create crops and edit files. Each raster input `xxx.png` has `xxx_skeleton.npz`: its strokes as dark as the part's lines, thinned to one-pixel traces and stored as points sorted by `y`, with arrays `x`, `y` (input pixels) and `angle` (0 to 180 degrees counterclockwise from the right). Use it to locate lines. `fit_edge` tells whether the strokes in a box are dashed (hidden) or solid.

- Base each view role on the drawing's projection method, explicit labels and agreement between views, as the coordinate-frame section describes.
- Inspect hidden lines and matching projections when a silhouette permits several shapes. Reconcile projections of the same physical feature before creating separate features; a circle in one view and a rounded outline in another may describe one protrusion.
- Save the views and current features to `$interpretation_output_path` early, provisional where you are unsure. After a tool turn changes the file, validation checks it against this contract, including that each crop is the exact 1:1 crop its Region declares and that the views' arrangement fits the projection method. A valid file does not confirm the 3D reading.
- Check every feature against all views before you answer.
