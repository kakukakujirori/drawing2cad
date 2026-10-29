## Drawing skeleton

Each raster input `xxx.png` (e.g., `view_drawing.png`) has a corresponding `xxx_skeleton.npz` (e.g., `view_drawing_skeleton.npz`), where the drawing's strokes are thinned to one-pixel-wide traces and stored as points sorted by `y`. Every stroke as dark as the part's lines is kept, including dimension lines and text drawn that dark. Strokes in a clearly lighter tone, such as the gray dimension lines of some drawings, are left out. The `.npz` file holds these arrays, with one entry per point:

- `x`, `y`: position in the input's pixels, sub-pixel.
- `angle`: line direction, 0 to 180 degrees counterclockwise from the right as seen on the page; a horizontal line reads near 0 or 180.

Use it to locate lines and curves. To further describe them as parametric curves, pass a bounding box around each to `fit_edge` if the tool is available.
