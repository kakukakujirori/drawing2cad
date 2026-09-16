## Coordinate frames

Model XY is the horizontal plane, and +Z points upward.
Region coordinates are relative to the file of the DrawingView identified by Region.view.
Region.box_px uses pixel boundaries: origin at the file's top left, +x right, +y down.
Region.box_uv uses millimetres: origin at the file's lower left, +U right, +V up. For DXF the origin is the geometry bounding box's lower left.

Orthographic view directions:

| View | +U | +V | Toward the viewer |
|---|---|---|---|
| Front | +X | +Z | -Y |
| Back | -X | +Z | +Y |
| Top | +X | +Y | +Z |
| Bottom | +X | -Y | -Z |
| Right | +Y | +Z | +X |
| Left | -Y | +Z | -X |

"Toward the viewer" points from the part to the person looking at that view. The frame is right-handed, so in the Front view +X is right, +Z is up and +Y points away from the viewer, into the page: the Front view shows the part's smallest-Y faces, and the Back view its largest-Y faces. Check your datum against this table rather than a habitual CAD frame.
These mappings specify directions, not a shared origin.
A DrawingView's local UV coordinates are not absolute model coordinates.

### Which is the front view?
For a conventional third-angle L arrangement, the lower-left view is front, the view directly above it is top, and the view directly to its right is right. Explicit view labels or projection symbols take precedence. Do not move front to another view because it has the most informative silhouette or a familiar shape, and do not call a conventional arrangement free-style for that reason. Treat the arrangement as free-style only when labels, projection symbols and the agreement between views do not fix the roles; then choose roles whose projections agree with each other.
