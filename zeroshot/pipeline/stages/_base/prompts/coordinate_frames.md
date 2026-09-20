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

"Toward the viewer" points from the part to the person looking at that view, so the Front view shows the part's smallest-Y faces. Follow this table rather than a habitual CAD frame. It specifies directions, not a shared origin, and a DrawingView's local UV coordinates are not absolute model coordinates.

Every orthographic DrawingView states the model axes its sheet +U (rightwards) and +V (upwards) point along. The front view is always +X and +Z. A drawing may turn any other view on the page, so its +U and +V columns hold only while it sits in front's row or column: rather than copying them, find a printed length the view shares with a neighbour and see which way that length runs in each. A side view placed beside the top view rather than beside front, for instance, shares the top view's vertical: its +V runs along ±Y, not +Z.

### Which is the front view?
In a third-angle L arrangement the lower-left view is front, the view above it is top, and the view to its right is right. Explicit labels and projection symbols take precedence. Do not move front to another view because its silhouette is more informative or more familiar, and do not call a conventional arrangement free-style for that reason. Treat the arrangement as free-style only when labels, symbols and the agreement between views do not fix the roles; then choose roles whose projections agree with each other.
