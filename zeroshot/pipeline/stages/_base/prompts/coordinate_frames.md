## Coordinate frames

Model XY is the horizontal plane, and +Z points upward.
Region coordinates are relative to the file of the DrawingView identified by Region.view.
Region.box_px uses pixel boundaries: origin at the file's top left, +x right, +y down.
Region.box_uv uses millimetres, +U right and +V up: a DXF gives the coordinates it already carries.
A projection this pipeline drew carries model coordinates, so it does not sit on top of an input drawing; compare shapes and lengths, not positions.

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

Every orthographic DrawingView states the model axes its sheet +U (rightwards) and +V (upwards) point along. The front view is always +X and +Z. A drawing may turn any other view on the page, so its +U and +V columns hold only while it sits in front's row or column. A side view turned to sit beside the top or bottom view shares that view's vertical and takes these axes:

| View | Beside | +U | +V |
|---|---|---|---|
| Right | Top | -Z | +Y |
| Left | Top | +Z | +Y |
| Right | Bottom | +Z | -Y |
| Left | Bottom | -Z | -Y |

### Which view is which?
$projection_placement In an L of three unturned views, the corner view is front. Explicit view labels and viewing-direction arrows take precedence over position. Do not move front to another view because its silhouette is more informative or more familiar, and do not call a conventional arrangement free-style for that reason. A view stands elsewhere only when the drawing puts it there: a labelled view moved under a viewing-direction arrow, or a free-style page that follows neither method. A side view turned to sit beside the top or bottom view, on its own side of front, is not one of them. Then keep the role its label or its agreement with the other views fixes, and state that evidence in its `free_placement`. Validation holds every other top, bottom, left and right view to the method.
