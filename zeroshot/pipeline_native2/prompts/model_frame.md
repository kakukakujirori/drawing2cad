### Model frame

Model XY is the horizontal plane, and +Z points upward. The pipeline draws each orthographic projection of the solid in these directions:

| View | +U | +V | Toward the viewer |
|---|---|---|---|
| Front | +X | +Z | -Y |
| Back | -X | +Z | +Y |
| Top | +X | +Y | +Z |
| Bottom | +X | -Y | -Z |
| Right | +Y | +Z | +X |
| Left | -Y | +Z | -X |

"Toward the viewer" points from the part to the person looking at that view, so the Front view shows the part's smallest-Y faces. Follow this table rather than a habitual CAD frame. A projection this pipeline drew carries model coordinates, so it does not sit on top of an input drawing; compare shapes and lengths, not positions.
