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

These mappings specify directions, not a shared origin.
A DrawingView's local UV coordinates are not absolute model coordinates.
