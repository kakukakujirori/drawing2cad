## Coordinate frames

Model XY is the horizontal plane, and +Z points upward.
Each drawing view has its own local UV coordinates:
+U points right on the sheet, and +V points up.

| View | +U | +V | Toward the viewer |
|---|---|---|---|
| Front | +X | +Z | -Y |
| Back | -X | +Z | +Y |
| Top | +X | +Y | +Z |
| Bottom | +X | -Y | -Z |
| Right | +Y | +Z | +X |
| Left | -Y | +Z | -X |

These mappings specify directions, not a shared origin.
A sheet's local UV coordinates are not absolute model coordinates.