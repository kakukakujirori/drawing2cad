# dim_drawer

Adds dimension annotations to un-dimensioned 2D drawings and renders them to PNG.

`data/test_vlm/techdraw/dxf` carries geometry only. Real drawings reach us as raster
images with dimensions on them, so this turns the DXF source into a plausible
annotated drawing that can be fed to the pipeline as an image.

## Usage

```bash
conda activate drawing2cad
python -m dim_drawer data/test_vlm/techdraw/dxf/*.dxf --out data/test_vlm/techdraw_dimensioned
```

Writes `<stem>_dim.dxf` and `<stem>_dim.png` per input. The DXF is the source
document with DIMENSION entities added on a `DIM_DRAWER` layer; splines,
ellipses, polylines and block references stay as they are and no existing entity
is edited. `--seed` selects the annotation style (one per drawing,
reproducible), `--dpi` and `--text-height` override render resolution and text
size.

```python
from dim_drawer import annotate

dxf, png, n_dims, n_views, line_style = annotate("part.dxf", "out/", seed=0)
```

To match the normalized STEP targets to the DXF's millimetre dimensions:

```bash
python -m dim_drawer.scale_step \
  --step-dir data/test_vlm/target_step \
  --dxf-dir data/test_vlm/techdraw/dxf \
  --out data/test_vlm/target_step_ori
```

Each part gets one uniform scale fitted to all six view extents. The exported
STEP is read back and checked against every view within `--tolerance-mm`
(default 0.01 mm) before it replaces the output; input STEPs are preserved.
This assumes the third-angle front XY / top XZ / right ZY arrangement. The
`target_step_ori` outputs are the millimetre-scale targets for this dataset.

Run the regression check with `python -m dim_drawer.test_geometry`.

## How it works

1. `extract.py` — flattens the drawing to line/circle/arc records for analysis
   only, expanding blocks and polylines and splitting off dashed linetypes. The
   source document is not rebuilt from these. Center lines and center-mark
   blocks remain in the drawing but are excluded from measurements.
2. `views.py` — separates the projections by recursive XY-cut on whitespace bands.
3. `placement.py` — per view, picks overall extents, significant axis-aligned
   edges, hole diameters and arc radii, and places them on the side facing away
   from the sheet centroid. Edge grouping preserves the actual anchor coordinates.
4. `dimensions.py` — draws the DIMENSION entities into the source document. Each
   one is rendered, its real text box read back from the generated block, and
   undone if it collides with something already placed.

Style is randomised per drawing: arrow blocks, text style, decimal separator,
optional `mm` suffix and background fill. Dimensions use two decimal places
(trailing zeros omitted) to keep the displayed values within 0.005 drawing units.
Line weight (ISO 128 groups
0.25/0.35/0.5/0.7 mm at 2:1 thick-to-thin) and dash length are applied through a
render-time property override, so they vary per drawing without the DXF
recording them.

## Limits

- Dimensions measure the DXF geometry with `DIMLFAC=1`, rounded to two decimal
  places. Nothing simulates manufacturing tolerances.
- Arc radius candidates are selected without a minimum sweep angle.
- Radius dimensions use the ARC radii stored in the DXF. In drawings such as
  `005154` and `001546`, arcs approximate projected ellipses, so these values
  differ from the radii of the corresponding 3D circles.
- Only text boxes and view bounding boxes are collision-checked, so dimension
  and extension lines can still cross a label.
- No title block, section views, GD&T or surface finish symbols.

## Credit

The dimension style vocabulary and the font table come from
[ParaCAD](https://www.modelscope.cn/datasets/yuwenbonnie/ParaCAD_dataset)
(`dxfWriter_cyw_white.py`, Apache-2.0). Its dimension *selection* is not reused:
it pairs random endpoints, which suits a single sketch but spans two projections
on a multi-view drawing.
