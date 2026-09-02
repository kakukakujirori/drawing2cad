"""DXF in, the same DXF plus dimensions out, and a PNG of it.

The source document is never rebuilt. Splines, ellipses, polylines and block
references survive as themselves; the only entities added are DIMENSION, on
their own layer. Line weights and dash lengths are applied while rendering, so
they vary between drawings without editing the source entities.
"""

import math
import os
import random
from dataclasses import dataclass

import ezdxf
from ezdxf.addons.drawing import Frontend, RenderContext, config, layout, pymupdf

from dim_drawer.dimensions import (
    DIM_LAYER,
    DimensionWriter,
    Occupancy,
    create_text_styles,
    sample_style,
)
from dim_drawer.extract import DASH_LINETYPES, extract_doc
from dim_drawer.placement import dimension_view
from dim_drawer.views import split_views

# ISO 128 line groups in 1/100 mm, thick to thin at 2:1.
LINE_GROUPS = ((25, 13), (35, 18), (50, 25), (70, 35))


@dataclass(frozen=True)
class LineStyle:
    """How the drawing is inked. Applied at render time, not stored in the DXF."""

    thick: int
    thin: int
    dash_scale: float

    @classmethod
    def sample(cls, rng):
        thick, thin = rng.choice(LINE_GROUPS)
        return cls(thick, thin, rng.uniform(0.7, 1.6))


def _ink_override(style):
    """Visible outlines thick, hidden lines thin, dimensions left as drawn."""

    def override(entity, properties):
        if properties.layer == DIM_LAYER:
            return
        if properties.linetype_name.upper() in DASH_LINETYPES:
            properties.lineweight = style.thin / 100.0
            if properties.linetype_pattern:
                properties.linetype_pattern = tuple(
                    p * style.dash_scale for p in properties.linetype_pattern
                )
        else:
            properties.lineweight = style.thick / 100.0

    return override


def render_png(
    dxf_path, png_path, style=None, dpi=200, page_mm=(297, 210), margin_mm=8
):
    doc = ezdxf.readfile(dxf_path)
    backend = pymupdf.PyMuPdfBackend()
    # Only ABSOLUTE honours per-entity lineweight; both RELATIVE policies draw
    # every stroke at one width.
    cfg = config.Configuration(
        background_policy=config.BackgroundPolicy.WHITE,
        lineweight_policy=config.LineweightPolicy.ABSOLUTE,
    )
    frontend = Frontend(RenderContext(doc), backend, config=cfg)
    if style is not None:
        frontend.push_property_override_function(_ink_override(style))
    frontend.draw_layout(doc.modelspace())

    page = layout.Page(
        page_mm[0], page_mm[1], layout.Units.mm, margins=layout.Margins.all(margin_mm)
    )
    with open(png_path, "wb") as fp:
        fp.write(backend.get_pixmap_bytes(page, fmt="png", dpi=dpi))


def annotate(src_dxf, out_dir, seed=0, text_height=None, dpi=200):
    """Dimension one drawing in place. Returns (dxf, png, dims, views, style)."""
    rng = random.Random(seed)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(src_dxf))[0]

    doc = ezdxf.readfile(src_dxf)
    msp = doc.modelspace()
    views = split_views(extract_doc(doc))
    if not views:
        raise ValueError(f"no geometry in {src_dxf}")

    create_text_styles(doc)
    if DIM_LAYER not in doc.layers:
        doc.layers.add(DIM_LAYER, color=7)

    xs = [v["bbox"][0] for v in views] + [v["bbox"][2] for v in views]
    ys = [v["bbox"][1] for v in views] + [v["bbox"][3] for v in views]
    diag = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    # Text scales with the sheet: ~3.5mm on an A4-sized drawing.
    txt = text_height or max(diag * 0.0165, 1.0)
    sheet = (
        min(xs) - diag * 0.18,
        min(ys) - diag * 0.18,
        max(xs) + diag * 0.18,
        max(ys) + diag * 0.18,
    )

    style = LineStyle.sample(rng)
    writer = DimensionWriter(
        doc, msp, sample_style(rng), txt_height=txt, lineweight=style.thin
    )

    # Reserving the views themselves keeps labels off the part.
    occupancy = Occupancy(pad=txt * 0.6)
    for view in views:
        occupancy.add(view["bbox"])

    placed = 0
    for i, view in enumerate(views):
        others = [v["bbox"] for j, v in enumerate(views) if j != i]
        placed += dimension_view(writer, view, others, sheet, occupancy)

    dxf_path = os.path.join(out_dir, f"{stem}_dim.dxf")
    png_path = os.path.join(out_dir, f"{stem}_dim.png")
    doc.saveas(dxf_path)
    render_png(dxf_path, png_path, style=style, dpi=dpi)
    return dxf_path, png_path, placed, len(views), style
