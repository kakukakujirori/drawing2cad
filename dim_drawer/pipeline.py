"""DXF in, dimensioned DXF and PNG out."""

import math
import os
import random

import ezdxf
from ezdxf.addons.drawing import Frontend, RenderContext, config, layout, pymupdf

from dim_drawer.dimensions import (
    DimensionWriter,
    Occupancy,
    create_text_styles,
    sample_style,
)
from dim_drawer.extract import extract
from dim_drawer.placement import dimension_view
from dim_drawer.views import split_views

# ISO 128 line groups, thick to thin at 2:1.
LINE_GROUPS = ((25, 13), (35, 18), (50, 25), (70, 35))


def setup_layers(doc, rng):
    thick, thin = rng.choice(LINE_GROUPS)
    dash = rng.uniform(0.7, 1.6)
    doc.linetypes.add(
        "HIDDEN2", pattern=[p * dash for p in (3.0, 2.0, -1.0)], description="Hidden"
    )
    doc.layers.add("OUTLINE", color=7, lineweight=thick)
    doc.layers.add("HIDDEN", color=7, linetype="HIDDEN2", lineweight=thin)
    doc.layers.add("DIM", color=7, lineweight=thin)
    return thick, thin


def write_geometry(msp, data):
    """Re-emit the source geometry with visible and hidden edges separated."""
    for r in data["line"]:
        msp.add_line(
            (r["start_x"], r["start_y"]),
            (r["end_x"], r["end_y"]),
            dxfattribs={"layer": "OUTLINE"},
        )
    for r in data["dash_line"]:
        msp.add_line(
            (r["start_x"], r["start_y"]),
            (r["end_x"], r["end_y"]),
            dxfattribs={"layer": "HIDDEN"},
        )
    for r in data["circle"]:
        msp.add_circle(
            (r["center_x"], r["center_y"]),
            r["radius"],
            dxfattribs={"layer": "HIDDEN" if r.get("dashed") else "OUTLINE"},
        )
    for r in data["arc"]:
        msp.add_arc(
            (r["center_x"], r["center_y"]),
            r["radius"],
            r["start_angle"],
            r["end_angle"],
            dxfattribs={"layer": "HIDDEN" if r.get("dashed") else "OUTLINE"},
        )


def render_png(dxf_path, png_path, dpi=200, page_mm=(297, 210), margin_mm=8):
    doc = ezdxf.readfile(dxf_path)
    backend = pymupdf.PyMuPdfBackend()
    # Only ABSOLUTE honours per-layer lineweight; both RELATIVE policies draw
    # every stroke at one width.
    cfg = config.Configuration(
        background_policy=config.BackgroundPolicy.WHITE,
        lineweight_policy=config.LineweightPolicy.ABSOLUTE,
    )
    Frontend(RenderContext(doc), backend, config=cfg).draw_layout(doc.modelspace())
    page = layout.Page(
        page_mm[0], page_mm[1], layout.Units.mm, margins=layout.Margins.all(margin_mm)
    )
    with open(png_path, "wb") as fp:
        fp.write(backend.get_pixmap_bytes(page, fmt="png", dpi=dpi))


def annotate(src_dxf, out_dir, seed=0, text_height=None, dpi=200):
    """Dimension one drawing. Returns (dxf_path, png_path, dimension count)."""
    rng = random.Random(seed)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(src_dxf))[0]

    data = extract(src_dxf)
    views = split_views(data)
    if not views:
        raise ValueError(f"no geometry in {src_dxf}")

    doc = ezdxf.new("R2010", setup=True)
    msp = doc.modelspace()
    thick, thin = setup_layers(doc, rng)
    write_geometry(msp, data)
    create_text_styles(doc)

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

    writer = DimensionWriter(
        doc, msp, sample_style(rng), txt_height=txt, lineweight=thin
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
    render_png(dxf_path, png_path, dpi=dpi)
    return dxf_path, png_path, placed, len(views), thick, thin
