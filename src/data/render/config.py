"""Shared configuration and data contracts for the ECCV-style renderer.

The renderer reproduces the drawing domain of data/eccv2026-cad-challenge-data:
SolidWorks-generated 3-view technical drawings (A4 landscape, no frame / title
block / dimensions) plus three perspective 3D renders per part.

Modules:
  - techdraw.py  (HLR projection -> typed 2D primitives -> svg/dxf/pdf writers)
  - render3d.py  (hlg / transparent_shaded_edges / hlg_translucent_faces PNGs)
  - render_dataset.py (CLI batch driver with per-part process isolation)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Target-domain constants (measured on data/eccv2026-cad-challenge-data)
# ---------------------------------------------------------------------------

# Sheet: A4 landscape in points (72 dpi), identical to the GT SVG/PDF header.
SHEET_W_PT = 841.89
SHEET_H_PT = 595.276
# Same sheet in mm (DXF model space is laid out in sheet-mm).
SHEET_W_MM = 297.0
SHEET_H_MM = 210.0

# GT SVG line styles (stroke widths in pt, dasharrays in pt).
SVG_STYLE_VISIBLE = {"stroke-width": 0.70866, "dasharray": None}
SVG_STYLE_HIDDEN = {"stroke-width": 0.51024, "dasharray": (3.60001, 1.8)}
SVG_STYLE_CENTER = {
    "stroke-width": 0.51024,
    "dasharray": (86.40001, 3.60001, 3.60001, 3.60001),
}

# render_3d target raster size (GT PNGs are 1400x1000 RGB, white background).
RENDER3D_SIZE = (1400, 1000)

RENDER3D_STYLES = (
    "hlg_perspective",
    "transparent_shaded_edges_perspective",
    "hlg_translucent_faces_perspective",
)

TECHDRAW_FORMATS = ("svg", "dxf", "pdf")

# mm -> pt (SVG path coordinates are emitted in points; 72 pt / 25.4 mm).
MM_TO_PT = 72.0 / 25.4  # 2.834645...

# The exact affine the GT SVGs carry on every <path> (constant across the whole
# train split): near-identity scale with a Y-flip about the sheet.  Path coords
# are already in points, so this maps points -> device points (top-left origin).
SVG_TRANSFORM = (0.998785, 0.0, 0.0, -0.998785, 0.456416, 594.552879)

# --- Layout (measured on GT DXF, sheet-mm) --------------------------------
# The 3-view cluster is centred on the sheet (measured cluster centre
# ~= (145, 106) ~= sheet centre).  Top view shares the front view's x-centre
# (dx_align == 0.00 over 351 parts); right view shares its y-centre.
SHEET_CENTER_MM = (SHEET_W_MM / 2.0, SHEET_H_MM / 2.0)  # (148.5, 105.0)
# Inter-view gaps.  GT gaps are scale-coupled (median v~44, h~72 mm) and the
# absolute scale is unrecoverable (GT STEPs are normalised); these fixed gaps
# only affect global composition, never per-view geometry.
VIEW_GAP_V_MM = 25.0
VIEW_GAP_H_MM = 35.0
# Usable envelope the 3-view cluster must fit inside (drives scale selection).
# GT clusters reach ~240x201 mm; keep a small margin.
ENVELOPE_W_MM = 255.0
ENVELOPE_H_MM = 192.0
# Standard drawing scales SolidWorks chooses from (drawing-mm per model-unit).
SCALE_LADDER = (
    100.0,
    50.0,
    20.0,
    10.0,
    5.0,
    2.0,
    1.0,
    0.5,
    0.2,
    0.1,
    0.05,
    0.02,
    0.01,
    0.005,
    0.002,
    0.001,
)

# --- Center marks (measured on SW_CENTERMARKSYMBOL blocks) -----------------
CENTERMARK_CROSS_MM = 2.5  # central cross half-length
CENTERMARK_GAP_MM = 2.5  # gap from cross tip to the extension dash start
CENTERMARK_EXT_MM = 2.5  # arm extension beyond the circle edge
# SVG center-line dash pattern (pt) and its DXF-mm equivalent.
SVG_CENTER_DASH = (86.40001, 3.60001, 3.60001, 3.60001)


# ---------------------------------------------------------------------------
# Data contracts between modules
# ---------------------------------------------------------------------------


@dataclass
class TechdrawPaths:
    svg: Path
    dxf: Path
    pdf: Path


@dataclass
class Render3dPaths:
    hlg: Path
    shaded: Path
    hlg_translucent: Path


@dataclass
class PartResult:
    """Per-part render outcome (printed live; failures logged to render_errors.jsonl)."""

    name: str
    ok: bool
    error: str = ""
    techdraw_ok: bool = False
    render3d_ok: bool = False
    seconds: float = 0.0
    extra: dict = field(default_factory=dict)


def techdraw_paths(out_dir: Path, stem: str) -> TechdrawPaths:
    """OUTDIR/techdraw/{svg,dxf,pdf}/{stem}.{ext} (mirrors the ECCV tree)."""
    td = out_dir / "techdraw"
    return TechdrawPaths(
        svg=td / "svg" / f"{stem}.svg",
        dxf=td / "dxf" / f"{stem}.dxf",
        pdf=td / "pdf" / f"{stem}.pdf",
    )


def render3d_paths(out_dir: Path, stem: str) -> Render3dPaths:
    """OUTDIR/render_3d/<style>/{stem}.png (mirrors the ECCV tree)."""
    r3 = out_dir / "render_3d"
    return Render3dPaths(
        hlg=r3 / "hlg_perspective" / f"{stem}.png",
        shaded=r3 / "transparent_shaded_edges_perspective" / f"{stem}.png",
        hlg_translucent=r3 / "hlg_translucent_faces_perspective" / f"{stem}.png",
    )
