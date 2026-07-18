"""PDF writer: rasterless conversion of the SAME SVG via cairosvg."""

from __future__ import annotations

import cairosvg


def write_pdf(path, svg_text: str):
    cairosvg.svg2pdf(bytestring=svg_text.encode("utf-8"), write_to=str(path))
