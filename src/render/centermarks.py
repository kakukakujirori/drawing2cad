"""Center marks for circular features (holes).

Rule (reverse-engineered from SW_CENTERMARKSYMBOL blocks + circle counts):
  - Every full circle in a view gets a cross center mark, EXCEPT concentric
    circles (same centre) collapse to a single mark using the largest radius
    (this reproduces the measured 298 circles -> 207 marks ratio: counterbores
    /countersinks are concentric).
  - Geometry per mark (matching the block): a central cross of half-length
    CROSS (2.5 mm), a gap, then an outer arm reaching radius + EXT (2.5 mm).
  - Pattern-centroid marks (SolidWorks adds one long mark spanning a hole
    array) are NOT reproduced -- documented residual gap.

Marks live on DXF layer "10" as INSERT blocks; in SVG they are drawn as four
arms with the center-line dash pattern.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.render.config import CENTERMARK_CROSS_MM, CENTERMARK_EXT_MM, CENTERMARK_GAP_MM
from src.render.layout import PlacedView


@dataclass
class CenterMark:
    cx: float
    cy: float
    r: float  # circle radius (sheet-mm)
    L: float  # arm half-length = r + EXT


def _dedupe_circles(circles, tol=0.15):
    """Collapse concentric circles; return [(cx, cy, max_r)]."""
    groups: list[list] = []
    for c in circles:
        cx, cy = c.center
        placed = False
        for g in groups:
            if abs(g[0] - cx) <= tol and abs(g[1] - cy) <= tol:
                g[2] = max(g[2], c.radius)
                placed = True
                break
        if not placed:
            groups.append([cx, cy, c.radius])
    return [(g[0], g[1], g[2]) for g in groups]


def marks_for_view(view: PlacedView) -> list[CenterMark]:
    circles = list(view.visible.circles)
    marks = []
    for cx, cy, r in _dedupe_circles(circles):
        marks.append(CenterMark(cx, cy, r, r + CENTERMARK_EXT_MM))
    return marks


def all_marks(views) -> list[CenterMark]:
    out = []
    for v in views:
        out.extend(marks_for_view(v))
    return out


def mark_block_lines(mark: CenterMark):
    """8 line segments in block-local coords (centre at origin), like SW.

    4 inner cross arms (0 -> CROSS) + 4 outer arms (CROSS+GAP -> L).
    """
    cross = CENTERMARK_CROSS_MM
    gap_end = CENTERMARK_CROSS_MM + CENTERMARK_GAP_MM
    L = mark.L
    lines = [
        ((0, 0), (0, cross)),
        ((0, 0), (-cross, 0)),
        ((0, 0), (0, -cross)),
        ((0, 0), (cross, 0)),
    ]
    if L > gap_end:
        lines += [
            ((0, gap_end), (0, L)),
            ((-gap_end, 0), (-L, 0)),
            ((0, -gap_end), (0, -L)),
            ((gap_end, 0), (L, 0)),
        ]
    return lines


def mark_svg_arms(mark: CenterMark):
    """4 full arms (centre -> ±L) for SVG; the dash pattern renders the gaps."""
    L = mark.L
    cx, cy = mark.cx, mark.cy
    return [
        ((cx, cy - L), (cx, cy + L)),  # vertical
        ((cx - L, cy), (cx + L, cy)),  # horizontal
    ]
