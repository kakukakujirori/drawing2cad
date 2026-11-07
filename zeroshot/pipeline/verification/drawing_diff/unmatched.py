"""Find and display differences in the input drawing's pixel coordinates.

Shared pixel operations support silhouette and line comparison. Three rules find
line differences. A far line leaves the other image's lines by a wide margin
over some length; it applies both ways, and its size keeps the input's annotation
out. A near line is a long input line that leaves the output by a smaller margin;
it ranks below every other group. An unsupported stretch runs along one of the
output's DXF curves, which carry no annotation, so it can be short and near.
find_unmatched joins far lines and stretches before selecting display groups;
descriptions and drawing use that same selection.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from operator import attrgetter
from typing import Any, Literal

import cv2
import numpy as np
from ezdxf.layouts import Modelspace
from ezdxf.path import make_path
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from skimage.morphology import thin

from .image_ops import disk, distance_map, drawing_area, filled_area

# Tuned on two drawings and their ground truth: outputs/a2_gate_20260927.
FAR_PEAK_PX = 12.0
JOIN_PX = 4
PART_MARGIN_PX = 20
FAR_MIN_LENGTH_PX = 50
# A line that leaves its place gradually lies beyond FAR_PEAK_PX only in short
# stretches; its group grows from them along the line while it stays this far.
# Chosen where ground-truth builds gain no missing groups, on the ECCV views of
# outputs/20261101_minimax3.1_flash; 4 px admits them.
FAR_GROW_PX = 5.0
# A long input line that stays within FAR_PEAK_PX of the output but reaches this
# far takes only the display slots the other groups leave free.
NEAR_PEAK_PX = 6.0

# Tuned the same way: outputs/silhouette_20260927, with image_ops.CLOSE_PX.
# Narrower slivers come from misregistered edges and from annotation.
MATERIAL_THIN_PX = 30
MIN_AREA_PX = 1000
# A line group this close to a material group's outline is that outline;
# the width covers the corners that the material's opening rounds off.
OUTLINE_PX = 8

# Unsupported stretches: a stretch of an output curve beyond STRETCH_GROW_PX of every
# input line, the far end of matched-line noise, that somewhere reaches
# STRETCH_PEAK_PX. Between the input's dashes a hidden curve stays nearer than half
# the gap, which STRETCH_HIDDEN_PEAK_PX exceeds. Chosen where ground-truth builds
# gain few groups, on outputs/20261101_minimax3.1_flash (ECCV) and
# outputs/20261101_minimax3.1_flash_mixed_with_20261027_itercad_ortho2cad (Ortho2CAD).
STRETCH_GROW_PX = 3.0
STRETCH_PEAK_PX = 6.0
STRETCH_HIDDEN_PEAK_PX = 7.0

# The most difference groups displayed for one view.
MAX_GROUPS = 6
# Wider than the displayed strokes, so the band shows on both sides of them.
BAND_PX = 5
# Centre lines this near are one line: registering a line both images draw
# leaves it up to this far off, beyond the pixel that rasterizing moves it.
SHARED_PX = 3.0
# Gray is ink that is never a difference: annotation, optional tangents and ink
# the comparison does not reach. Lines both images draw are blue; a line only
# one image draws takes its color, input green and output magenta, toned so
# that neither outweighs the other.
NEUTRAL_GRAY = (185, 185, 185)
MATCHED_BLUE = (0, 0, 255)
INPUT_GREEN = (0, 160, 0)
OUTPUT_MAGENTA = (200, 60, 200)
# Material is hatched, as a drawing marks cut material, so it reads as an area.
HATCH_PX = 12
# One color per listed group, in rank order. Neighbouring hues are kept apart,
# each differs in hue or lightness from the blue, magenta and green lines
# drawn on top, and each is dark enough for its number on white.
COLORS = {
    "dark orange": (205, 100, 0),
    "slate": (110, 125, 165),
    "maroon": (140, 40, 50),
    "sky": (90, 170, 240),
    "ochre": (190, 145, 30),
    "teal": (0, 160, 160),
}


################################################################

# Shared pixel operations:
# Coordinates, cluster measurements and connectivity.


@dataclass(frozen=True)
class Unmatched:
    """One difference group, measured on the input drawing's pixel grid."""

    direction: Literal["missing", "extra", "line_style"]
    xy: np.ndarray  # (x, y): skeleton pixels of lines, or every pixel of material
    material: bool = False
    input_style: str | None = None
    output_style: str | None = None
    coverage: float | None = None

    @property
    def size_px(self) -> int:
        return len(self.xy)

    @property
    def box_px(self) -> tuple[int, int, int, int]:
        (x0, y0), (x1, y1) = self.xy.min(axis=0), self.xy.max(axis=0) + 1
        return int(x0), int(y0), int(x1), int(y1)


def warp_output_to_drawing(
    output_ink: np.ndarray, matrix: np.ndarray, shape: tuple[int, int]
) -> np.ndarray:
    """The output's lines on the drawing's pixel grid, unbroken when shrunk."""
    height, width = shape
    # Shrinking by n skips thin lines unless they are about n pixels wide first.
    shrink = np.sqrt(abs(np.linalg.det(matrix[:2, :2])))
    size = max(int(np.ceil(shrink)), 1)
    # Let strokes at the canvas edge expand fully before sampling them.
    padded = np.pad(output_ink.astype(np.uint8), size)
    thick = cv2.dilate(padded, np.ones((size, size), np.uint8))
    padding_transform = np.eye(3)
    padding_transform[:2, 2] = size
    warped = cv2.warpPerspective(
        thick * 255,
        np.linalg.inv(padding_transform @ matrix),
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderValue=0,
    )
    return warped > 0


def _groups(
    mask: np.ndarray,
    joined: np.ndarray,
    direction: Literal["missing", "extra"],
    material: bool = False,
) -> list[Unmatched]:
    """Use joined for connectivity, but count/store only the original mask pixels."""
    _, labels = cv2.connectedComponents(joined.astype(np.uint8))
    ys, xs = np.nonzero(mask)
    owners = labels[ys, xs]
    return [
        Unmatched(
            direction, np.stack([xs[owners == n], ys[owners == n]], axis=1), material
        )
        for n in np.unique(owners)
    ]


def _line_groups(
    mask: np.ndarray,
    direction: Literal["missing", "extra"],
    min_length: int = FAR_MIN_LENGTH_PX,
) -> list[Unmatched]:
    groups = _groups(mask, cv2.dilate(mask.astype(np.uint8), disk(JOIN_PX)), direction)
    return [group for group in groups if group.size_px >= min_length]


def _long_line_groups(
    lines: np.ndarray, distances: np.ndarray, direction: Literal["missing", "extra"]
) -> list[Unmatched]:
    """Line groups beyond FAR_GROW_PX at least FAR_MIN_LENGTH_PX long.

    The whole group counts towards its length, not only its farthest stretches.
    """
    groups = _line_groups(lines & (distances > FAR_GROW_PX), direction, min_length=0)
    return [group for group in groups if _length_px(group) >= FAR_MIN_LENGTH_PX]


def _peak_px(group: Unmatched, distances: np.ndarray) -> float:
    return float(distances[group.xy[:, 1], group.xy[:, 0]].max())


def _far_line_groups(
    lines: np.ndarray, distances: np.ndarray, direction: Literal["missing", "extra"]
) -> list[Unmatched]:
    """Long line groups that reach beyond FAR_PEAK_PX somewhere."""
    return [
        group
        for group in _long_line_groups(lines, distances, direction)
        if _peak_px(group, distances) > FAR_PEAK_PX
    ]


def _near_line_groups(lines: np.ndarray, distances: np.ndarray) -> list[Unmatched]:
    """Long missing line groups within FAR_PEAK_PX that reach NEAR_PEAK_PX."""
    return [
        group
        for group in _long_line_groups(lines, distances, "missing")
        if NEAR_PEAK_PX <= _peak_px(group, distances) <= FAR_PEAK_PX
    ]


def _length_px(group: Unmatched) -> float:
    """The length of a group's lines, with gaps up to 2 * JOIN_PX bridged.

    A band of radius JOIN_PX around a line of length L covers about
    L * (2 * JOIN_PX + 1) + pi * JOIN_PX**2 pixels in any direction, while
    a count of skeleton pixels undercounts diagonals and dashed lines.
    """
    x0, y0, x1, y1 = group.box_px
    pad = JOIN_PX
    mask = np.zeros((y1 - y0 + 2 * pad, x1 - x0 + 2 * pad), np.uint8)
    mask[group.xy[:, 1] - y0 + pad, group.xy[:, 0] - x0 + pad] = 1
    band = int(cv2.dilate(mask, disk(JOIN_PX)).sum())
    return (band - np.pi * JOIN_PX**2) / (2 * JOIN_PX + 1)


################################################################

# Material differences:
# Filled silhouettes, area statistics and outline bands.


def _material_groups(
    mask: np.ndarray, direction: Literal["missing", "extra"]
) -> list[Unmatched]:
    opened = cv2.morphologyEx(
        mask.astype(np.uint8), cv2.MORPH_OPEN, disk(MATERIAL_THIN_PX // 2)
    ).astype(bool)
    groups = _groups(opened, opened, direction, material=True)
    return [group for group in groups if group.size_px >= MIN_AREA_PX]


def measure_material_error(
    drawing_ink: np.ndarray, output: np.ndarray
) -> dict[str, Any]:
    """Stable filled-outline statistics, independent of display groups and quotas.

    Ambiguous fills at narrow openings are excluded in both scoring and display.
    Display's thin-sliver and cluster-size filters do not affect these counts.
    """
    input_area, output_area = _material_areas(drawing_ink, output)
    if not input_area.any():
        return {
            "status": "unavailable",
            "reason": "the input outline cannot be filled (thin walls or open lines)",
            "ratio": None,
        }
    missing = int((input_area & ~output_area).sum())
    extra = int((output_area & ~input_area).sum())
    union = int((input_area | output_area).sum())
    contours, _ = cv2.findContours(
        input_area.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    return {
        "status": "ok",
        "missing_px2": missing,
        "extra_px2": extra,
        "union_px2": union,
        "ratio": (missing + extra) / union,
        "input_perimeter_px": float(sum(cv2.arcLength(c, True) for c in contours)),
    }


def _material_areas(
    drawing_ink: np.ndarray, output: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Filled outlines, excluding bays whose fill depends on a narrow opening.

    A small split can open an entire ring or frame to flood fill. Its closure
    in the other raster is not evidence of a large material difference. The
    lines at the split are still compared by the line detectors.
    """
    input_area, output_area = drawing_area(drawing_ink), filled_area(output)
    uncertain = np.zeros(drawing_ink.shape, bool)
    for ink in (drawing_ink, output):
        closed = cv2.morphologyEx(
            ink.astype(np.uint8), cv2.MORPH_CLOSE, disk(MATERIAL_THIN_PX // 2)
        )
        uncertain |= binary_fill_holes(closed) & ~binary_fill_holes(ink)
    uncertain &= input_area ^ output_area
    return input_area & ~uncertain, output_area & ~uncertain


def _outline(groups: list[Unmatched], shape: tuple[int, ...]) -> np.ndarray:
    """A band around the edges of the groups' areas."""
    area = np.zeros(shape, np.uint8)
    for group in groups:
        area[group.xy[:, 1], group.xy[:, 0]] = 1
    band = cv2.dilate(area, disk(OUTLINE_PX)) - cv2.erode(area, disk(OUTLINE_PX))
    return band.astype(bool)


def _vicinity(output: np.ndarray) -> np.ndarray:
    """The output's bounding box widened by PART_MARGIN_PX; empty without output."""
    vicinity = np.zeros(output.shape, bool)
    if output.any():
        ys, xs = np.nonzero(output)
        vicinity[
            max(ys.min() - PART_MARGIN_PX, 0) : ys.max() + PART_MARGIN_PX + 1,
            max(xs.min() - PART_MARGIN_PX, 0) : xs.max() + PART_MARGIN_PX + 1,
        ] = True
    return vicinity


################################################################

# Unsupported stretches:
# Output DXF curves, the stretches no input line supports, and joining far lines.


def drawing_curves(
    modelspace: Modelspace, model_to_drawing: np.ndarray, tolerance_px: float
):
    """Yield flattened curves and styles; pixel boundaries map to OpenCV centres."""
    tolerance = tolerance_px / np.linalg.norm(model_to_drawing[:2, :2], axis=0).max()
    for entity in modelspace:
        if entity.dxftype() not in {"LINE", "ARC", "CIRCLE", "ELLIPSE", "SPLINE"}:
            continue
        kind = entity.dxf.get("linetype", "BYLAYER").upper()
        if kind == "BYLAYER":
            kind = modelspace.doc.layers.get(entity.dxf.layer).dxf.linetype.upper()
        if kind not in {"CONTINUOUS", "HIDDEN"}:
            continue
        points = np.array(
            [tuple(p)[:2] for p in make_path(entity).flattening(tolerance)]
        )
        if len(points) < 2:
            continue
        xy = (
            cv2.perspectiveTransform(points[None].astype(float), model_to_drawing)[0]
            - 0.5
        )
        xy = xy[np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-6]]
        if len(xy) >= 2:
            style = (
                "tangent"
                if entity.dxf.layer == "TANGENT"
                else ("hidden" if kind == "HIDDEN" else "visible")
            )
            yield xy, style


def output_curves(
    modelspace: Modelspace, model_to_drawing: np.ndarray
) -> list[tuple[np.ndarray, str]]:
    """The output's visible and hidden curves in drawing pixels, with their styles.

    Whole curves, unlike rendered strokes, carry no hidden-line dash phase.
    Optional tangent boundaries are left out.
    """
    return [
        (xy, style)
        for xy, style in drawing_curves(modelspace, model_to_drawing, tolerance_px=0.5)
        if style != "tangent"
    ]


def _unsupported_stretches(
    curves: Sequence[tuple[np.ndarray, str]], distances: np.ndarray
) -> list[Unmatched]:
    """Each stretch of a curve that leaves every input line; see STRETCH_PEAK_PX.

    Distance changes by at most 1 px per pixel along a curve, so a stretch
    between supported points spans at least twice the climb past STRETCH_GROW_PX;
    a stretch at the curve's end must be as long.
    """
    height, width = distances.shape
    groups = []
    for xy, style in curves:
        reach = STRETCH_HIDDEN_PEAK_PX if style == "hidden" else STRETCH_PEAK_PX
        lengths = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        steps = np.arange(0, lengths[-1] + 1e-9, 1.0)
        points = np.rint(
            np.column_stack([np.interp(steps, lengths, c) for c in xy.T])
        ).astype(int)
        inside = (points >= 0).all(axis=1) & (points < (width, height)).all(axis=1)
        # Beyond the drawing nothing is known, so nothing is unsupported there.
        far = np.zeros(len(points))
        far[inside] = distances[points[inside, 1], points[inside, 0]]
        edges = np.flatnonzero(np.diff(np.r_[0, far > STRETCH_GROW_PX, 0]))
        for start, stop in zip(edges[::2], edges[1::2], strict=True):
            long_enough = stop - start >= 2 * (reach - STRETCH_GROW_PX)
            if long_enough and far[start:stop].max() >= reach:
                groups.append(Unmatched("extra", np.unique(points[start:stop], axis=0)))
    return groups


def _join_stretches(
    far_lines: list[Unmatched], stretches: list[Unmatched], shape: tuple[int, ...]
) -> list[Unmatched]:
    """Join stretches to the far extra lines they touch."""
    combined = np.zeros(shape, bool)
    for group in [*(g for g in far_lines if g.direction == "extra"), *stretches]:
        combined[group.xy[:, 1], group.xy[:, 0]] = True
    merged = _groups(
        combined, cv2.dilate(combined.astype(np.uint8), disk(JOIN_PX)), "extra"
    )
    return [g for g in far_lines if g.direction == "missing"] + merged


################################################################

# Combine the difference candidates.
# Then select the displayed groups.


def find_unmatched(
    drawing_ink: np.ndarray,
    output: np.ndarray,
    curves: Sequence[tuple[np.ndarray, str]] | None = None,
    annotation: np.ndarray | None = None,
    outline: np.ndarray | None = None,
    tangent: np.ndarray | None = None,
    *,
    line_style_mismatches: Sequence[Unmatched] = (),
) -> list[Unmatched]:
    """Collect material and line groups, join stretches, then select display groups.

    Material outlines suppress duplicate line groups. curves, from output_curves,
    add unsupported stretches: tails extend far lines, and the rest stand alone.
    Near lines rank last.
    Annotation ink is never missing and never encloses material, but output
    lines along it are not extra either. Only outline ink, by default all part
    ink, encloses material.
    """
    annotated = annotation is not None and annotation.any()
    part_ink = drawing_ink & ~annotation if annotated else drawing_ink
    # Thin both line sets so cluster size does not depend on stroke width; unlike
    # skeletonize, thin keeps two-pixel diagonal strokes whole.
    drawing_lines, output_lines = thin(part_ink), thin(output)
    if not (drawing_lines.any() and output_lines.any()):
        return []
    all_lines = thin(drawing_ink) if annotated else drawing_lines
    # Limit input-only candidates to the output's vicinity to avoid far annotations.
    part = _vicinity(output_lines)

    # Filled silhouettes expose broad material differences beyond individual lines.
    input_area, output_area = _material_areas(
        part_ink if outline is None else outline, output
    )
    material = (
        _material_groups(input_area & ~output_area & part, "missing")
        + _material_groups(output_area & ~input_area, "extra")
        if input_area.any()
        else []
    )

    # Far lines are compared in both directions; material outlines are shown once.
    # NOTE: missing input lines are measured from CAD lines, which tangents extend;
    #       extra CAD lines are measured from input lines (optional tangents excluded).
    supporting_lines = output_lines if tangent is None else output_lines | tangent
    output_distances = distance_map(all_lines)
    input_lines = drawing_lines & part
    input_distances = distance_map(supporting_lines)
    lines = _far_line_groups(input_lines, input_distances, "missing")
    lines += _far_line_groups(output_lines, output_distances, "extra")
    near_lines = _near_line_groups(input_lines, input_distances)
    # A material group already stands for the lines along its edge.
    edge = _outline(material, drawing_lines.shape)
    lines = [line for line in lines if edge[line.xy[:, 1], line.xy[:, 0]].mean() < 0.8]
    near_lines = [g for g in near_lines if edge[g.xy[:, 1], g.xy[:, 0]].mean() < 0.8]

    if curves is not None:
        stretches = _unsupported_stretches(curves, output_distances)
        stretches = [g for g in stretches if edge[g.xy[:, 1], g.xy[:, 0]].mean() < 0.8]
        lines = _join_stretches(lines, stretches, drawing_lines.shape)

    # Qualified style errors need no geometric distance or far line length threshold.
    style_band = np.zeros(drawing_ink.shape, np.uint8)
    for group in line_style_mismatches:
        style_band[group.xy[:, 1], group.xy[:, 0]] = 1
    style_band = cv2.dilate(style_band, disk(3))
    lines = [g for g in lines if style_band[g.xy[:, 1], g.xy[:, 0]].mean() < 0.8]
    near_lines = [
        g for g in near_lines if style_band[g.xy[:, 1], g.xy[:, 0]].mean() < 0.8
    ]

    # Material first, then all lines by length, then near lines by length.
    ranked = sorted(material, key=attrgetter("size_px"), reverse=True)
    ranked += sorted([*line_style_mismatches, *lines], key=_length_px, reverse=True)
    ranked += sorted(near_lines, key=_length_px, reverse=True)
    return ranked[:MAX_GROUPS]


################################################################

# Describe and render the same selected groups in the same color order.


def describe_unmatched(groups: list[Unmatched]) -> list[dict[str, Any]]:
    """Line sizes are lengths; material sizes count filled pixels."""
    return [
        {
            "direction": group.direction,
            "kind": "material" if group.material else "lines",
            "unit": "px²" if group.material else "px",
            "size_px": group.size_px if group.material else round(_length_px(group)),
            "box_px": list(group.box_px),
            "color": color,
            **(
                {
                    "input_style": group.input_style,
                    "output_style": group.output_style,
                    "coverage": group.coverage,
                }
                if group.direction == "line_style"
                else {}
            ),
        }
        for group, color in zip(groups, COLORS, strict=False)
    ]


def _unshared(
    stroke: np.ndarray, centre: np.ndarray, other_centre: np.ndarray
) -> np.ndarray:
    """Stroke pixels whose nearest centre pixel lies beyond SHARED_PX of the
    other image's centre lines, so stroke widths do not decide what is shared."""
    if not centre.any():
        return np.zeros_like(stroke)
    apart = centre & (distance_map(other_centre) > SHARED_PX)
    _, (rows, cols) = distance_transform_edt(~centre, return_indices=True)
    return stroke & apart[rows, cols]


def draw_unmatched(
    drawing_gray: np.ndarray,
    output: np.ndarray,
    groups: list[Unmatched],
    annotation: np.ndarray | None = None,
    *,
    tangent: np.ndarray | None = None,
) -> np.ndarray:
    """Each group as a colored band under lines colored by the image that draws them.

    Lines both images draw are blue; a line only one image draws takes its
    color, input green and output magenta, however near the other image's line.
    Whether that is a difference is left to the groups. A tangent can share an
    input line, and annotation an output line. A band is wider than the lines
    drawn over it. Material is banded along its outline and hatched inside.
    Annotation ink, which the comparison does not count as the part, ink
    beyond the output's vicinity, which it never compares, and optional
    tangents are gray.

    output contains display strokes, not the dilated comparison mask.
    """
    image = np.full((*drawing_gray.shape, 3), 255, np.uint8)
    # Paint bands underneath the source lines so their mismatch remains readable.
    for group, color in zip(groups, COLORS.values()):
        mask = np.zeros(drawing_gray.shape, np.uint8)
        mask[group.xy[:, 1], group.xy[:, 0]] = 1
        if group.material:
            outline, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            cv2.drawContours(image, outline, -1, color, 2 * BAND_PX + 1)
            stripe = (group.xy[:, 0] + group.xy[:, 1]) % HATCH_PX < 3
            image[group.xy[stripe, 1], group.xy[stripe, 0]] = color
        else:
            image[cv2.dilate(mask, disk(BAND_PX)) > 0] = color

    ink = drawing_gray < 128
    centre = thin(ink)
    # Thick lines narrow to show the band beside them; thin ones keep their width,
    # since widening would fill small text and details.
    drawn = (cv2.dilate(centre.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & ink
    uncompared = ink & ~_vicinity(output)
    annotation = uncompared if annotation is None else annotation | uncompared
    image[annotation] = NEUTRAL_GRAY
    drawn &= ~annotation
    # A tangent lies under the input line it matches.
    if tangent is not None:
        image[tangent] = NEUTRAL_GRAY
    output_centre = thin(output)
    supporting = output_centre if tangent is None else output_centre | thin(tangent)
    image[drawn] = MATCHED_BLUE
    image[_unshared(drawn, centre, supporting)] = INPUT_GREEN
    image[output] = MATCHED_BLUE
    image[_unshared(output, output_centre, centre)] = OUTPUT_MAGENTA

    occupied = []
    # ponytail: greedy placement for at most six labels; no full layout solver.
    shifts = sorted(
        ((dx, dy) for dx in range(-5, 6) for dy in range(-5, 6)),
        key=lambda shift: abs(shift[0]) + abs(shift[1]),
    )
    for number, (group, color) in enumerate(zip(groups, COLORS.values()), 1):
        x, y = map(int, group.xy[np.argmin(group.xy[:, 1])])
        (width, height), baseline = cv2.getTextSize(
            str(number), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 10
        )
        box_width, box_height = width + 10, height + baseline + 10
        # Put the label above the group's topmost pixel, with a 5px halo.
        anchor_x, anchor_y = x - 5, y - height - 15
        for dx, dy in shifts:
            left = max(0, min(anchor_x + dx * box_width, image.shape[1] - box_width))
            top = max(0, min(anchor_y + dy * box_height, image.shape[0] - box_height))
            right, bottom = left + box_width, top + box_height
            if all(
                right <= x0 or left >= x1 or bottom <= y0 or top >= y1
                for x0, y0, x1, y1 in occupied
            ):
                break
        occupied.append((left, top, right, bottom))
        origin = (left + 5, top + height + 5)
        # A white halo keeps the number legible over lines and text.
        for ink, weight in (((255, 255, 255), 10), (color, 4)):
            cv2.putText(
                image, str(number), origin, cv2.FONT_HERSHEY_SIMPLEX, 1.5, ink, weight
            )
    return image
