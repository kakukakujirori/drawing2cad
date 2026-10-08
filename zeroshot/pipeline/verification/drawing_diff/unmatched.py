"""Find and display differences in the input drawing's pixel coordinates.

Shared pixel operations support silhouette and line comparison. Local DXF
geometry adds fine edge differences to the coarse candidates. find_unmatched
merges them before selecting display groups; descriptions and drawing use
that same selection.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from operator import attrgetter
from typing import Any, Literal

import cv2
import numpy as np
from ezdxf.layouts import Modelspace
from scipy.ndimage import binary_fill_holes
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from .image_ops import distance_map, drawing_area

# Tuned on two drawings and their ground truth: outputs/a2_gate_20260927.
UNMATCHED_PX = 12.0
JOIN_PX = 4
PART_MARGIN_PX = 20
MIN_LENGTH_PX = 50

# Tuned the same way: outputs/silhouette_20260927.
CLOSE_PX = 3
# Narrower slivers come from misregistered edges and from annotation.
MATERIAL_THIN_PX = 30
MIN_AREA_PX = 1000
# A line group this close to a material group's outline is that outline;
# the width covers the corners that the material's opening rounds off.
OUTLINE_PX = 8

# Local edge finishes: the same radius limits curvature and expands the seed ROI.
DETAIL_RADIUS_PX = 40
DETAIL_UNMATCHED_PX = 4.0
DETAIL_MIN_LENGTH_PX = 12
# Fine tails may extend coarse defects; independent local labels need longer runs.
MIN_LOCAL_LENGTH_PX = 24

# Display limits reserve coarse slots before independent local edge finishes.
MAX_COARSE_GROUPS = 4
MAX_LOCAL_GROUPS = 2
# Wider than the displayed strokes, so the band shows on both sides of them.
BAND_PX = 5
INPUT_GRAY = (110, 110, 110)
OUTPUT_SKY = (90, 190, 255)
# Where both draw a line, the gray over the light blue deepens it to pure blue.
BOTH_BLUE = (0, 0, 255)
# Material is hatched, as a drawing marks cut material, so it reads as an area.
HATCH_PX = 12
# One bright color per listed group, in rank order. Neighbouring hues are kept
# apart, and none is near the gray, light blue or pure blue lines drawn on top.
COLORS = {
    "red": (230, 0, 0),
    "magenta": (220, 0, 220),
    "pink": (255, 80, 190),
    "orange": (255, 140, 0),
    "yellow": (215, 165, 0),
    "green": (0, 155, 50),
}


################################################################

# Shared pixel operations:
# Coordinates, cluster measurements and connectivity.


@dataclass(frozen=True)
class Unmatched:
    """One difference group, measured on the input drawing's pixel grid."""

    direction: Literal["missing", "extra"]
    xy: np.ndarray  # (x, y): skeleton pixels of lines, or every pixel of material
    material: bool = False
    local: bool = False  # independent fine error; coarse groups may include fine tails

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
    thick = cv2.dilate(output_ink.astype(np.uint8), np.ones((size, size), np.uint8))
    warped = cv2.warpPerspective(
        thick * 255,
        np.linalg.inv(matrix),
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderValue=0,
    )
    return warped > 0


def _disk(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)


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
    min_length: int = MIN_LENGTH_PX,
) -> list[Unmatched]:
    groups = _groups(mask, cv2.dilate(mask.astype(np.uint8), _disk(JOIN_PX)), direction)
    return [group for group in groups if group.size_px >= min_length]


################################################################

# Material differences:
# Filled silhouettes, area statistics and outline bands.


def _filled(ink: np.ndarray) -> np.ndarray:
    """The area inside the outermost lines."""
    closed = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_CLOSE, _disk(CLOSE_PX))
    return binary_fill_holes(closed)


def _material_groups(
    mask: np.ndarray, direction: Literal["missing", "extra"]
) -> list[Unmatched]:
    opened = cv2.morphologyEx(
        mask.astype(np.uint8), cv2.MORPH_OPEN, _disk(MATERIAL_THIN_PX // 2)
    ).astype(bool)
    groups = _groups(opened, opened, direction, material=True)
    return [group for group in groups if group.size_px >= MIN_AREA_PX]


def measure_material_error(
    drawing_ink: np.ndarray, output: np.ndarray
) -> dict[str, Any]:
    """Unfiltered silhouette statistics, independent of display groups and quotas.

    The thin-sliver and cluster-size filters used for display do not affect
    these counts of the extracted silhouette errors.
    """
    input_area = drawing_area(drawing_ink)
    if not input_area.any():
        return {
            "status": "unavailable",
            "reason": "the input outline cannot be filled (thin walls or open lines)",
            "ratio": None,
        }
    output_area = _filled(output)
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


def _outline(groups: list[Unmatched], shape: tuple[int, ...]) -> np.ndarray:
    """A band around the edges of the groups' areas."""
    area = np.zeros(shape, np.uint8)
    for group in groups:
        area[group.xy[:, 1], group.xy[:, 0]] = 1
    band = cv2.dilate(area, _disk(OUTLINE_PX)) - cv2.erode(area, _disk(OUTLINE_PX))
    return band.astype(bool)


################################################################

# Local line differences:
# Output DXF geometry, coarse/fine merging and acceptance.


@dataclass(frozen=True)
class DetailGeometry:
    """Local ROI and continuous output curves on the drawing's pixel grid.

    samples.data, curvature and owners share indices; owners contain curve IDs.
    corners contain (first_curve_id, second_curve_id, corner_xy).
    """

    mask: np.ndarray  # neighborhood of high-curvature points and angular joins
    visible: np.ndarray  # continuous output strokes eligible for fine comparison
    samples: cKDTree
    curvature: np.ndarray
    owners: np.ndarray
    corners: list[tuple[int, int, np.ndarray]]


def _curvature(xy: np.ndarray) -> np.ndarray:
    """Inverse circumradius of each consecutive triple, in drawing pixels."""
    segments = np.diff(xy, axis=0)
    a, b = segments[:-1], segments[1:]
    lengths = np.linalg.norm(segments, axis=1)
    denominator = lengths[:-1] * lengths[1:] * np.linalg.norm(a + b, axis=1)
    cross = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
    return np.divide(
        2 * cross, denominator, out=np.zeros_like(cross), where=denominator > 0
    )


def detail_geometry(
    modelspace: Modelspace, model_to_drawing: np.ndarray, shape: tuple[int, int]
) -> DetailGeometry:
    """Local ROIs and sampled continuous output curves, in drawing pixels.

    Curvature is measured after mapping to drawing pixels; its radius must be
    no larger than DETAIL_RADIUS_PX. Continuous entities avoid fine errors due
    to hidden-line dash phase. Angular joins seed ROIs for omitted edge finishes.
    The transform uses pixel boundaries; OpenCV strokes use integer centres.
    """
    from ezdxf.path import make_path

    seeds = np.zeros(shape, np.uint8)
    visible = np.zeros(shape, np.uint8)
    endpoints, directions = [], []
    samples, curvatures, owners = [], [], []
    tolerance = 0.5 / np.linalg.norm(model_to_drawing[:2, :2], axis=0).max()

    # Map eligible DXF curves first, so curvature is judged in drawing pixels.
    for entity in modelspace:
        linetype = entity.dxf.get("linetype", "BYLAYER").upper()
        if linetype == "BYLAYER":
            linetype = modelspace.doc.layers.get(entity.dxf.layer).dxf.linetype.upper()
        if linetype != "CONTINUOUS" or entity.dxftype() not in {
            "LINE",
            "ARC",
            "CIRCLE",
            "ELLIPSE",
            "SPLINE",
        }:
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
        if len(xy) < 2:
            continue
        cv2.polylines(visible, [np.rint(xy).astype(np.int32)], False, 1, 3)
        segments = np.diff(xy, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        curvature = _curvature(xy)
        # Two endpoints per curve keep endpoint_index // 2 aligned with curve IDs.
        endpoints.extend([xy[0], xy[-1]])
        directions.extend([segments[0] / lengths[0], -segments[-1] / lengths[-1]])
        # Seed only high-curvature portions, rather than an entire curved entity.
        for point in xy[1:-1][curvature >= 1 / DETAIL_RADIUS_PX]:
            cv2.circle(seeds, tuple(np.rint(point).astype(int)), 1, 1, -1)

        # Dense samples tie error pixels to their own curve, not a nearby ROI seed.
        distance = np.r_[0, np.cumsum(lengths)]
        along = np.linspace(0, distance[-1], max(2, int(np.ceil(distance[-1] * 2)) + 1))
        samples.append(np.column_stack([np.interp(along, distance, c) for c in xy.T]))
        k = np.pad(curvature, 1, mode="edge") if curvature.size else np.zeros(len(xy))
        curvatures.append(np.interp(along, distance, k))
        owners.append(np.full(len(along), len(owners), int))

    # Straight edges can expose a missing finish at their angular connection.
    corners = []
    if endpoints:
        for a, b in cKDTree(endpoints).query_pairs(1.5):
            if a // 2 == b // 2:
                continue
            angle = np.arccos(np.clip(abs(np.dot(directions[a], directions[b])), 0, 1))
            if angle > np.pi / 6:
                cv2.circle(seeds, tuple(np.rint(endpoints[a]).astype(int)), 1, 1, -1)
                corners.append((a // 2, b // 2, (endpoints[a] + endpoints[b]) / 2))

    # Proximity defines a search ROI; acceptance still checks the error's own curves.
    local_mask = seeds.astype(bool)
    if seeds.any():
        local_mask = distance_map(local_mask) <= DETAIL_RADIUS_PX
    return DetailGeometry(
        local_mask,
        visible.astype(bool),
        cKDTree(np.concatenate(samples) if samples else np.empty((0, 2))),
        np.concatenate(curvatures) if curvatures else np.empty(0),
        np.concatenate(owners) if owners else np.empty(0, int),
        corners,
    )


def _merge_extra_groups(
    coarse: list[Unmatched], fine: list[Unmatched], local_mask: np.ndarray
) -> list[Unmatched]:
    """Merge fine output errors with coarse ones without duplicate local labels.

    Only fine components wholly inside the ROI can stand alone. Others may
    extend a coarse defect, but detached ROI fragments must not inherit that
    coarse defect's eligibility.
    """
    # Preserve coarse provenance even when fine pixels extend the same cluster.
    extra = np.zeros(local_mask.shape, bool)
    for group in coarse:
        if group.direction == "extra":
            extra[group.xy[:, 1], group.xy[:, 0]] = True
    combined = extra.copy()
    bounded = np.zeros(local_mask.shape, bool)  # fine pixels allowed to stand alone

    # Inspect whole fine components before clipping, so long offsets stay broad.
    for group in fine:
        inside = local_mask[group.xy[:, 1], group.xy[:, 0]]
        if not inside.all() and not extra[group.xy[:, 1], group.xy[:, 0]].any():
            continue
        xy = group.xy[inside]
        if len(xy) >= DETAIL_MIN_LENGTH_PX:
            combined[xy[:, 1], xy[:, 0]] = True
            if inside.all():
                bounded[xy[:, 1], xy[:, 0]] = True

    # Clipping can split a component; classify each resulting cluster by provenance.
    merged = _groups(
        combined, cv2.dilate(combined.astype(np.uint8), _disk(JOIN_PX)), "extra"
    )
    return [g for g in coarse if g.direction == "missing"] + [
        replace(g, local=not extra[g.xy[:, 1], g.xy[:, 0]].any())
        for g in merged
        if extra[g.xy[:, 1], g.xy[:, 0]].any() or bounded[g.xy[:, 1], g.xy[:, 0]].any()
    ]


def _local_detail(group: Unmatched, geometry: DetailGeometry) -> bool:
    """Accept a whole local group via its own curvature or both sides of one corner.

    One high-curvature match qualifies the group; the corner branch requires
    errors on each incident curve away from the shared endpoint.
    """
    if not geometry.samples.n:
        return False
    # A nearby corner must not qualify an error on an unrelated straight curve.
    _, indices = geometry.samples.query(group.xy)
    if np.any(geometry.curvature[indices] >= 1 / DETAIL_RADIUS_PX):
        return True
    owners = geometry.owners[indices]
    for a, b, point in geometry.corners:
        if a not in owners or b not in owners:
            continue
        distance = np.linalg.norm(group.xy - point, axis=1)
        # The shared endpoint cannot establish an error on both sides of a corner.
        sides = owners[(distance > 1.5) & (distance <= DETAIL_RADIUS_PX)]
        if a in sides and b in sides:
            return True
    return False


################################################################

# Combine the difference candidates.
# Then select the displayed groups.


def find_unmatched(
    drawing_ink: np.ndarray,
    output: np.ndarray,
    geometry: DetailGeometry | None = None,
) -> list[Unmatched]:
    """Collect material/coarse defects, merge fine errors, then select display groups.

    Material outlines suppress duplicate line groups. Fine tails extend coarse
    groups before independent locals pass the size and curvature/corner checks.
    """
    # Thin both line sets so cluster size does not depend on stroke width.
    drawing_lines, output_lines = skeletonize(drawing_ink), skeletonize(output)
    if not (drawing_lines.any() and output_lines.any()):
        return []
    # Limit input-only candidates to the output's vicinity to avoid far annotations.
    ys, xs = np.nonzero(output_lines)
    part = np.zeros_like(drawing_lines)
    part[
        max(ys.min() - PART_MARGIN_PX, 0) : ys.max() + PART_MARGIN_PX + 1,
        max(xs.min() - PART_MARGIN_PX, 0) : xs.max() + PART_MARGIN_PX + 1,
    ] = True

    # Filled silhouettes expose broad material differences beyond individual lines.
    input_area, output_area = drawing_area(drawing_ink), _filled(output)
    material = (
        _material_groups(input_area & ~output_area & part, "missing")
        + _material_groups(output_area & ~input_area, "extra")
        if input_area.any()
        else []
    )

    # Coarse comparison works in both directions; material outlines are shown once.
    far_from_output = distance_map(output_lines) > UNMATCHED_PX
    output_distances = distance_map(drawing_lines)
    far_from_drawing = output_distances > UNMATCHED_PX
    lines = _line_groups(drawing_lines & part & far_from_output, "missing")
    lines += _line_groups(output_lines & far_from_drawing, "extra")
    # A material group already stands for the lines along its edge.
    edge = _outline(material, drawing_lines.shape)
    lines = [line for line in lines if edge[line.xy[:, 1], line.xy[:, 0]].mean() < 0.8]

    if geometry is not None:
        # Cluster before clipping to the ROI, so a broad offset stays broad.
        details = _line_groups(
            output_lines & geometry.visible & (output_distances > DETAIL_UNMATCHED_PX),
            "extra",
            DETAIL_MIN_LENGTH_PX,
        )
        details = [g for g in details if edge[g.xy[:, 1], g.xy[:, 0]].mean() < 0.8]
        lines = _merge_extra_groups(lines, details, geometry.mask)
        # The stricter local size/shape gate must not remove tails merged into coarse.
        lines = [
            g
            for g in lines
            if not g.local
            or (g.size_px >= MIN_LOCAL_LENGTH_PX and _local_detail(g, geometry))
        ]

    # Select after all filtering: material first, coarse lines next, two local slots.
    size = attrgetter("size_px")
    ranked = sorted(material, key=size, reverse=True)
    ranked += sorted((g for g in lines if not g.local), key=size, reverse=True)
    return (
        ranked[:MAX_COARSE_GROUPS]
        + sorted((g for g in lines if g.local), key=size, reverse=True)[
            :MAX_LOCAL_GROUPS
        ]
    )


################################################################

# Describe and render the same selected groups in the same color order.


def describe_unmatched(groups: list[Unmatched]) -> list[dict[str, Any]]:
    """Line sizes count skeleton pixels; material sizes count filled pixels."""
    return [
        {
            "direction": group.direction,
            "kind": "material" if group.material else "lines",
            "unit": "px²" if group.material else "px",
            "size_px": group.size_px,
            "box_px": list(group.box_px),
            "color": color,
        }
        for group, color in zip(groups, COLORS, strict=False)
    ]


def draw_unmatched(
    drawing_gray: np.ndarray, output: np.ndarray, groups: list[Unmatched]
) -> np.ndarray:
    """Each group as a colored band under the gray drawing and the light blue output.

    A band is wider than the lines drawn over it, so they show which image a
    group came from; lines both images draw are pure blue. Material is banded along
    its outline and hatched inside.
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
            image[cv2.dilate(mask, _disk(BAND_PX)) > 0] = color

    # Restore source context: input gray, output sky blue, overlap pure blue.
    drawn = skeletonize(drawing_gray < 128).astype(np.uint8)
    drawn = cv2.dilate(drawn, np.ones((3, 3), np.uint8)) > 0
    shown = cv2.dilate(output.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0
    image[drawn] = INPUT_GRAY
    image[shown] = OUTPUT_SKY
    image[drawn & shown] = BOTH_BLUE

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
