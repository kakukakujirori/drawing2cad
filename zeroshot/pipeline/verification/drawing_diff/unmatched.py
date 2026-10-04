"""Lines and material one image has and the other lacks, in the drawing's pixels."""

from __future__ import annotations

from dataclasses import dataclass
from operator import attrgetter
from typing import Any, Literal

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
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
}


@dataclass(frozen=True)
class Unmatched:
    direction: Literal["missing", "extra"]
    xy: np.ndarray  # skeleton pixels of lines, or every pixel of material
    material: bool = False

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
    mask: np.ndarray, direction: Literal["missing", "extra"]
) -> list[Unmatched]:
    groups = _groups(mask, cv2.dilate(mask.astype(np.uint8), _disk(JOIN_PX)), direction)
    return [group for group in groups if group.size_px >= MIN_LENGTH_PX]


def _material_groups(
    mask: np.ndarray, direction: Literal["missing", "extra"]
) -> list[Unmatched]:
    opened = cv2.morphologyEx(
        mask.astype(np.uint8), cv2.MORPH_OPEN, _disk(MATERIAL_THIN_PX // 2)
    ).astype(bool)
    groups = _groups(opened, opened, direction, material=True)
    return [group for group in groups if group.size_px >= MIN_AREA_PX]


def _filled(ink: np.ndarray) -> np.ndarray:
    """The area inside the outermost lines."""
    closed = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_CLOSE, _disk(CLOSE_PX))
    return binary_fill_holes(closed)


def measure_material_error(
    drawing_ink: np.ndarray, output: np.ndarray
) -> dict[str, Any]:
    """Unfiltered mismatch of the extracted filled silhouettes after alignment."""
    input_area = drawing_area(drawing_ink)
    if not input_area.any():
        return {
            "status": "unavailable",
            "reason": "input part area unavailable",
            "ratio": None,
        }
    output_area = _filled(output)
    missing = int((input_area & ~output_area).sum())
    extra = int((output_area & ~input_area).sum())
    union = int((input_area | output_area).sum())
    return {
        "status": "ok",
        "missing_px2": missing,
        "extra_px2": extra,
        "union_px2": union,
        "ratio": (missing + extra) / union,
    }


def _outline(groups: list[Unmatched], shape: tuple[int, ...]) -> np.ndarray:
    """A band around the edges of the groups' areas."""
    area = np.zeros(shape, np.uint8)
    for group in groups:
        area[group.xy[:, 1], group.xy[:, 0]] = 1
    band = cv2.dilate(area, _disk(OUTLINE_PX)) - cv2.erode(area, _disk(OUTLINE_PX))
    return band.astype(bool)


def find_unmatched(drawing_ink: np.ndarray, output: np.ndarray) -> list[Unmatched]:
    """The largest groups within the output's extent: material first, then lines.

    Both line sets are thinned first, so length does not depend on stroke width.
    """
    drawing_lines, output_lines = skeletonize(drawing_ink), skeletonize(output)
    if not (drawing_lines.any() and output_lines.any()):
        return []
    ys, xs = np.nonzero(output_lines)
    part = np.zeros_like(drawing_lines)
    part[
        max(ys.min() - PART_MARGIN_PX, 0) : ys.max() + PART_MARGIN_PX + 1,
        max(xs.min() - PART_MARGIN_PX, 0) : xs.max() + PART_MARGIN_PX + 1,
    ] = True
    input_area, output_area = drawing_area(drawing_ink), _filled(output)
    material = (
        _material_groups(input_area & ~output_area & part, "missing")
        + _material_groups(output_area & ~input_area, "extra")
        if input_area.any()
        else []
    )
    far_from_output = distance_map(output_lines) > UNMATCHED_PX
    far_from_drawing = distance_map(drawing_lines) > UNMATCHED_PX
    lines = _line_groups(drawing_lines & part & far_from_output, "missing")
    lines += _line_groups(output_lines & far_from_drawing, "extra")
    # A material group already stands for the lines along its edge.
    edge = _outline(material, drawing_lines.shape)
    lines = [line for line in lines if edge[line.xy[:, 1], line.xy[:, 0]].mean() < 0.8]
    size = attrgetter("size_px")
    ranked = sorted(material, key=size, reverse=True)
    ranked += sorted(lines, key=size, reverse=True)
    return ranked[: len(COLORS)]


def describe_unmatched(groups: list[Unmatched]) -> list[dict[str, Any]]:
    return [
        {
            "direction": group.direction,
            "kind": "material" if group.material else "lines",
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
    drawn = cv2.dilate(
        skeletonize(drawing_gray < 128).astype(np.uint8), np.ones((3, 3), np.uint8)
    ) > 0
    shown = cv2.dilate(output.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0
    image[drawn] = INPUT_GRAY
    image[shown] = OUTPUT_SKY
    image[drawn & shown] = BOTH_BLUE
    for number, (group, color) in enumerate(zip(groups, COLORS.values()), 1):
        x, y = group.xy[np.argmin(group.xy[:, 1])]  # the top of the lines, not the box
        origin = (int(x), max(int(y) - 10, 40))
        # A white halo keeps the number legible over lines and text.
        for ink, weight in (((255, 255, 255), 10), (color, 4)):
            cv2.putText(
                image, str(number), origin, cv2.FONT_HERSHEY_SIMPLEX, 1.5, ink, weight
            )
    return image
