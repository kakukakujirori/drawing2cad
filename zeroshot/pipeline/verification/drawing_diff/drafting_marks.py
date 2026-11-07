"""Centre lines and leaders: drafting marks a drawing adds that are not part edges.

A drawing marks the axis of a hole or a boss with a centre line, and points at
a hole or an arc with a leader. Extension lines and leaders can also cross the
outline to reach a feature inside the part. None of these is an edge of the
part, so the comparison does not list the input lines they explain as missing.

All are recognised by heuristics, kept here behind three calls:
record_centre_lines stores the axes of a build's revolved faces in each
projection DXF, and drafting_marks reads them back and finds leaders and lines
crossing into the part when the views are compared. thin_solid_marks finds lines
of the weight a drawing keeps for its dimensions. Without these calls every input
line counts as the part's.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import cadquery as cq
import cv2
import ezdxf
import numpy as np
from ezdxf.entities import Arc
from ezdxf.layouts import Modelspace
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepTools import BRepTools
from OCP.GeomAbs import GeomAbs_Cone, GeomAbs_Cylinder
from OCP.Standard import Standard_Failure
from skimage.morphology import skeletonize

from zeroshot.pipeline.stages.interpretation.contracts import AXIS_VECTOR, Axis

from .image_ops import (
    MIN_WIDTH_SAMPLES,
    WIDTH_RATIO,
    disk,
    distance_map,
    external_boundary,
    stroke_widths,
)

################################################################

# Centre lines, recorded when a view is projected.

# The record in a projection DXF that holds its centre lines. It is no entity,
# so it is neither drawn nor measured with the view's edges.
CENTRE_LINES = "CENTRE_LINES"

# A cylinder or cone face of at least half a turn has an axis to mark: a hole,
# a boss, or one half of either. Fillets turn less.
MIN_TURN = math.pi - 1e-3
# How far drawings carry centre lines, in radii of the feature: a centre
# mark's arm from the centre, and a centre line past each end of the face.
MARK_ARM = 2.0
LINE_OVERRUN = 1.0


@dataclass(frozen=True)
class CentreLine:
    """A centre line in a view's U and V, and the radius of what it marks."""

    start: tuple[float, float]
    end: tuple[float, float]
    radius: float


def record_centre_lines(
    dxf_path: Path, shape: cq.Shape, u_axis: Axis, v_axis: Axis
) -> None:
    """Store the centre lines of one view in its DXF, in the view's U and V."""
    lines = centre_lines(shape, u_axis, v_axis)
    if not lines:
        return
    doc = ezdxf.readfile(dxf_path)
    tags = []
    for line in lines:
        tags += [(10, (*line.start, 0.0)), (11, (*line.end, 0.0)), (40, line.radius)]
    doc.rootdict.add_xrecord(CENTRE_LINES).reset(tags)
    doc.saveas(dxf_path)


def centre_lines(shape: cq.Shape, u_axis: Axis, v_axis: Axis) -> list[CentreLine]:
    """A cross where an axis faces the viewer, a line where it lies in the sheet.

    Faces on one axis, such as a counterbore and its bore, share one line.
    Axes at other angles are left out.
    """
    u, v = (np.array(AXIS_VECTOR[axis]) for axis in (u_axis, v_axis))
    spans: dict[tuple[float, ...], list[tuple[float, float, float]]]
    spans = defaultdict(list)
    for face in shape.Faces():
        try:
            revolved = _revolved(face)
        except Standard_Failure:
            continue  # a face OCC cannot read marks nothing
        if revolved is None:
            continue
        origin, direction, radius, (low, high) = revolved
        centre = np.array([origin @ u, origin @ v])
        along = np.array([direction @ u, direction @ v])
        if np.linalg.norm(along) < 1e-6:
            arm = MARK_ARM * radius
            for heading in (np.array([1.0, 0.0]), np.array([0.0, 1.0])):
                _add_span(spans, centre, heading, -arm, arm, radius)
        elif np.linalg.norm(along) > 1 - 1e-6:
            overrun = LINE_OVERRUN * radius
            _add_span(spans, centre, along, low - overrun, high + overrun, radius)
    return _merged(spans)


def _revolved(face: cq.Face):
    """Axis origin and direction, radius and axial extent of a revolved face."""
    surface = BRepAdaptor_Surface(face.wrapped)
    u0, u1, v0, v1 = BRepTools.UVBounds_s(face.wrapped)
    if u1 - u0 < MIN_TURN:
        return None
    if surface.GetType() == GeomAbs_Cylinder:
        cylinder = surface.Cylinder()
        axis, radius, reach = cylinder.Axis(), cylinder.Radius(), (v0, v1)
    elif surface.GetType() == GeomAbs_Cone:
        # A cone's V runs along its slant, from the radius at its reference.
        cone = surface.Cone()
        axis, slant = cone.Axis(), cone.SemiAngle()
        radius = max(abs(cone.RefRadius() + v * math.sin(slant)) for v in (v0, v1))
        reach = (v0 * math.cos(slant), v1 * math.cos(slant))
    else:
        return None
    if radius <= 0:
        return None
    location, heading = axis.Location(), axis.Direction()
    origin = np.array([location.X(), location.Y(), location.Z()])
    direction = np.array([heading.X(), heading.Y(), heading.Z()])
    return origin, direction, radius, (min(reach), max(reach))


def _add_span(spans, centre, heading, low, high, radius) -> None:
    """File the span from low to high along heading, from centre, under its line."""
    heading = heading / np.linalg.norm(heading)
    if heading[0] < -1e-9 or (abs(heading[0]) <= 1e-9 and heading[1] < 0):
        heading, low, high = -heading, -high, -low
    normal = np.array([-heading[1], heading[0]])
    key = (*(round(float(x), 6) for x in heading), round(float(centre @ normal), 6))
    middle = float(centre @ heading)
    spans[key].append((middle + low, middle + high, radius))


def _merged(spans) -> list[CentreLine]:
    """One line per run of overlapping spans; it keeps the smallest radius."""
    lines = []
    for (dx, dy, offset), parts in sorted(spans.items()):
        heading, normal = np.array([dx, dy]), np.array([-dy, dx])
        parts = sorted(parts)
        start, end, radius = parts[0]
        for low, high, r in [*parts[1:], (math.inf, math.inf, 0.0)]:
            if low <= end:
                end, radius = max(end, high), min(radius, r)
                continue
            a, b = (offset * normal + t * heading for t in (start, end))
            lines.append(
                CentreLine(
                    (float(a[0]), float(a[1])), (float(b[0]), float(b[1])), radius
                )
            )
            start, end, radius = low, high, r
    return lines


################################################################

# Marks found when a view is compared, on the input drawing's pixel grid.

# Input lines nearer than this to an output line are the part's.
OUTPUT_PX = 3.0
# A centre line runs along a recorded axis: each piece of it lies this near the
# axis, though within half the feature's radius so a small bore's sides stay
# out, and within its length; and it turns from it by no more than MAX_SLOPE.
AXIS_PX = 5.0
MAX_SLOPE = math.tan(math.radians(10))
# A leader is straight: a run at least this long whose pixels lie this near one
# line, with gaps no longer than GAP_PX.
MIN_RUN_PX = 20
LINE_PX = 1.5
GAP_PX = 4
# It is radial: it passes this near the centre, plus a share of the radius.
RADIAL_PX = 3.0
RADIAL_SHARE = 0.05
# It reaches its circle, or the circle's centre, this closely. On an arc it
# meets the arc away from both ends: a sector's radial edge ends on one.
TOUCH_PX = 3.0
ARC_END_PX = 6.0
# Its other end joins annotation within this distance: straight on, or after
# the one bend to its text.
JOIN_PX = 6
# Following a line, this many blank pixels in a row end it.
TRACE_GAP_PX = 3
# An annotation line crossing into the part reaches the annotation beside it
# within CROSS_PX along its ink, across the outline and the margins kept free
# around output lines and the silhouette; a build's outline can lie some pixels
# off the drawing's. That annotation carries the same line on over CARRY_SHARE
# of MIN_RUN_PX, dashes and all.
CROSS_PX = 25
CARRY_SHARE = 0.8
# A solid stroke keeps one width. Dashes whose pale gaps binarizing bridges do
# not: the spread of their widths, p90 less p10, reaches this share of the median.
MAX_WIDTH_SPREAD = 0.1
# A neighbouring line can only widen a stroke, so the outline's weight is read
# low among its widths.
OUTLINE_PERCENTILE = 25


@dataclass(frozen=True)
class _Circle:
    """A circle or arc of the build: centre, radius and arc ends in input pixels."""

    centre: np.ndarray
    radius: float
    arc: Arc | None = None
    arc_ends: tuple[np.ndarray, ...] = ()


class _Run(NamedTuple):
    """A straight run of pixels: its middle, unit direction and two ends."""

    pixels: np.ndarray
    middle: np.ndarray
    direction: np.ndarray
    ends: tuple[np.ndarray, np.ndarray]


def drafting_marks(
    modelspace: Modelspace,
    model_to_drawing: np.ndarray,
    ink: np.ndarray,
    annotation: np.ndarray,
    output: np.ndarray,
    *,
    part_edges: np.ndarray,
) -> np.ndarray:
    """Input ink drawn as a centre line, a leader, or an annotation line crossing
    into the part, where no output line runs.

    model_to_drawing maps the projection's model U and V to input pixel
    boundaries. annotation is ink already known not to be the part's, such as
    ink beside it; a leader ends at it, and a crossing line runs on into it.
    output holds the build's lines on the input grid.
    part_edges holds input strokes whose weight rules out mere crossing annotation.
    """
    skeleton = skeletonize(ink & ~annotation)
    free = skeleton & (distance_map(skeletonize(output)) > OUTPUT_PX)
    boundary = distance_map(external_boundary(ink & ~annotation)) <= JOIN_PX
    junctions = _junctions(skeleton) & ~boundary
    gap_boundaries = junctions | boundary
    marks = np.zeros_like(free)
    # A line that crosses a centre line parts from it at their junction.
    ys, xs = np.nonzero(free & ~_junctions(free))
    pieces = np.column_stack([xs, ys])
    for start, end, radius in _recorded(modelspace, model_to_drawing):
        _mark_along(marks, pieces, start, end, min(AXIS_PX, radius / 2))
    supported = marks.copy()

    inked = distance_map(ink) <= LINE_PX
    circles = list(_circles(modelspace, model_to_drawing))
    drawing_to_model = np.linalg.inv(model_to_drawing)
    # Leaders and crossing lines end at annotation or at marks found before
    # them: a leader's shoulder may be a crossing line, and a crossing line may
    # run on from a leader. Look again while either finds more.
    while True:
        # Return whole strokes, not their centres.
        marked = _strokes(marks, ink, annotation)
        # Only axis/leader evidence may exclude protected strokes after widening.
        marked &= ~part_edges | _strokes(supported, ink, annotation)
        known = annotation | marked
        joined = cv2.dilate(known.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        found = np.zeros_like(free)
        for run in _straight_runs(free & ~marks):
            # Hough segments overlap; a run already taken needs no second look.
            if found[run.pixels[:, 1], run.pixels[:, 0]].all():
                continue
            if _runs_on_into(run, inked, joined):
                crossing = run.pixels[~part_edges[run.pixels[:, 1], run.pixels[:, 0]]]
                # shortcut: thin solid edges remain ambiguous; add stronger evidence if these recur.
                if len(crossing) and not _has_gaps(_fit(crossing), ink, gap_boundaries):
                    found[crossing[:, 1], crossing[:, 0]] = True
                    continue
            if _has_gaps(run, ink, gap_boundaries):
                continue
            has_part_edge = part_edges[run.pixels[:, 1], run.pixels[:, 0]].any()
            for circle in circles:
                leader = _leader(
                    run, circle, inked, known, drawing_to_model,
                    has_part_edge=has_part_edge,
                )
                if leader is not None:
                    # Output lines it crosses split it; it runs on to its tip.
                    leader_pixels = _along(free, *leader)
                    found |= leader_pixels
                    supported |= leader_pixels
        if not (found & ~marks).any():
            return marked
        marks |= found


def thin_solid_marks(
    gray: np.ndarray,
    ink: np.ndarray,
    annotation: np.ndarray,
    output: np.ndarray,
    tangent: np.ndarray,
) -> np.ndarray:
    """Strokes of the weight of known annotation, where no output line runs.

    A drawing can print dimension and extension lines thinner than the part's
    outline; annotation, ink known not to be the part's, shows whether this one
    does. Where it does, a stroke of that weight unbroken over MIN_RUN_PX or more
    is such a line, unless a gap along it or a varying width shows the dashes of
    a hidden line. A drawing in one line weight has no such strokes.
    """
    none = np.zeros_like(ink)
    centres, widths = stroke_widths(gray, ink)
    outline = widths[centres & (distance_map(external_boundary(output)) <= 2)]
    known = widths[centres & annotation]
    if min(outline.size, known.size) < MIN_WIDTH_SAMPLES:
        return none
    weight = np.median(known)
    if weight >= np.percentile(outline, OUTLINE_PERCENTILE) / WIDTH_RATIO:
        return none
    limit = WIDTH_RATIO * weight
    junctions = _junctions(centres)
    thin = centres & ~junctions & (widths < limit)
    count, labels, boxes, _ = cv2.connectedComponentsWithStats(
        thin.astype(np.uint8), connectivity=8
    )
    solid = none.copy()
    for label in range(1, count):
        x, y, width, height = boxes[label, :4]
        if math.hypot(width, height) < MIN_RUN_PX:
            continue
        ys, xs = np.nonzero(labels[y : y + height, x : x + width] == label)
        ys, xs = ys + y, xs + x
        low, middle, high = np.percentile(widths[ys, xs], [10, 50, 90])
        if high - low < MAX_WIDTH_SPREAD * middle and not _has_gaps(
            _fit(np.column_stack([xs, ys])), ink, junctions
        ):
            solid[ys, xs] = True
    # Only as wide as such a stroke, so a line beside it keeps its edge.
    reach = disk(math.ceil(limit))
    strokes = cv2.dilate((solid & ~annotation).astype(np.uint8), reach).astype(bool)
    return strokes & ink & ~annotation & (distance_map(output | tangent) > OUTPUT_PX)


def _strokes(
    centres: np.ndarray, ink: np.ndarray, annotation: np.ndarray
) -> np.ndarray:
    return (
        cv2.dilate(centres.astype(np.uint8), disk(3)).astype(bool) & ink & ~annotation
    )


def _to_pixels(model_to_drawing: np.ndarray, point) -> np.ndarray:
    mapped = model_to_drawing @ np.array([point[0], point[1], 1.0])
    # Pixel boundaries to OpenCV pixel centres.
    return mapped[:2] / mapped[2] - 0.5


def _recorded(
    modelspace: Modelspace, model_to_drawing: np.ndarray
) -> Iterator[tuple[np.ndarray, np.ndarray, float]]:
    """Each recorded centre line: its ends and the feature radius, in pixels."""
    record = modelspace.doc.rootdict.get(CENTRE_LINES)
    if record is None:
        return
    scale = math.sqrt(abs(np.linalg.det(model_to_drawing[:2, :2])))
    values = [tag.value for tag in record.tags if tag.code in (10, 11, 40)]
    for start, end, radius in zip(values[::3], values[1::3], values[2::3], strict=True):
        yield (
            _to_pixels(model_to_drawing, start),
            _to_pixels(model_to_drawing, end),
            radius * scale,
        )


def _junctions(skeleton: np.ndarray) -> np.ndarray:
    """Pixels where three or more skeleton branches meet, with their neighbours."""
    lines = skeleton.astype(np.uint8)
    neighbours = cv2.filter2D(lines, -1, np.ones((3, 3), np.float32)) - lines
    meeting = (lines > 0) & (neighbours >= 3)
    return cv2.dilate(meeting.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0


def _mark_along(
    marks: np.ndarray,
    pixels: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    reach: float,
) -> None:
    """Mark the pieces of line that run along the axis from start to end.

    A line that crosses the axis, or carries on past its end, is no centre line.
    """
    length = np.linalg.norm(end - start)
    if length < 1e-9 or not len(pixels):
        return
    direction = (end - start) / length
    offset = pixels - start
    along = offset @ direction
    across = offset @ np.array([-direction[1], direction[0]])
    # Look past each end, so that a line which outruns the axis is seen to.
    near = np.flatnonzero(
        (np.abs(across) <= reach)
        & (np.abs(along - length / 2) <= length / 2 + 2 * AXIS_PX)
    )
    if not len(near):
        return
    # Label the band's pieces in its own box, not on the whole image.
    (x0, y0), (x1, y1) = pixels[near].min(axis=0), pixels[near].max(axis=0)
    band = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
    band[pixels[near, 1] - y0, pixels[near, 0] - x0] = 1
    _, labels = cv2.connectedComponents(band)
    owners = labels[pixels[near, 1] - y0, pixels[near, 0] - x0]
    for owner in np.unique(owners):
        own = near[owners == owner]
        a, c = along[own], across[own]
        if (
            a.min() >= -AXIS_PX
            and a.max() <= length + AXIS_PX
            and np.ptp(a) >= 2 * (np.ptp(c) + 1)
            and abs(np.polyfit(a, c, 1)[0]) <= MAX_SLOPE
        ):
            marks[pixels[own, 1], pixels[own, 0]] = True


def _circles(modelspace: Modelspace, model_to_drawing: np.ndarray) -> Iterator[_Circle]:
    scale = math.sqrt(abs(np.linalg.det(model_to_drawing[:2, :2])))
    for entity in modelspace.query("CIRCLE ARC"):
        if entity.dxf.layer == "TANGENT":
            continue
        centre = _to_pixels(model_to_drawing, entity.dxf.center)
        radius = entity.dxf.radius * scale
        if isinstance(entity, Arc):
            ends = (entity.start_point, entity.end_point)
            yield _Circle(
                centre,
                radius,
                entity,
                tuple(_to_pixels(model_to_drawing, end) for end in ends),
            )
        else:
            yield _Circle(centre, radius)


def _straight_runs(mask: np.ndarray) -> list[_Run]:
    """Straight runs of mask pixels.

    The probabilistic Hough transform finds segments across the gaps of a
    dashed line. A trace at a slightly wrong angle can take a short line's
    pixels before the line itself is traced, so the straight sides of each
    piece's outline, which do not depend on the order pixels are visited in,
    are added. Each run takes the pixels along its segment's line, out to the
    first gap longer than GAP_PX, since a segment can be shorter than its line.
    """
    segments = _hough(mask) + _outline_sides(mask)
    if not segments:
        return []
    ys, xs = np.nonzero(mask)
    pixels = np.column_stack([xs, ys])
    runs = []
    for x0, y0, x1, y1 in segments:
        start, end = np.array([x0, y0], float), np.array([x1, y1], float)
        direction = (end - start) / np.linalg.norm(end - start)
        offset = pixels - start
        on_line = np.abs(offset @ np.array([-direction[1], direction[0]])) <= LINE_PX
        members = np.flatnonzero(on_line)
        along = offset[members] @ direction
        order = np.argsort(along)
        stretches = np.split(order, np.flatnonzero(np.diff(along[order]) > GAP_PX) + 1)
        # The stretch that holds the segment's own start.
        stretch = min(stretches, key=lambda part: np.abs(along[part]).min())
        run = _fit(pixels[members[stretch]])
        # An outline side can cut across a bend, past the pixels of its line.
        if np.linalg.norm(run.ends[1] - run.ends[0]) >= MIN_RUN_PX:
            runs.append(run)
    return runs


def _hough(mask: np.ndarray) -> list[np.ndarray]:
    segments = cv2.HoughLinesP(
        mask.astype(np.uint8),
        rho=1,
        theta=np.pi / 360,
        threshold=MIN_RUN_PX // 2,
        minLineLength=MIN_RUN_PX,
        maxLineGap=GAP_PX,
    )
    return [] if segments is None else list(segments.reshape(-1, 4))


def _outline_sides(mask: np.ndarray) -> list[np.ndarray]:
    """Sides at least MIN_RUN_PX long of each piece's outline, simplified to
    within LINE_PX; a thin piece's outline runs along it and back."""
    outlines, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )
    sides = set()
    for outline in outlines:
        corners = cv2.approxPolyDP(outline, LINE_PX, True).reshape(-1, 2)
        for start, end in zip(corners, np.roll(corners, -1, axis=0)):
            if np.linalg.norm(end - start) >= MIN_RUN_PX:
                sides.add(tuple(sorted((tuple(start), tuple(end)))))
    return [np.array([*start, *end]) for start, end in sorted(sides)]


def _fit(pixels: np.ndarray) -> _Run:
    middle = pixels.mean(axis=0)
    _, _, axes = np.linalg.svd(pixels - middle, full_matrices=False)
    direction = axes[0]
    along = (pixels - middle) @ direction
    ends = (middle + along.min() * direction, middle + along.max() * direction)
    return _Run(pixels, middle, direction, ends)


def _leader(
    run: _Run,
    circle: _Circle,
    inked: np.ndarray,
    annotation: np.ndarray,
    drawing_to_model: np.ndarray,
    *,
    has_part_edge: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    """A radial run that reaches the circle and whose ink ends at annotation:
    its far end and the tip its ink runs on to; None for any other run.
    """
    c, r = circle.centre, circle.radius
    normal = np.array([-run.direction[1], run.direction[0]])
    if abs((c - run.middle) @ normal) > RADIAL_PX + RADIAL_SHARE * r:
        return None
    near, far = sorted(run.ends, key=lambda end: np.linalg.norm(end - c))
    # Along the run toward the centre; the far end never sits on it.
    inward = run.direction if (c - far) @ run.direction > 0 else -run.direction

    def touches(point: np.ndarray) -> bool:
        gap = np.linalg.norm(point - c)
        return abs(gap - r) <= TOUCH_PX or gap <= TOUCH_PX

    crosses_circle = np.linalg.norm(near - c) < r < np.linalg.norm(far - c)
    if not (
        touches(near) or crosses_circle or _follow(near, inward, inked, touches)[1]
    ):
        return None
    tip = _follow(near, inward, inked)[0]
    outer_joins_annotation = _joins(_follow(far, -inward, inked)[0], annotation)
    if circle.arc is not None and not (
        _meets_arc(circle, c - inward * r, drawing_to_model)
        or (
            not has_part_edge
            and (tip - c) @ inward >= r - TOUCH_PX
            and outer_joins_annotation
            and _meets_arc(circle, c + inward * r, drawing_to_model)
        )
    ):
        return None
    if not (outer_joins_annotation or _joins(tip, annotation)):
        return None
    return far, tip


def _along(mask: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Mask pixels within LINE_PX of the segment from start to end."""
    ys, xs = np.nonzero(mask)
    points = np.column_stack([xs, ys]) - start
    span = end - start
    t = np.clip(points @ span / max(span @ span, 1e-9), 0, 1)
    near = np.linalg.norm(points - t[:, None] * span, axis=1) <= LINE_PX
    along = np.zeros_like(mask)
    along[ys[near], xs[near]] = True
    return along


def _meets_arc(
    circle: _Circle, point: np.ndarray, drawing_to_model: np.ndarray
) -> bool:
    arc = circle.arc
    assert arc is not None
    if any(np.linalg.norm(point - end) <= ARC_END_PX for end in circle.arc_ends):
        return False
    mapped = drawing_to_model @ np.array([*(point + 0.5), 1.0])
    x, y = mapped[:2] / mapped[2]
    angle = math.degrees(math.atan2(y - arc.dxf.center.y, x - arc.dxf.center.x))
    start, end = arc.dxf.start_angle, arc.dxf.end_angle
    return (angle - start) % 360 <= (end - start) % 360


def _has_gaps(run: _Run, ink: np.ndarray, boundaries: np.ndarray) -> bool:
    """Source gaps inside a run or between its neighboring stroke segments."""
    start, end = run.ends
    length = np.linalg.norm(end - start)
    # A wide dash gap splits Hough runs; look one run length past each end.
    first, last = start - length * run.direction, end + length * run.direction
    height, width = ink.shape
    x0, y0 = np.maximum(np.floor(np.minimum(first, last) - LINE_PX), 0).astype(int)
    x1, y1 = np.minimum(
        np.ceil(np.maximum(first, last) + LINE_PX) + 1, [width, height]
    ).astype(int)
    ys, xs = np.nonzero(ink[y0:y1, x0:x1])
    offset = np.column_stack([xs + x0, ys + y0]) - start
    along = offset @ run.direction
    normal = np.array([-run.direction[1], run.direction[0]])
    near = np.abs(offset @ normal) <= LINE_PX
    meetings = along[near & boundaries[ys + y0, xs + x0]]
    # Do not continue past a junction or the part's outline into other strokes.
    low = max(-length, meetings[meetings <= 0].max(initial=-length))
    high = min(2 * length, meetings[meetings >= length].min(initial=2 * length))
    positions = np.sort(along[near & (along >= low) & (along <= high)])
    if not len(positions):
        return True
    # A pixel's projected width keeps diagonally touching pixels continuous.
    pixel_span = np.abs(run.direction).sum()
    breaks = np.flatnonzero(np.diff(positions) > pixel_span + 1e-6)
    starts = positions[np.r_[0, breaks + 1]]
    ends = positions[np.r_[breaks, len(positions) - 1]]
    gaps = starts[1:] - ends[:-1] - pixel_span
    spans = ends - starts + pixel_span
    within = (ends[:-1] >= 0) & (starts[1:] <= length)
    overlaps = (ends >= 0) & (starts <= length)
    # A long extension followed by short dashes is not itself a dashed edge.
    neighboring = (
        (overlaps[:-1] | overlaps[1:])
        & (gaps <= np.minimum(spans[:-1], spans[1:]))
        & (np.minimum(spans[:-1], spans[1:]) >= length / 2)
    )
    return bool(np.any(within | neighboring))


def _runs_on_into(run: _Run, inked: np.ndarray, annotation: np.ndarray) -> bool:
    """Whether the ink carries the run's line from one end into annotation that
    continues the same line.
    """
    height, width = annotation.shape

    def reached(point: np.ndarray) -> bool:
        x, y = round(point[0]), round(point[1])
        return 0 <= x < width and 0 <= y < height and bool(annotation[y, x])

    for end in run.ends:
        outward = run.direction * np.sign((end - run.middle) @ run.direction)
        last, met = _follow(end, outward, inked, reached)
        if not met or np.linalg.norm(last - end) > CROSS_PX:
            continue
        ahead = last + np.arange(1, MIN_RUN_PX + 1)[:, None] * outward
        ahead = np.rint(ahead).astype(int)
        if not ((ahead >= 0) & (ahead < (width, height))).all():
            continue
        if annotation[ahead[:, 1], ahead[:, 0]].mean() > CARRY_SHARE:
            return True
    return False


def _follow(
    point: np.ndarray, step: np.ndarray, inked: np.ndarray, stop=None
) -> tuple[np.ndarray, bool]:
    """Walk along ink by a unit step: the last inked point, and if stop was met."""
    height, width = inked.shape
    last, blank = point.copy(), 0
    while blank <= TRACE_GAP_PX:
        point = point + step
        x, y = round(point[0]), round(point[1])
        if not (0 <= x < width and 0 <= y < height):
            break
        if not inked[y, x]:
            blank += 1
            continue
        last, blank = point.copy(), 0
        if stop is not None and stop(point):
            return last, True
    return last, False


def _joins(point: np.ndarray, annotation: np.ndarray) -> bool:
    x, y = round(point[0]), round(point[1])
    height, width = annotation.shape
    x0, x1 = max(x - JOIN_PX, 0), min(x + JOIN_PX + 1, width)
    y0, y1 = max(y - JOIN_PX, 0), min(y + JOIN_PX + 1, height)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    window = (xs - x) ** 2 + (ys - y) ** 2 <= JOIN_PX**2
    return bool((annotation[y0:y1, x0:x1] & window).any())
