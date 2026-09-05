"""The reference split a model's own view assignment is scored against.

Only for three views in an L with clear space between them.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from zeroshot.pipeline.messages.contracts.drawings import (
    DrawingEvidence,
    DrawingSheet,
    DrawingSource,
    DrawnEntity,
    View,
)

# Narrower than any inter-view spacing and wider than any drawing detail: the
# gaps a slot or a counterbore leaves are millimetres, the gap between views
# is tens of them.
MIN_VIEW_GAP_MM = 2.0
# Views share an extent by construction, so a residual larger than this means
# the cut fell somewhere other than between two views.
ALIGNMENT_TOLERANCE_MM = 1.0
_SAMPLES_PER_SWEEP = 64

type Box = tuple[float, float, float, float]


class ViewSplitError(ValueError):
    """The page does not separate into a three-view arrangement."""


@dataclass(frozen=True)
class ViewPlacement:
    """A page separated into views, beside the residual that settled the cut."""

    drawing: DrawingSource
    boxes: Mapping[View, Box]
    alignment_error: float


# --- What each entity reaches ---


def _named_points(values: Mapping[str, list[float]]) -> list[tuple[float, float]]:
    found: list[tuple[float, float]] = []
    for name in ("start", "end", "center"):
        if name in values:
            found.append((values[name][0], values[name][1]))
    for name in ("control_points", "vertices"):
        if name in values:
            flat = values[name]
            found.extend(zip(flat[0::2], flat[1::2]))
    return found


def _circle_points(values: Mapping[str, list[float]]) -> list[tuple[float, float]]:
    (x, y), radius = values["center"], values["radius"][0]
    return [(x - radius, y - radius), (x + radius, y + radius)]


def _arc_points(values: Mapping[str, list[float]]) -> list[tuple[float, float]]:
    """The ends, and each quarter turn the arc sweeps past on its way."""
    (x, y), radius = values["center"], values["radius"][0]
    start = math.degrees(math.atan2(values["start"][1] - y, values["start"][0] - x))
    end = math.degrees(math.atan2(values["end"][1] - y, values["end"][0] - x))
    swept = (end - start) % 360 or 360

    def reached(degrees: float) -> tuple[float, float]:
        turn = math.radians(degrees)
        return (x + radius * math.cos(turn), y + radius * math.sin(turn))

    quarters = [(quarter - start) % 360 for quarter in (0, 90, 180, 270)]
    return [
        reached(start),
        reached(start + swept),
        *(reached(start + turn) for turn in quarters if turn <= swept),
    ]


def _ellipse_points(values: Mapping[str, list[float]]) -> list[tuple[float, float]]:
    """The swept arc, sampled: its extremes have no closed form worth carrying."""
    center, major = values["center"], values["major_axis"]
    semi_major = math.hypot(major[0], major[1])
    minor_radius = values["minor_radius"][0]
    across = (-major[1] / semi_major, major[0] / semi_major)

    def parameter_at(place: Sequence[float]) -> float:
        x, y = place[0] - center[0], place[1] - center[1]
        along = (x * major[0] + y * major[1]) / semi_major
        off = (y * major[0] - x * major[1]) / semi_major
        return math.atan2(off / minor_radius, along / semi_major) % math.tau

    start = parameter_at(values["start"])
    swept = (parameter_at(values["end"]) - start) % math.tau or math.tau
    steps = range(_SAMPLES_PER_SWEEP + 1)
    return [
        (
            center[0]
            + major[0] * math.cos(turn)
            + across[0] * minor_radius * math.sin(turn),
            center[1]
            + major[1] * math.cos(turn)
            + across[1] * minor_radius * math.sin(turn),
        )
        for turn in (start + swept * step / _SAMPLES_PER_SWEEP for step in steps)
    ]


_REACH: Mapping[
    DrawnEntity, Callable[[Mapping[str, list[float]]], list[tuple[float, float]]]
] = {
    DrawnEntity.CIRCLE: _circle_points,
    DrawnEntity.ARC: _arc_points,
    DrawnEntity.ELLIPSE: _ellipse_points,
}


def extent(entry: DrawingEvidence) -> Box:
    """The box one entity occupies on its page."""
    values = {parameter.name.value: parameter.values for parameter in entry.parameters}
    points = _REACH.get(entry.entity, _named_points)(values)
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    return min(xs), min(ys), max(xs), max(ys)


# --- Separating the page ---


def _gaps(spans: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """The empty intervals between spans, widest first."""
    found: list[tuple[float, float]] = []
    reached = -math.inf
    for low, high in sorted(spans):
        if low - reached >= MIN_VIEW_GAP_MM and reached > -math.inf:
            found.append((reached, low))
        reached = max(reached, high)
    return sorted(found, key=lambda gap: gap[0] - gap[1])


def _bounding(boxes: Sequence[Box]) -> Box:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _alignment_error(front: Box, top: Box, right: Box) -> float:
    """Top reuses front's horizontal extent and right reuses its vertical one."""
    return max(
        abs(front[0] - top[0]),
        abs(front[2] - top[2]),
        abs(front[1] - right[1]),
        abs(front[3] - right[3]),
    )


_QUADRANT_VIEW: Mapping[str, View] = {
    "bl": View.FRONT,
    "tl": View.TOP,
    "br": View.RIGHT,
}


def _split(reach: Mapping[str, Box]) -> tuple[dict[View, list[str]], float]:
    """Cut the page into the three populated quadrants of an L arrangement.

    An entity belongs to the quadrant its least corner falls in, and a view is
    what its quadrant's entities span, so every entity lands in exactly one.
    """
    if len(reach) < 3:
        raise ViewSplitError(f"only {len(reach)} entities on the page")
    boxes = list(reach.values())
    horizontal = _gaps([(box[0], box[2]) for box in boxes])
    vertical = _gaps([(box[1], box[3]) for box in boxes])
    if not horizontal or not vertical:
        raise ViewSplitError(
            f"need a gap of at least {MIN_VIEW_GAP_MM} mm each way, found "
            f"{len(horizontal)} across and {len(vertical)} down"
        )

    closest = math.inf
    for left, right_edge in horizontal:
        across = (left + right_edge) / 2
        for low, high in vertical:
            down = (low + high) / 2
            quadrants: dict[str, list[str]] = {"bl": [], "tl": [], "br": [], "tr": []}
            for name, box in reach.items():
                key = ("t" if box[1] > down else "b") + (
                    "r" if box[0] > across else "l"
                )
                quadrants[key].append(name)
            if quadrants["tr"] or not all(quadrants[key] for key in _QUADRANT_VIEW):
                continue
            filed = {view: quadrants[key] for key, view in _QUADRANT_VIEW.items()}
            spanned = {
                view: _bounding([reach[name] for name in names])
                for view, names in filed.items()
            }
            residual = _alignment_error(
                spanned[View.FRONT], spanned[View.TOP], spanned[View.RIGHT]
            )
            if residual <= ALIGNMENT_TOLERANCE_MM:
                return filed, residual
            closest = min(closest, residual)
    raise ViewSplitError(
        f"no cut leaves three views aligned; the closest misaligns their shared "
        f"extents by {closest:.2f} mm"
        if closest < math.inf
        else "no cut leaves the fourth quadrant empty"
    )


def _origin_on(view: View, box: Box) -> list[float]:
    """The part's least corner in x, y and z, placed by each view's own frame.

    Front reads right=+x up=+y, top right=+x up=-z, and right right=-z up=+y,
    so +z runs down the top view and leftward across the right one. The three
    views share the extents that carry x and y, so the corner they each place
    is the same physical point.
    """
    x0, y0, x1, y1 = box
    return {
        View.FRONT: [x0, y0],
        View.TOP: [x0, y1],
        View.RIGHT: [x1, y0],
    }[view]


def place_views(drawing: DrawingSource, page: str = "sheet_page") -> ViewPlacement:
    """Separate one unsplit page into front, top and right views."""
    held = {sheet.name: sheet for sheet in drawing.sheets}
    if page not in held:
        raise ViewSplitError(f"{page} is not a sheet of this drawing")

    reach = {entry.name: extent(entry) for entry in held[page].evidence}
    filed, residual = _split(reach)
    held_entries = {entry.name: entry for entry in held[page].evidence}
    boxes = {
        view: _bounding([reach[name] for name in names])
        for view, names in filed.items()
    }

    sheets = [
        held[page].model_copy(update={"evidence": []}),
        *(
            DrawingSheet(
                name=f"sheet_{view.value}",
                role=view,
                label=None,
                derived_from=page,
                file=None,
                origin=_origin_on(view, boxes[view]),
                evidence=[held_entries[name] for name in names],
                dimensions=[],
            )
            for view, names in filed.items()
        ),
    ]
    return ViewPlacement(
        drawing=DrawingSource(
            sheets=sheets, rationale="Views separated by position on the page."
        ),
        boxes=boxes,
        alignment_error=residual,
    )
