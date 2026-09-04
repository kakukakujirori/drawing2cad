"""What a drawing literally shows, and how a run's drawings are handed in.

Everything here is asserted by the input; `semantics` holds the 3D claims made
by reading it. `Parameter` lives here because both sides are measured with it.
"""

import re
from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class View(StrEnum):
    """Which projection a sheet is."""

    # Orthographic: a coordinate read off one of these can be lifted into
    # the model, because `VIEW_FRAME` says how its axes sit.
    FRONT = "front"
    BACK = "back"
    TOP = "top"
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"
    # Drawn to their own frame or their own scale, so nothing lifts.
    SECTION = "section"
    DETAIL = "detail"
    ISOMETRIC = "isometric"
    PERSPECTIVE = "perspective"
    UNKNOWN = "unknown"


# Fixed by unfolding the glass box, not chosen here: a hinge along a horizontal
# edge keeps left and right, a vertical one reverses them, and right-handedness
# settles what points up. First- and third-angle projection differ in placement
# on the sheet, not in these axes. `back` is sometimes hinged from the top view
# instead and comes out inverted; that development is rare and obvious on sight.
VIEW_FRAME: Mapping[View, tuple[str, str, str]] = {
    # view: (what points right on the sheet, what points up, what points at you)
    View.FRONT: ("+x", "+y", "+z"),
    View.BACK: ("-x", "+y", "-z"),
    View.TOP: ("+x", "-z", "+y"),
    View.BOTTOM: ("+x", "+z", "-y"),
    View.RIGHT: ("-z", "+y", "+x"),
    View.LEFT: ("+z", "+y", "-x"),
}
ORTHOGRAPHIC_VIEWS: tuple[View, ...] = tuple(VIEW_FRAME.keys())


def view_frame_sentence(views: Iterable[View] | None = None) -> str:
    """How the given views' sheet coordinates sit in the model, as one sentence.

    Rendered from `VIEW_FRAME` so no stage can carry a frame that fell behind.
    `views` narrows it to the ones a run holds; the fallback names all six.
    """
    wanted = ORTHOGRAPHIC_VIEWS if views is None else tuple(views)
    return "; ".join(
        f"{view.value.capitalize()} is right={VIEW_FRAME[view][0]}, "
        f"up={VIEW_FRAME[view][1]}"
        for view in ORTHOGRAPHIC_VIEWS
        if view in wanted
    )


class DrawnEntity(StrEnum):
    LINE = "line"
    ARC = "arc"
    CIRCLE = "circle"
    ELLIPSE = "ellipse"
    SPLINE = "spline"
    POLYLINE = "polyline"


class EdgeStyle(StrEnum):
    VISIBLE = "visible"
    HIDDEN = "hidden"
    CENTERLINE = "centerline"
    PHANTOM = "phantom"
    OTHER = "other"


# Every linetype `ezdxf.tools.standards.linetypes()` defines, grouped by what
# it means, plus `HIDDEN`, which AutoCAD ships and ezdxf does not.
_LINETYPE_MEANINGS: Mapping[str, EdgeStyle] = {
    "CONTINUOUS": EdgeStyle.VISIBLE,
    "DASHED": EdgeStyle.HIDDEN,
    "DOT": EdgeStyle.HIDDEN,
    "HIDDEN": EdgeStyle.HIDDEN,
    "CENTER": EdgeStyle.CENTERLINE,
    "DASHDOT": EdgeStyle.CENTERLINE,
    "PHANTOM": EdgeStyle.PHANTOM,
    "DIVIDE": EdgeStyle.PHANTOM,
}


def edge_style_for_linetype(linetype: str) -> EdgeStyle:
    """What a DXF linetype name means as drawing linework.

    `BYLAYER` and `BYBLOCK` fall to `OTHER` with the unrecognised names:
    resolve them through the layer before asking.
    """
    base = linetype.strip().upper().removesuffix("X2").removesuffix("2")
    return _LINETYPE_MEANINGS.get(base, EdgeStyle.OTHER)


_DRAWN_PARAMETERS: Mapping[DrawnEntity, tuple[str, ...]] = {
    DrawnEntity.LINE: ("start", "end"),
    DrawnEntity.ARC: ("center", "radius", "start_angle", "end_angle"),
    DrawnEntity.CIRCLE: ("center", "radius"),
    DrawnEntity.ELLIPSE: ("center", "major_axis", "major_radius", "minor_radius"),
    DrawnEntity.SPLINE: ("control_points", "degree", "knots"),
    DrawnEntity.POLYLINE: ("vertices",),
}

# How many numbers each name takes, spanning evidence and 3D claims alike.
# Which names a subject may use is decided by the table it is held to.
_ARITY: Mapping[str, int] = {
    # a point or a direction on the sheet
    "start": 2,
    "end": 2,
    "center": 2,
    "major_axis": 2,
    # degrees about the centre, because that is what a DXF arc stores; asking
    # for the point instead got the angle back, one number where two were due.
    "start_angle": 1,
    "end_angle": 1,
    # a size
    "radius": 1,
    "major_radius": 1,
    "minor_radius": 1,
    "tube_radius": 1,
    "base_radius": 1,
    "top_radius": 1,
    "height": 1,
    "degree": 1,
    # a list, whose length the shape below decides
    "control_points": 0,
    "vertices": 0,
    "knots": 0,
}

# Which of the list-valued names hold sheet points -- x, y, x, y, ... A knot
# vector is neither: it runs one entry per control point plus the degree plus 1.
_POINT_LISTS = frozenset({"control_points", "vertices"})

_DIRECTIONS = frozenset({"major_axis"})

ParameterName = StrEnum("ParameterName", {name.upper(): name for name in _ARITY})

_EVIDENCE_NAME = re.compile(r"^ev_[a-z0-9_]+$")
_DIMENSION_NAME = re.compile(r"^dim_[a-z0-9_]+$")

DRAWING_SUFFIXES = frozenset({".dxf", ".png", ".jpg", ".jpeg"})


def require_name(name: str, prefix: str, pattern: re.Pattern[str]) -> None:
    if pattern.fullmatch(name):
        return
    stray = dict.fromkeys(re.findall(r"[^a-z0-9_]", name.removeprefix(prefix)))
    raise ValueError(
        f"{name!r} is not a usable {prefix.removesuffix('_') or 'sheet'} name. "
        f"Begin with {prefix} and carry on in lower_snake_case."
        + (f" Remove {', '.join(map(repr, stray))}." if stray else "")
    )


def rows(table: Mapping[Any, tuple[str, ...]]) -> str:
    """Lay a table out for the field description that carries it, so the
    description cannot come to say something the validator does not enforce."""
    return "\n".join(
        f"  {key.value} takes {', '.join(row) or 'nothing'}"
        for key, row in table.items()
    )


class Parameter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: ParameterName = Field(
        ...,
        description=(
            "Which parameter this is. Give every one the thing you are "
            "describing takes, and none that it does not; that thing's own "
            "field says which set it takes."
        ),
    )
    values: list[float] = Field(
        ...,
        description=(
            "The number, or the numbers, this parameter is made of: one for a "
            "radius or a degree, two for a point or a direction, a flat "
            "x, y, x, y, ... for a list of points, and the whole vector for a "
            "spline's knots."
        ),
    )


def require_parameters(
    subject: str,
    expected: tuple[str, ...],
    parameters: Sequence[Parameter],
) -> None:
    """Hold `parameters` to exactly `expected`, and each to its own arity.

    The message names the whole set rather than the first thing wrong with it,
    because the model reads it back through `middleware/model_retry.py`.
    """
    given = {parameter.name.value: parameter.values for parameter in parameters}
    if len(given) != len(parameters):
        raise ValueError("a parameter is given twice")

    missing = [name for name in expected if name not in given]
    unknown = [name for name in given if name not in expected]
    if missing or unknown:
        raise ValueError(
            f"{subject} takes {list(expected) or 'no parameters'}"
            + (f"; missing {missing}" if missing else "")
            + (f"; unknown {unknown}" if unknown else "")
        )

    for name, values in given.items():
        arity = _ARITY[name]
        if arity and len(values) != arity:
            raise ValueError(f"`{name}` takes {arity} number(s), got {len(values)}")
        if not arity and name in _POINT_LISTS and (not values or len(values) % 2):
            raise ValueError(f"`{name}` is a flat list of x, y pairs")
        if not arity and name not in _POINT_LISTS and not values:
            raise ValueError(f"`{name}` is a list of numbers and cannot be empty")
        if name in _DIRECTIONS and not any(values):
            raise ValueError(f"`{name}` must be a non-zero direction")


# How well a number is supported: the input says it, you worked it out from
# what the input says, or nothing supports it. How a derivation was done is a
# method rather than a source, so it does not split the middle answer.
# A docstring here would reach the provider as schema prose, which
# `test_state.py` refuses.
class ClaimSource(StrEnum):
    GIVEN = "given"
    DERIVED = "derived"
    ASSUMED = "assumed"


# One entity as it was read off a view, in millimetres on the sheet whatever
# format the drawing arrived in. It lives in the hypothesis rather than in a
# `DrawingSheet` because nothing populates a sheet yet.
# A docstring would reach the provider as schema prose, which `test_state.py`
# refuses; the field descriptions are what the model is meant to read.
class DrawingEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        description=(
            "Stable reference name for this entry, unique within the "
            "hypothesis, beginning ev_ and continuing in lower_snake_case: "
            "ev_front_outer_circle. Later stages cite a parameter as, for "
            "example, ev_front_outer_circle.center. The name is the entry's "
            "identity, not a display label: keep it when revising the entry "
            "and give a new entry a new name."
        ),
    )
    view: View = Field(
        ...,
        description="Which view this entry comes from.",
    )
    entity: DrawnEntity = Field(
        ...,
        description=(
            "The 2D entity type as the drawing literally shows it. Evidence, "
            "not conclusion: it may differ from the 3D geometry claimed of it, "
            "and an ellipse usually does -- in an orthographic view a circle "
            "seen obliquely draws as one."
        ),
    )
    edge_style: EdgeStyle = Field(
        ...,
        description=(
            "'visible' for continuous linework, 'hidden' for an edge behind "
            "material. Hidden edges are how depth is read: whether a hole is "
            "through or blind, where a pocket stops."
        ),
    )
    source: ClaimSource = Field(
        ...,
        description=(
            "How the numbers below were obtained. 'given' when the input "
            "states them outright -- a printed dimension, or a curve "
            "definition in a vector sheet; 'derived' when you worked them out "
            "from what it states, whether by arithmetic or by measuring the "
            "linework through the sheet's scale; 'assumed' when nothing in the "
            "drawing supports them."
        ),
    )
    parameters: list[Parameter] = Field(
        ...,
        description=(
            "The entity's own numbers, in millimetres on the sheet. Give every "
            "one the entity takes or give none at all: an entry cited for "
            "what it proves rather than what it measures -- a hidden line pair "
            "showing a bore runs through -- takes no parameters, and a "
            "half-transcribed entity is not evidence. Do not report pixels."
            "\n\n"
            "Reading an entity off a view:\n"
            f"{rows(_DRAWN_PARAMETERS)}"
        ),
    )

    @model_validator(mode="after")
    def require_the_parameters_the_entity_states(self) -> Self:
        require_name(self.name, "ev_", _EVIDENCE_NAME)
        if not self.parameters:
            # 'given' says the input handed the numbers over, so an empty one
            # contradicts itself. Name the set: for a one-parameter entity
            # that is the only way a half-transcribed entry shows up.
            if self.source is ClaimSource.GIVEN:
                raise ValueError(
                    f"{self.name} is sourced 'given' but states none of "
                    f"{list(_DRAWN_PARAMETERS[self.entity])}; give them, or say "
                    "how the numbers were really obtained"
                )
            return self
        require_parameters(
            f"entity={self.entity.value!r}",
            _DRAWN_PARAMETERS[self.entity],
            self.parameters,
        )
        return self


class DimensionKind(StrEnum):
    LINEAR = "linear"
    DIAMETER = "diameter"
    RADIUS = "radius"
    ANGULAR = "angular"


# One figure printed on the drawing. Separate from `DrawingEvidence` because
# its check is separate: linework is compared geometrically, a printed figure
# is read. Folding the figure into the linework it measures would make
# `nominal = scale * distance` a tautology, and that identity is the only
# signal that tells a wrong scale from a wrong target from a misreading.
class Dimension(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        description=(
            "Stable reference name, unique within the drawing, beginning dim_ "
            "and continuing in lower_snake_case: dim_bore_diameter."
        ),
    )
    kind: DimensionKind = Field(..., description="What sort of figure this is.")
    text: str = Field(
        ...,
        description="The annotation exactly as printed, symbols and all.",
    )
    nominal: float = Field(
        ...,
        description=(
            "The size itself, as a number, with no multiplier applied. "
            "Degrees when `kind` is angular, millimetres otherwise."
        ),
    )
    quantity: int = Field(
        1,
        description=(
            "How many features the figure covers: the 4 of '4X 12 THRU'. One "
            "when the annotation names no count."
        ),
        ge=1,
    )
    note: str | None = Field(
        None,
        description=(
            "What the annotation says beyond the size -- THRU, 15 DEEP, CBORE, "
            "M12x1.75-6H, TYP. -- or null when it says nothing else. Anything "
            "`kind` or `quantity` can carry belongs there instead."
        ),
    )
    targets: list[str] = Field(
        default_factory=list,
        description=(
            "The ev_ names this figure measures: one circle for a diameter, "
            "the two edges a length runs between. Empty when the annotation "
            "could not be tied to any linework."
        ),
    )

    @model_validator(mode="after")
    def require_a_usable_name(self) -> Self:
        require_name(self.name, "dim_", _DIMENSION_NAME)
        return self


class DrawingSheet(BaseModel):
    """One view of the part, as the file it arrived in.

    What it draws and at what scale are an analyser's answers, and nothing
    answers them yet: a run's evidence lives in the hypothesis.
    """

    model_config = ConfigDict(extra="forbid")

    role: View
    label: str | None = Field(
        None,
        description=(
            "What tells this sheet apart from another with the same role: the "
            "caption printed on it -- 'SECTION A-A' -- or, for a sheet that "
            "was rendered rather than drawn, which rendering it is."
        ),
    )
    detail_of: View | None = Field(
        None,
        description=(
            "For a section or a detail, the view carrying the cutting line or "
            "the ringed region it was taken from."
        ),
    )
    file: Path

    @model_validator(mode="after")
    def require_a_drawing_this_pipeline_can_open(self) -> Self:
        if self.file.suffix.lower() not in DRAWING_SUFFIXES:
            raise ValueError(
                f"{self.file} is not a drawing; expected one of "
                f"{sorted(DRAWING_SUFFIXES)}"
            )
        if self.detail_of is not None and self.detail_of is self.role:
            raise ValueError(f"a {self.role.value} sheet cannot be taken from itself")
        return self

    @property
    def name(self) -> str:
        """What this sheet is announced and staged under."""
        return self.label or self.role.value

    @property
    def frame(self) -> tuple[str, str, str] | None:
        return VIEW_FRAME.get(self.role)


class DrawingSource(BaseModel):
    """The drawings a run was given, and whatever has been made of them.

    A sheet nobody has separated into views is one with the `unknown` role,
    so a split drawing and an unsplit one are the same list; `active_sheets()`
    is what decides which of them a reasoning stage reads.
    """

    model_config = ConfigDict(extra="forbid")

    sheets: list[DrawingSheet] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_a_drawing_whose_views_are_each_given_once(self) -> Self:
        if not self.sheets:
            raise ValueError("a drawing source needs at least one sheet")

        # Only the orthographic roles are unique. Two isometrics, two sections,
        # or two unsplit pages are a drawing convention rather than a mistake.
        roles = [sheet.role for sheet in self.sheets]
        for view in ORTHOGRAPHIC_VIEWS:
            if roles.count(view) > 1:
                raise ValueError(f"more than one {view.value} view")

        # A sheet is announced and staged under its label, or its role when it
        # has none, so two sheets answering to one name would leave one of the
        # pair unreachable and one file written over the other.
        names = [sheet.name for sheet in self.sheets]
        duplicated = sorted({name for name in names if names.count(name) > 1})
        if duplicated:
            raise ValueError(f"more than one sheet named {', '.join(duplicated)}")

        for sheet in self.sheets:
            if sheet.detail_of is not None and sheet.detail_of not in roles:
                raise ValueError(
                    f"a {sheet.role.value} sheet is taken from "
                    f"{sheet.detail_of.value}, which this drawing does not hold"
                )
        return self

    def active_sheets(self) -> list[DrawingSheet]:
        """The views a reasoning stage works with."""
        named = [sheet for sheet in self.sheets if sheet.role is not View.UNKNOWN]
        return named or list(self.sheets)

    def orthographic(self) -> list[DrawingSheet]:
        """The sheets a cross-view check may compare"""
        return [sheet for sheet in self.sheets if sheet.role in VIEW_FRAME]

    def views(self) -> tuple[View, ...]:
        return tuple(sheet.role for sheet in self.sheets)

    def paths(self) -> list[Path]:
        """Every file this drawing is made of."""
        return [sheet.file for sheet in self.sheets]
