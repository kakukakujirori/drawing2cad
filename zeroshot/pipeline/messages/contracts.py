"""The semantics stage's answer, as geometry rather than prose.

No vocabulary here is invented: each is taken from the format that produces it
-- OCC's closed geometry enums, ezdxf's linetype table -- and weighed against a
census in `geometry_census.json`, which `test_contract_vocabulary.py` checks.
"""

from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class View(StrEnum):
    FRONT = "front"
    TOP = "top"
    RIGHT = "right"


VIEW_FRAME: Mapping[View, tuple[str, str, str]] = {
    # view: (what points right on the sheet, what points up, what points at you)
    View.FRONT: ("+x", "+y", "+z"),
    View.TOP: ("+x", "-z", "+y"),
    View.RIGHT: ("-z", "+y", "+x"),
}


def view_frame_sentence() -> str:
    """How a view's sheet coordinates sit in the model, as one sentence.

    Every stage after semantics reads numbers tagged with a view, so every one
    of them needs this. Rendering it from `VIEW_FRAME` rather than writing it
    into each stage's guidelines is what stops a stage from carrying an old
    frame: a stage that guesses is wrong in silence, and so is a stage whose
    copy fell behind.
    """
    return "; ".join(
        f"{view.value.capitalize()} is right={right}, up={up}"
        for view, (right, up, _) in VIEW_FRAME.items()
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

    Unrecognised names give `OTHER` rather than a guess. `BYLAYER` and
    `BYBLOCK` are not linetypes but instructions to resolve one, so they land
    there too: resolve them through the layer before asking.
    """
    base = linetype.strip().upper().removesuffix("X2").removesuffix("2")
    return _LINETYPE_MEANINGS.get(base, EdgeStyle.OTHER)


# OCC's geometry set: `GeomAbs_SurfaceType` and `GeomAbs_CurveType`
# `arc` is the single addition: OCC stores one as a bounded Circle, but this
# contract is what the model reasons in, and an arc and a full circle are
# different things to read off a drawing.
class GeometryKind(StrEnum):
    # edges
    ARC = "arc"
    BSPLINE_CURVE = "bspline_curve"
    CIRCLE = "circle"
    ELLIPSE = "ellipse"
    LINE = "line"
    # faces
    BSPLINE_SURFACE = "bspline_surface"
    CONE = "cone"
    CYLINDER = "cylinder"
    PLANE = "plane"
    SPHERE = "sphere"
    TORUS = "torus"


# The OCC members deliberately left out, so that a new one in the kernel's
# enums has to be a decision made here rather than an omission nobody notices.
_EXCLUDED_GEOMETRY = frozenset(
    {
        # Produced by an operation rather than described.
        "SurfaceOfExtrusion",
        "SurfaceOfRevolution",
        "OffsetSurface",
        "OffsetCurve",
        # No occurrence in the ABC dataset.
        # Bezier is a special case of BSpline and OCC reports it as the latter.
        "BezierSurface",
        "BezierCurve",
        "Hyperbola",
        "Parabola",
        # OCC's bucket for a shape it could not classify. Not a shape.
        "OtherSurface",
        "OtherCurve",
    }
)


class ClaimSource(StrEnum):
    EXACT = "exact"
    DERIVED = "derived"
    ASSUMED = "assumed"


# Which way a feature faces. `other` means an oblique axis.
class Axis(StrEnum):
    X = "x"
    Y = "y"
    Z = "z"
    OTHER = "other"


_DRAWN_PARAMETERS: Mapping[DrawnEntity, tuple[str, ...]] = {
    DrawnEntity.LINE: ("start", "end"),
    DrawnEntity.ARC: ("center", "radius", "start", "end"),
    DrawnEntity.CIRCLE: ("center", "radius"),
    DrawnEntity.ELLIPSE: ("center", "major_axis", "major_radius", "minor_radius"),
    DrawnEntity.SPLINE: ("control_points", "degree", "knots"),
    DrawnEntity.POLYLINE: ("vertices",),
}

_CLAIMED_PARAMETERS: Mapping[GeometryKind, tuple[str, ...]] = {
    # A line has no size
    GeometryKind.LINE: (),
    GeometryKind.ARC: ("radius",),
    GeometryKind.CIRCLE: ("radius",),
    GeometryKind.ELLIPSE: ("major_radius", "minor_radius"),
    GeometryKind.BSPLINE_CURVE: ("degree",),
    # A plane has no size
    GeometryKind.PLANE: (),
    GeometryKind.CYLINDER: ("radius", "height"),
    GeometryKind.CONE: ("base_radius", "top_radius", "height"),
    GeometryKind.SPHERE: ("radius",),
    GeometryKind.TORUS: ("major_radius", "tube_radius"),
    GeometryKind.BSPLINE_SURFACE: ("degree",),
}

# How many numbers each name takes
_ARITY: Mapping[str, int] = {
    # a point or a direction on the sheet
    "start": 2,
    "end": 2,
    "center": 2,
    "major_axis": 2,
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

# Of the names that take a list, which hold sheet points -- x, y, x, y, ... --
# rather than plain numbers. A knot vector is neither points nor a fixed count:
# it runs one entry per control point plus the degree plus one.
_POINT_LISTS = frozenset({"control_points", "vertices"})

_DIRECTIONS = frozenset({"major_axis"})

ParameterName = StrEnum("ParameterName", {name.upper(): name for name in _ARITY})


def _rows(table: Mapping[Any, tuple[str, ...]]) -> str:
    """A table laid out for the field description that carries it.

    Rendered rather than written out, so the description cannot come to say
    something the validator does not enforce.
    """
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
            "describing takes, and none that it does not.\n"
            "\n"
            "Reading an entity off a view:\n"
            f"{_rows(_DRAWN_PARAMETERS)}\n"
            "\n"
            "Claiming a shape in 3D:\n"
            f"{_rows(_CLAIMED_PARAMETERS)}"
        ),
    )
    values: list[float] = Field(
        ...,
        description=(
            "The number, or the numbers, this parameter is made of: one for a "
            "radius or a degree, two for a point or a direction, a flat "
            "x, y, x, y, ... for a list of points, and the whole vector for a "
            "spline's knots. Use the drawing's own figures and do not convert "
            "them."
        ),
    )


def _checked(
    subject: str,
    expected: tuple[str, ...],
    parameters: list[Parameter],
) -> None:
    """Hold `parameters` to exactly `expected`, naming the set when they differ.

    Shared by the reading and the claim because the two differ only in which
    table they are held to; the message is what the model reads back through
    `middleware/model_retry.py`, so it says the whole set rather than the
    first thing wrong with it.
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


class ViewEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    view: View = Field(
        ...,
        description=(
            "Which view this reading comes from. Views are separated by their "
            "position on the sheet, not by layer."
        ),
    )
    entity: DrawnEntity = Field(
        ...,
        description=(
            "The 2D entity type as the drawing literally shows it. Evidence, "
            "not conclusion: it may differ from the 3D geometry you claim, and "
            "an ellipse usually does -- in an orthographic view a circle seen "
            "obliquely draws as one."
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
    parameters: list[Parameter] = Field(
        ...,
        description=(
            "The entity's own numbers, copied from the drawing unchanged. This "
            "is the one place exact geometry belongs, because it is read "
            "rather than inferred."
        ),
    )

    @model_validator(mode="after")
    def require_the_parameters_the_entity_states(self) -> Self:
        _checked(
            f"entity={self.entity.value!r}",
            _DRAWN_PARAMETERS[self.entity],
            self.parameters,
        )
        return self


class FeatureGeometry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: GeometryKind = Field(
        ...,
        description=(
            "The curve or face this feature is made of. Name the real one "
            "rather than something that resembles it: a rounded edge is a "
            "torus or a cylinder, a tapered face a cone. Never a chain of "
            "lines standing in for a curve."
        ),
    )
    source: ClaimSource = Field(
        ...,
        description=(
            "'exact' for a size read from the drawing's own curve definition; "
            "'derived' when computed from other geometry in the drawing; "
            "'assumed' when chosen to close the part rather than supported "
            "by it."
        ),
    )
    axis: Axis | None = Field(
        ...,
        description=(
            "Which global axis the geometry turns about, runs along, or "
            "faces, or "
            "'other' when the drawing shows it oblique. Null for a kind with "
            "no axis. This is a direction, not a position: say which way it "
            "points, not where it sits."
        ),
    )
    parameters: list[Parameter] = Field(
        ...,
        description=(
            "Exactly the sizes the kind is measured by, and no others. Give "
            "them: a kind on its own is a label, and a label is what the coder "
            "approximates. Do not give a position here -- where the feature "
            "sits belongs to the modelling plan, working from the evidence."
        ),
    )

    @model_validator(mode="after")
    def require_the_parameters_the_kind_is_measured_by(self) -> Self:
        _checked(
            f"kind={self.kind.value!r}",
            _CLAIMED_PARAMETERS[self.kind],
            self.parameters,
        )
        return self


class SemanticFeature(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int = Field(
        ...,
        description=(
            "Identifier for this feature, unique within the hypothesis and 1 "
            "or greater. Later stages cite it as sem<id>. Keep a feature's id "
            "when you revise it; renumber only when you merge two lists."
        ),
    )
    name: str = Field(
        ..., description="Short label, e.g. 'main bore' or 'top flange fillet'."
    )
    description: str = Field(
        ...,
        description=(
            "What the feature is and how it sits on the part, in one or two "
            "sentences. A gloss on `geometry`, never a substitute for it: no "
            "dimension belongs here that belongs there."
        ),
    )
    geometry: list[FeatureGeometry] = Field(
        ...,
        description=(
            "The faces and curves this feature must be built from, where the "
            "type matters. A fillet is one torus, a flat chamfer one plane, a "
            "bored boss a cylinder. Name a plane where you have determined a "
            "face is flat, so that saying nothing means you found nothing "
            "rather than that you did not look. Straight edges need no entry: "
            "the evidence already carries them."
        ),
    )
    evidence: list[ViewEvidence] = Field(
        ...,
        description=(
            "What was seen, and where, that supports the feature. Record the "
            "2D entity as drawn even when the 3D claim differs from it."
        ),
    )
    open_question: str | None = Field(
        ...,
        description=(
            "What the input leaves undetermined about this feature, or null "
            "when nothing is."
        ),
    )

    @model_validator(mode="after")
    def require_geometry(self) -> Self:
        if self.id < 1:
            raise ValueError("id must be 1 or greater")
        if not self.evidence:
            raise ValueError(
                f"feature {self.id} cites no evidence; a feature nothing in "
                "the drawing supports is a guess, and the exact numbers a "
                "later stage builds from live in the evidence"
            )
        return self


class SemanticHypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal: list[SemanticFeature] = Field(
        ...,
        description=(
            "Every feature of the part, the base body first. That order is "
            "for reading; a feature is named by its `id`, never by where it "
            "sits in this list."
        ),
    )
    rationale: str = Field(
        ...,
        description=(
            "Why this reading of the drawing, and what was resolved against what."
        ),
    )

    @model_validator(mode="after")
    def require_uniquely_numbered_features(self) -> Self:
        if not self.proposal:
            raise ValueError("proposal must hold at least one feature")
        ids = [feature.id for feature in self.proposal]
        if len(set(ids)) != len(ids):
            raise ValueError(
                "feature ids must be unique; renumber the merged list from 1"
            )
        return self


def _numbers(values: list[float]) -> str:
    """`repr`, not a format: the coder builds from these, so every digit the
    stage transcribed has to survive. `f"{x:g}"` would round 33.0015507591 to
    six figures and quietly move the part."""
    if len(values) == 1:
        return repr(values[0])
    return "[" + " ".join(repr(value) for value in values) + "]"


def render_hypothesis(hypothesis: "SemanticHypothesis") -> str:
    """The hypothesis as text for the stages that read it.

    `model_dump_json(indent=2)` spends about half its length on braces, key
    names and indentation -- 50k characters for seven features, of which 30k
    was evidence and 14k was punctuation. The same content laid out one reading
    to a line is under a third of that, with every number still verbatim.

    Size is the smaller reason. The shape is the larger one: a feature's name,
    its description and what it claims come first, and the readings that
    support it hang underneath, so what a planner needs to orient itself is not
    buried in the transcription it needs afterwards.
    """
    lines: list[str] = []
    for feature in hypothesis.proposal:
        lines.append(f"sem{feature.id} {feature.name}")
        lines.append(f"  {feature.description}")
        claims = []
        for claim in feature.geometry:
            sizes = " ".join(
                f"{parameter.name.value}={_numbers(parameter.values)}"
                for parameter in claim.parameters
            )
            head = f"{claim.kind.value}({sizes})" if sizes else claim.kind.value
            # `axis` is null for a kind that has none, so the phrase is absent
            # rather than rendered as the word for "no answer".
            axis = f" axis {claim.axis.value}" if claim.axis is not None else ""
            claims.append(f"{head}{axis} {claim.source.value}")
        lines.append("  geometry: " + ("; ".join(claims) or "(none stated)"))
        if feature.open_question:
            lines.append(f"  open question: {feature.open_question}")
        for reading in feature.evidence:
            given = " ".join(
                f"{parameter.name.value}={_numbers(parameter.values)}"
                for parameter in reading.parameters
            )
            lines.append(
                f"    {reading.view.value} {reading.entity.value} "
                f"{reading.edge_style.value} {given}"
            )
    lines.append(f"rationale: {hypothesis.rationale}")
    return "\n".join(lines)
