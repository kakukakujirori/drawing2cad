"""The semantics stage's answer: claims about the 3D part, and what supports them.

What a drawing shows lives in `drawings`; this module holds what someone
concluded from it.  The geometry vocabulary is OCC's closed enums, weighed
against a census in `geometry_census.json`, which `test_contract_vocabulary.py`
checks.
"""

import re
from collections.abc import Mapping
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from zeroshot.pipeline.messages.contracts.drawings import (
    DrawingEvidence,
    Parameter,
    require_name,
    require_parameters,
    rows,
)


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


# Which way a feature faces. `other` means an oblique axis.
class Axis(StrEnum):
    X = "x"
    Y = "y"
    Z = "z"
    OTHER = "other"


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

_SEMANTIC_NAME = re.compile(r"^sem_[a-z0-9_]+$")
_GEOMETRY_NAME = re.compile(r"^geo_[a-z0-9_]+$")


class FeatureGeometry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        description=(
            "Stable reference name for this geometry claim, unique within its "
            "feature, beginning geo_ and continuing in lower_snake_case: "
            "geo_bore_cylinder. Later stages cite a parameter as, for example, "
            "sem_main_bore.geo_bore_cylinder.radius. The name is the claim's "
            "identity, not a display label: keep it when revising the claim "
            "and give a new claim a new name."
        ),
    )
    kind: GeometryKind = Field(
        ...,
        description=(
            "The curve or face this feature is made of. Name the real one "
            "rather than something that resembles it: a rounded edge is a "
            "torus or a cylinder, a tapered face a cone. Never a chain of "
            "lines standing in for a curve."
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
            "sits belongs to the modelling plan, working from the evidence.\n"
            "\n"
            "Claiming a shape in 3D:\n"
            f"{rows(_CLAIMED_PARAMETERS)}"
        ),
    )

    @model_validator(mode="after")
    def require_the_parameters_the_kind_is_measured_by(self) -> Self:
        require_name(self.name, "geo_", _GEOMETRY_NAME)
        require_parameters(
            f"kind={self.kind.value!r}",
            _CLAIMED_PARAMETERS[self.kind],
            self.parameters,
        )
        return self


class SemanticFeature(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        description=(
            "Stable reference name for this feature, unique within the "
            "hypothesis, beginning sem_ and continuing in lower_snake_case: "
            "sem_main_bore or sem_top_flange_fillet. This is the feature's "
            "identity, not a separate display label. Keep it when revising the "
            "feature and give a new feature a new name."
        ),
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
    evidence: list[str] = Field(
        ...,
        description=(
            "The ev_ names of the evidence that supports this feature. A "
            "piece of evidence is named rather than restated, so two "
            "features may rest on the same one and neither owns it."
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
    def require_usable_names(self) -> Self:
        """Check the feature's own name, and that it cites nothing twice.

        Not that it cites any evidence at all: a feature also travels as a
        revision carrying only the members that changed, where an empty list
        means "unchanged" rather than "unsupported". `SemanticHypothesis`
        checks that instead, where the evidence itself is.
        """
        require_name(self.name, "sem_", _SEMANTIC_NAME)
        claims = [claim.name for claim in self.geometry]
        duplicated = sorted({name for name in claims if claims.count(name) > 1})
        if duplicated:
            raise ValueError(
                f"feature {self.name} has duplicate geometry names "
                f"{', '.join(duplicated)}; keep each member's name unique"
            )
        cited = sorted(
            {name for name in self.evidence if self.evidence.count(name) > 1}
        )
        if cited:
            raise ValueError(
                f"feature {self.name} cites {', '.join(cited)} more than once"
            )
        return self


class SemanticHypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # NOTE: This must be deleted after uploading DrawingSource on Snapshot.
    evidence: list[DrawingEvidence] = Field(
        ...,
        description=(
            "Every entity read off the drawing, in millimetres on the sheet. "
            "They stand outside the features so that one entry can support "
            "several -- a circle in the top view can be both the bore and the "
            "boss it is concentric with -- and so that a later stage can cite "
            "one as ev_front_circle.center without naming a feature."
        ),
    )
    proposal: list[SemanticFeature] = Field(
        ...,
        description=(
            "Every feature of the part, the base body among them and first. "
            "That order is for reading; a feature is named by its `id`, never "
            "by where it sits in this list."
        ),
    )
    rationale: str = Field(
        ...,
        description=(
            "Why this account of the drawing, and what was resolved against what."
        ),
    )

    @model_validator(mode="after")
    def require_uniquely_named_features_that_cite_evidence(self) -> Self:
        if not self.proposal:
            raise ValueError("proposal must hold at least one feature")
        names = [feature.name for feature in self.proposal]
        if len(set(names)) != len(names):
            raise ValueError(
                "feature names must be unique; keep each sem_ name as one "
                "feature's stable identity"
            )

        read = [entry.name for entry in self.evidence]
        duplicated = sorted({name for name in read if read.count(name) > 1})
        if duplicated:
            raise ValueError(
                f"evidence names {', '.join(duplicated)} more than once; keep "
                "each ev_ name as one entry's stable identity"
            )

        unsupported = [
            feature.name for feature in self.proposal if not feature.evidence
        ]
        if unsupported:
            raise ValueError(
                f"{', '.join(unsupported)} cites no evidence; a feature nothing "
                "in the drawing supports is a guess, and the numbers a later "
                "stage builds from live in the evidence"
            )

        known = set(read)
        for feature in self.proposal:
            missing = [name for name in feature.evidence if name not in known]
            if missing:
                raise ValueError(
                    f"{feature.name} cites {', '.join(missing)}, which this "
                    "hypothesis does not read"
                )
        return self


def render_parameter_values(values: list[float]) -> str:
    """`repr`, not a format: the coder builds from these, so every digit the
    stage transcribed has to survive. `f"{x:g}"` would round 33.0015507591 to
    six figures and quietly move the part."""
    if len(values) == 1:
        return repr(values[0])
    return "[" + " ".join(repr(value) for value in values) + "]"
