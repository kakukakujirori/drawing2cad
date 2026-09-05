"""Least-effort valid instances of the semantics contract, for tests about
something else.

Tests *about* the contract build it explicitly -- see `messages/test_contracts.py`.
"""

from typing import Any

from zeroshot.pipeline.messages.contracts.drawings import (
    _DRAWN_PARAMETERS,
    DrawingEvidence,
    DrawnEntity,
)
from zeroshot.pipeline.messages.contracts.operations import OperationPlan
from zeroshot.pipeline.messages.contracts.parameters import Parameter
from zeroshot.pipeline.messages.contracts.semantics import (
    _GEOMETRY_PARAMETERS,
    FeatureGeometry,
    GeometryKind,
    SemanticFeature,
    SemanticHypothesis,
)

_A_SHEET_POINT = [0.0, 0.0]
_SOME_POINTS = [0.0, 0.0, 1.0, 1.0, 2.0, 0.0]


def _named(given: dict[str, float | list[float]]) -> list[Parameter]:
    return [
        Parameter(
            name=name,  # type: ignore[arg-type]
            values=value if isinstance(value, list) else [value],
        )
        for name, value in given.items()
    ]


_STAND_IN: dict[str, list[float]] = {
    "start": _A_SHEET_POINT,
    "end": [1.0, 0.0],
    "center": _A_SHEET_POINT,
    "major_axis": [1.0, 0.0],
    "control_points": _SOME_POINTS,
    "vertices": _SOME_POINTS,
    # one per control point, plus the degree, plus one
    "knots": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
}

# Endpoints have to lie on the curve they bound, so these two carry a whole row.
_ON_CURVE: dict[DrawnEntity, dict[str, float | list[float]]] = {
    DrawnEntity.ARC: {"radius": 5.0, "start": [5.0, 0.0], "end": [0.0, 5.0]},
    DrawnEntity.ELLIPSE: {
        "major_axis": [5.0, 0.0],
        "minor_radius": 2.0,
        "start": [5.0, 0.0],
        "end": [0.0, 2.0],
    },
}


def evidence(
    entity: str = "line", *, name: str | None = None, **values: float | list[float]
) -> DrawingEvidence:
    """A reading of `entity` whose parameters are its own row."""
    drawn = DrawnEntity(entity)
    stand_in = _STAND_IN | _ON_CURVE.get(drawn, {})
    given = {
        name: values.pop(name, stand_in.get(name, 5.0))
        for name in _DRAWN_PARAMETERS[drawn]
    }
    given |= values
    return DrawingEvidence(
        name=name or f"ev_{entity}",
        entity=entity,  # type: ignore[arg-type]
        edge_style="visible",
        source=[],
        parameters=_named(given),
    )


def geometry(
    kind: str = "torus",
    axis: str | None = "z",
    *,
    name: str | None = None,
    **sizes: float | list[float],
) -> FeatureGeometry:
    """A claim of `kind` measured by its own row, overridden by name."""
    given: dict[str, float | list[float]] = {
        name: sizes.pop(name, 5.0) for name in _GEOMETRY_PARAMETERS[GeometryKind(kind)]
    }
    given |= sizes
    return FeatureGeometry(
        name=name or f"geo_{kind}",
        kind=kind,  # type: ignore[arg-type]
        axis=axis,  # type: ignore[arg-type]
        parameters=_named(given),
    )


def feature(
    identifier: int | str, description: str, **overrides: object
) -> SemanticFeature:
    fields: dict[str, object] = {
        "name": (
            f"sem_feature_{identifier}" if isinstance(identifier, int) else identifier
        ),
        "description": description,
        "geometry": [],
        "evidence": ["ev_line"],
        "open_question": None,
        **overrides,
    }
    return SemanticFeature(**fields)  # type: ignore[arg-type]


def hypothesis(*descriptions: str, **overrides: object) -> SemanticHypothesis:
    """One feature per description, numbered from 1."""
    fields: dict[str, object] = {
        "proposal": [
            feature(index, description)
            for index, description in enumerate(descriptions, start=1)
        ],
        "rationale": "the views agree",
        **overrides,
    }
    return SemanticHypothesis(**fields)  # type: ignore[arg-type]


def replacing(artifact: SemanticHypothesis | OperationPlan) -> dict[str, Any]:
    """The submission fields that build `artifact` from nothing, as round 0 does."""
    return {
        "edits": list(artifact.proposal),
        "deleted": [],
        "rationale": artifact.rationale,
    }


def unchanged() -> dict[str, Any]:
    """The submission fields that leave the preceding round's artifact as it is."""
    return {"edits": [], "deleted": [], "rationale": None}
