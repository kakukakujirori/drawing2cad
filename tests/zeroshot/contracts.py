"""Least-effort valid instances of the semantics contract, for tests about
something else.

Tests *about* the contract build it explicitly -- see `messages/test_contracts.py`.
"""

from zeroshot.pipeline.messages.contracts.semantics import (
    _CLAIMED_PARAMETERS,
    _DRAWN_PARAMETERS,
    DrawnEntity,
    FeatureGeometry,
    GeometryKind,
    Parameter,
    SemanticFeature,
    SemanticHypothesis,
    ViewEvidence,
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


def evidence(
    entity: str = "line", *, name: str | None = None, **values: float | list[float]
) -> ViewEvidence:
    """A reading of `entity` whose parameters are its own row."""
    given = {
        name: values.pop(name, _STAND_IN.get(name, 5.0))
        for name in _DRAWN_PARAMETERS[DrawnEntity(entity)]
    }
    given |= values
    return ViewEvidence(
        name=name or f"ev_{entity}",
        view="front",
        entity=entity,  # type: ignore[arg-type]
        edge_style="visible",
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
        name: sizes.pop(name, 5.0) for name in _CLAIMED_PARAMETERS[GeometryKind(kind)]
    }
    given |= sizes
    return FeatureGeometry(
        name=name or f"geo_{kind}",
        kind=kind,  # type: ignore[arg-type]
        source="exact",
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
        "evidence": [evidence()],
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
