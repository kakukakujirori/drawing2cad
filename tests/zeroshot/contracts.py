"""Minimal valid pipeline artifacts, for tests about something else.

Tests *about* the contract build it explicitly -- see `messages/test_contracts.py`.
"""

from zeroshot.pipeline.stages._base.parameters import Parameter
from zeroshot.pipeline.stages.drawings.contracts import (
    _DRAWN_PARAMETERS,
    DrawingEvidence,
    DrawingSheet,
    DrawingSource,
    DrawnEntity,
)
from zeroshot.pipeline.stages.interpretation.contracts import (
    DrawingInterpretation,
    DrawingView,
    Region,
    View,
)
from zeroshot.pipeline.stages.interpretation.contracts import (
    SemanticFeature as InterpretedFeature,
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


def sheet(role: str = "front", **overrides: object) -> DrawingSheet:
    """A sheet named after the view it shows, carrying one reading.

    Its file is relative, so it reads the same from either side of a sandbox.
    """
    fields: dict[str, object] = {
        "name": f"sheet_{role}",
        "role": View(role),
        "crop_of": None,
        "scale": 1.0,
        "file": f"inputs/{role}.dxf",
        "evidence": [evidence(name=f"ev_{role}_line")],
        "dimensions": [],
        **overrides,
    }
    return DrawingSheet(**fields)  # type: ignore[arg-type]


def drawing(*roles: str, **overrides: object) -> DrawingSource:
    """One sheet per view, front alone by default."""
    fields: dict[str, object] = {
        "sheets": [sheet(role) for role in roles or ("front",)],
        **overrides,
    }
    return DrawingSource(**fields)  # type: ignore[arg-type]


def interpreted_feature(
    identifier: int | str, description: str, **overrides: object
) -> InterpretedFeature:
    return InterpretedFeature.model_validate(
        {
            "name": f"sem_feature_{identifier}"
            if isinstance(identifier, int)
            else identifier,
            "description": description,
            "parameters": {},
            "evidence": [Region(view="view_front", box_px=(0, 0, 10, 10))],
            "dimension_refs": [],
            **overrides,
        }
    )


def interpretation(*descriptions: str, **overrides: object) -> DrawingInterpretation:
    """Minimal adopted features with a real view reference, independent of input adapters."""
    return DrawingInterpretation.model_validate(
        {
            "datum": "Origin at the base; x right, y back, z up.",
            "views": [
                DrawingView(
                    name="view_front",
                    role="front",
                    file="inputs/front.png",
                    region=Region(view="view_front", box_px=(0, 0, 10, 10)),
                    dimensions=[],
                )
            ],
            "features": [
                interpreted_feature(index, description)
                for index, description in enumerate(descriptions, start=1)
            ],
            "questions": [],
            **overrides,
        }
    )
