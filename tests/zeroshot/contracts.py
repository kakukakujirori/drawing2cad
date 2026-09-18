"""Minimal valid pipeline artifacts, for tests about something else.

Tests *about* the contract build it explicitly -- see
`messages/test_interpretation_contracts.py`.
"""

from zeroshot.pipeline.stages.interpretation.contracts import (
    DrawingInterpretation,
    DrawingView,
    Region,
    View,
)
from zeroshot.pipeline.stages.interpretation.contracts import (
    SemanticFeature as InterpretedFeature,
)


def view(role: str = "front", **overrides: object) -> DrawingView:
    """A view named after what it shows, its region covering its own file.

    Its file is relative, so it reads the same from either side of a sandbox.
    """
    name = str(overrides.pop("name", f"view_{role}"))
    fields: dict[str, object] = {
        "name": name,
        "role": View(role),
        "file": f"inputs/{role}.dxf",
        "region": Region(view=name, box_uv=(0.0, 0.0, 100.0, 100.0)),
        "dimensions": [],
        **overrides,
    }
    return DrawingView(**fields)  # type: ignore[arg-type]


def drawing(*roles: str) -> list[DrawingView]:
    """One view per role, front alone by default."""
    return [view(role) for role in roles or ("front",)]


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
            **overrides,
        }
    )
