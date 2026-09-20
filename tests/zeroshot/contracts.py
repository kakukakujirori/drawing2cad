"""Minimal valid pipeline artifacts, for tests about something else.

Tests *about* the contract build it explicitly -- see
`messages/test_interpretation_contracts.py`.
"""

from collections.abc import Iterable

from zeroshot.pipeline.stages.audit.contracts import TicketReview
from zeroshot.pipeline.stages.interpretation.contracts import (
    DrawingInterpretation,
    DrawingView,
    Region,
    View,
)
from zeroshot.pipeline.stages.interpretation.contracts import (
    SemanticFeature as InterpretedFeature,
)
from zeroshot.pipeline.stages.tickets.contracts import TicketResponse

# How each role is drawn on a page that turns no view. Fixtures that are not
# about a turned view take these; the contract itself keeps no such default,
# because a real sheet is what says how its views are drawn.
UNTURNED: dict[View, tuple[str, str]] = {
    View.FRONT: ("+x", "+z"),
    View.BACK: ("-x", "+z"),
    View.TOP: ("+x", "+y"),
    View.BOTTOM: ("+x", "-y"),
    View.RIGHT: ("+y", "+z"),
    View.LEFT: ("-y", "+z"),
}


def answered(responses: Iterable[TicketResponse]) -> dict[str, str]:
    """The submitted shape of responses a test built as the stored ones."""
    return {response.ticket_id: response.summary for response in responses}


def view(role: str = "front", **overrides: object) -> DrawingView:
    """A view named after what it shows, its region covering its own file.

    Its file is relative, so it reads the same from either side of a sandbox.
    """
    name = str(overrides.pop("name", f"view_{role}"))
    axes = UNTURNED.get(View(role), (None, None))
    fields: dict[str, object] = {
        "name": name,
        "role": View(role),
        "file": f"inputs/{role}.dxf",
        "region": Region(view=name, box_uv=(0.0, 0.0, 100.0, 100.0)),
        "dimensions": [],
        "u_axis": axes[0],
        "v_axis": axes[1],
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
                    u_axis="+x",
                    v_axis="+z",
                )
            ],
            "features": [
                interpreted_feature(index, description)
                for index, description in enumerate(descriptions, start=1)
            ],
            **overrides,
        }
    )


def bootstrap_review(
    ticket_id: str = "ticket_initial", *, solved: bool = True
) -> dict[str, TicketReview]:
    """Round 0's one open ticket, which the audit disposes of like any other."""
    return {
        ticket_id: TicketReview(
            summary="The reconstruction answers the order it was given.",
            solved=solved,
        )
    }
