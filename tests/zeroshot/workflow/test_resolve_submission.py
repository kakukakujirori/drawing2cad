"""Interpretation values arrive once, at full precision, including unknowns."""

import pytest

from zeroshot.pipeline.stages.interpretation.contracts import DrawingInterpretation
from zeroshot.pipeline.stages.resolve_refs import (
    _references_resolved_in_prose,
    resolve_references,
)
from zeroshot.pipeline.stages.tickets.contracts import (
    StageReport,
    TicketAnswers,
    TicketResponse,
)
from zeroshot.pipeline.stages.types import PipelineStage


def interpretation() -> DrawingInterpretation:
    return DrawingInterpretation.model_validate(
        {
            "datum": "XY horizontal, Z up; origin at the base corner.",
            "views": [
                {
                    "name": "view_front",
                    "role": "front",
                    "file": "front.png",
                    "region": {"view": "view_front", "box_px": [0, 0, 100, 100]},
                    "u_axis": "+x",
                    "v_axis": "+z",
                    "dimensions": [
                        {
                            "name": "dim_diameter",
                            "kind": "diameter",
                            "text": "2X Ø12",
                            "nominal_value": 12,
                            "quantity": 2,
                            "note": None,
                            "region": {"view": "view_front", "box_px": [0, 0, 20, 20]},
                        }
                    ],
                }
            ],
            "features": [
                {
                    "name": "sem_bore",
                    "description": "A bore through the base.",
                    "parameters": {
                        "radius": 6.000123456789123,
                        "center": [0, None, 3],
                        "depth": None,
                        "zero": 0,
                        "empty": [],
                    },
                    "evidence": [{"view": "view_front", "box_px": [20, 20, 40, 40]}],
                    "dimension_refs": ["dim_diameter"],
                }
            ],
        }
    )


@pytest.mark.parametrize(
    ("address", "value"),
    [
        ("sem_bore.radius", "6.000123456789123"),
        ("sem_bore.center", "[0.0 null 3.0]"),
        ("sem_bore.depth", "null"),
        ("sem_bore.zero", "0.0"),
        ("sem_bore.empty", "[]"),
        ("dim_diameter.nominal_value", "12.0"),
        ("dim_diameter.quantity", "2"),
    ],
)
def test_scalar_array_null_and_dimension_references(address: str, value: str) -> None:
    held = interpretation()
    assert (
        _references_resolved_in_prose(f"Use {address}.", held)
        == f"Use {address} (= {value})."
    )


def test_resolution_refreshes_annotations_and_preserves_unknown_vs_missing() -> None:
    held = interpretation()
    first = _references_resolved_in_prose("sem_bore.radius and sem_bore.depth", held)
    assert _references_resolved_in_prose(first, held) == first
    held.features[0].parameters["radius"] = 9.0
    second = _references_resolved_in_prose(first, held)
    assert second == "sem_bore.radius (= 9.0) and sem_bore.depth (= null)"
    del held.features[0].parameters["radius"]
    assert (
        _references_resolved_in_prose(second, held)
        == "sem_bore.radius and sem_bore.depth (= null)"
    )


@pytest.mark.parametrize(
    "address",
    [
        "sem_missing.radius",
        "sem_bore.missing",
        "sem_bore.geo_cylinder.radius",
        "sem_bore.center.x",
        "ev_front_circle.center",
        "dim_diameter.nominal",
        "dim_diameter.measured_length",
        "dim_missing.quantity",
    ],
)
def test_unknown_or_retired_addresses_stay_unannotated(address: str) -> None:
    assert _references_resolved_in_prose(address, interpretation()) == address


def test_identity_names_and_sentence_punctuation_are_not_parameter_addresses() -> None:
    text = "sem_bore uses dim_diameter. Then cut sem_bore.radius."
    assert _references_resolved_in_prose(text, interpretation()) == (
        "sem_bore uses dim_diameter. Then cut sem_bore.radius (= 6.000123456789123)."
    )
    assert _references_resolved_in_prose("sem_bore.depth", None) == "sem_bore.depth"


def test_whole_answers_are_copied_and_strenum_identity_survives() -> None:
    response = TicketResponse(
        ticket_id="ticket_initial",
        stage=PipelineStage.CODING,
        summary="Cut sem_bore.radius at sem_bore.center; sem_bore.depth is open.",
    )
    original = response.model_dump_json()
    result = resolve_references(response, interpretation())
    assert response.model_dump_json() == original
    assert result.stage is PipelineStage.CODING
    assert "(= [0.0 null 3.0])" in result.summary
    assert result.summary.endswith("(= null) is open.")
    assert resolve_references(result, interpretation()) == result


def test_ticket_summary_and_stage_report_references_resolve_without_mutating_submission():
    submission = TicketAnswers(
        responses={
            "ticket_initial": "Kept sem_bore.radius despite the ticket's ambiguity."
        },
        stage_report=StageReport(
            concerns={
                "concern_checks": "Check dim_diameter.quantity and sem_bore.center."
            },
        ),
    )
    before = submission.model_dump_json()
    resolved = resolve_references(submission, interpretation())

    assert (
        "sem_bore.radius (= 6.000123456789123)" in resolved.responses["ticket_initial"]
    )
    concern = resolved.stage_report.concerns["concern_checks"]
    assert "dim_diameter.quantity (= 2)" in concern
    assert "sem_bore.center (= [0.0 null 3.0])" in concern
    assert submission.model_dump_json() == before
