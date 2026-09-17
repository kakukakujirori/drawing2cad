"""Interpretation values arrive once, at full precision, including unknowns."""

import pytest

from zeroshot.pipeline.stages._base.validate import SubmissionValidationError
from zeroshot.pipeline.stages.interpretation.contracts import DrawingInterpretation
from zeroshot.pipeline.stages.operations.contracts import (
    Operation,
    OperationPlan,
    OperationVerb,
)
from zeroshot.pipeline.stages.operations.validate import validate_operations
from zeroshot.pipeline.stages.resolve_refs import (
    _references_resolved_in_prose,
    reference_suggestions,
    resolve_references,
    unresolved_references,
)
from zeroshot.pipeline.stages.tickets.contracts import (
    StageReport,
    TicketAnswers,
    TicketResponse,
)


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
    assert unresolved_references(address, held) == []
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
    assert unresolved_references(second, held) == ["sem_bore.radius"]


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
def test_unknown_or_retired_addresses_are_rejected(address: str) -> None:
    assert unresolved_references(address, interpretation()) == [address]


@pytest.mark.parametrize(
    ("address", "suggested"),
    [
        ("sem_bore.radus", ["sem_bore.radius"]),
        ("sem_bor.radius", ["sem_bore.radius"]),
        ("sem_bore.center_x", ["sem_bore.center"]),
        ("dim_diameter.nominal", ["dim_diameter.nominal_value"]),
        ("sem_bore.material", []),
        ("dim_diameter.measured_length", []),
    ],
)
def test_an_unknown_address_suggests_only_close_legal_ones(
    address: str, suggested: list[str]
) -> None:
    assert reference_suggestions(address, interpretation()) == suggested


def test_a_key_no_reference_can_name_is_not_suggested() -> None:
    held = interpretation()
    held.features[0].parameters.update(
        {"Radius": 1.0, "radius-mm": 1.0, "radius_mm (= 1)": 1.0}
    )
    del held.features[0].parameters["radius"]
    assert reference_suggestions("sem_bore.radius", held) == []


def test_a_split_parameter_suggests_each_part_including_a_null_one() -> None:
    held = interpretation()
    parameters = held.features[0].parameters
    del parameters["center"]
    parameters.update(hole_center_x_mm=1.0, hole_center_z_mm=None)
    assert reference_suggestions("sem_bore.center", held) == [
        "sem_bore.hole_center_x_mm",
        "sem_bore.hole_center_z_mm",
    ]
    # Once the member is corrected, an exact parameter needs no alternative.
    assert reference_suggestions("sem_bor.hole_center_x_mm", held) == [
        "sem_bore.hole_center_x_mm"
    ]


def test_an_operation_error_adds_close_addresses_but_accepts_a_null_one() -> None:
    plan = OperationPlan(
        proposal=[
            Operation(
                name="op_bore",
                verb=OperationVerb.HOLE,
                detail="Cut sem_bore.radus to sem_bore.depth.",
                semantics=["sem_bore"],
            )
        ],
        rationale="One bore.",
    )
    with pytest.raises(SubmissionValidationError) as caught:
        validate_operations(plan, interpretation())
    assert str(caught.value) == (
        "op_bore: unknown reference sem_bore.radus. Maybe: sem_bore.radius?"
    )


def test_identity_names_and_sentence_punctuation_are_not_parameter_addresses() -> None:
    text = "sem_bore uses dim_diameter. Then cut sem_bore.radius."
    assert unresolved_references(text, interpretation()) == []
    assert _references_resolved_in_prose(text, interpretation()) == (
        "sem_bore uses dim_diameter. Then cut sem_bore.radius (= 6.000123456789123)."
    )
    assert unresolved_references("sem_bore.depth", None) == ["sem_bore.depth"]


def test_whole_answers_are_copied_and_strenum_identity_survives() -> None:
    plan = OperationPlan(
        proposal=[
            Operation(
                name="op_bore",
                verb=OperationVerb.HOLE,
                detail="Cut sem_bore.radius at sem_bore.center.",
                semantics=["sem_bore"],
            )
        ],
        rationale="Use dim_diameter.quantity holes.",
    )
    original = plan.model_dump_json()
    result = resolve_references(plan, interpretation())
    assert plan.model_dump_json() == original
    assert result.proposal[0].verb is OperationVerb.HOLE
    assert result.proposal[0].name == "op_bore"
    assert result.proposal[0].semantics == ["sem_bore"]
    assert "(= [0.0 null 3.0])" in result.proposal[0].detail
    assert "(= 2)" in result.rationale
    assert resolve_references(result, interpretation()) == result
    response = TicketResponse(
        ticket_id="ticket_initial",
        stage="operations",
        summary="Reviewed sem_bore.depth.",
    )
    assert resolve_references(response, interpretation()).summary.endswith("(= null).")


def test_ticket_summary_and_stage_report_references_resolve_without_mutating_submission():
    submission = TicketAnswers(
        responses=[
            TicketResponse(
                ticket_id="ticket_initial",
                stage="coding",
                summary="Kept sem_bore.radius despite the ticket's ambiguity.",
            )
        ],
        stage_report=StageReport(
            remark="Separately check dim_diameter.quantity and sem_bore.center.",
            dimension_checks={
                "dim_diameter": "Not checked: compare dim_diameter.nominal_value with sem_bore.radius."
            },
        ),
    )
    before = submission.model_dump_json()
    resolved = resolve_references(submission, interpretation())

    assert "sem_bore.radius (= 6.000123456789123)" in resolved.responses[0].summary
    assert "dim_diameter.quantity (= 2)" in resolved.stage_report.remark
    assert "sem_bore.center (= [0.0 null 3.0])" in resolved.stage_report.remark
    assert resolved.stage_report.dimension_checks == {
        "dim_diameter": (
            "Not checked: compare dim_diameter.nominal_value (= 12.0) "
            "with sem_bore.radius (= 6.000123456789123)."
        )
    }
    assert submission.model_dump_json() == before
