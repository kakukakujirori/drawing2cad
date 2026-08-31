"""Contextual validation shared by all reconstruction stages."""

from collections.abc import Sequence
from typing import cast

import pytest

from tests.zeroshot.contracts import feature, geometry, hypothesis
from zeroshot.pipeline.messages.contracts import (
    Operation,
    OperationPlan,
    OperationVerb,
    PipelineStage,
    ReasoningStage,
    SemanticHypothesis,
)
from zeroshot.pipeline.messages.contracts.reconstruction import (
    BootstrapWork,
    CodingSubmission,
    OperationSubmission,
    ReconstructionSnapshot,
    SemanticSubmission,
    Ticket,
    TicketResponse,
)
from zeroshot.pipeline.messages.contracts.stages import REASONING_STAGES
from zeroshot.pipeline.verification import ExecutionStatus, VerifyOutputResult
from zeroshot.pipeline.workflow.validate_deliverable import (
    DeliverableValidationError,
    validate_deliverable,
)


def _semantics() -> SemanticHypothesis:
    return hypothesis("the base")


def _operations(*, semantics: list[str] | None = None) -> OperationPlan:
    return OperationPlan(
        proposal=[
            Operation(
                name="op_base",
                verb=OperationVerb.EXTRUDE,
                detail="Extrude the base.",
                depends_on=[],
                semantics=(semantics if semantics is not None else ["sem_feature_1"]),
            )
        ],
        rationale="The base is one extrusion.",
    )


def _plan_for(
    semantics: list[str],
    *,
    detail: str = "Build the feature.",
) -> OperationPlan:
    return OperationPlan(
        proposal=[
            Operation(
                name="op_feature",
                verb=OperationVerb.EXTRUDE,
                detail=detail,
                depends_on=[],
                semantics=semantics,
            )
        ],
        rationale="The operation constructs the named features.",
    )


def _response(ticket_id: str, stage: ReasoningStage) -> TicketResponse:
    return TicketResponse(
        ticket_id=ticket_id,
        stage=stage,
        summary=f"Reviewed {ticket_id} during {stage}.",
    )


def _ticket(
    ticket_id: str,
    *completed_stages: ReasoningStage,
    assigned: Sequence[ReasoningStage] = REASONING_STAGES,
) -> Ticket:
    return Ticket(
        ticket_id=ticket_id,
        subject=BootstrapWork(instruction="Reconstruct the part."),
        assigned_stages=list(assigned),
        responses=[_response(ticket_id, stage) for stage in completed_stages],
    )


def _snapshot(
    completed_stage: ReasoningStage | None,
    *,
    tickets: list[Ticket] | None = None,
    semantics: SemanticHypothesis | None = None,
) -> ReconstructionSnapshot:
    current_semantics = None
    if completed_stage is not None:
        current_semantics = semantics if semantics is not None else _semantics()
    operations = (
        _operations()
        if completed_stage in {PipelineStage.OPERATIONS, PipelineStage.CODING}
        else None
    )
    verification = (
        VerifyOutputResult(
            status=ExecutionStatus.VERIFIED,
            source="ret_base = object()\nresult = ret_base\n",
            returncode=0,
        )
        if completed_stage is PipelineStage.CODING
        else None
    )
    completed_stages = cast(
        tuple[ReasoningStage, ...],
        {
            None: (),
            PipelineStage.SEMANTICS: (PipelineStage.SEMANTICS,),
            PipelineStage.OPERATIONS: (
                PipelineStage.SEMANTICS,
                PipelineStage.OPERATIONS,
            ),
            PipelineStage.CODING: (
                PipelineStage.SEMANTICS,
                PipelineStage.OPERATIONS,
                PipelineStage.CODING,
            ),
        }[completed_stage],
    )
    return ReconstructionSnapshot(
        open_tickets=tickets or [_ticket("ticket_initial", *completed_stages)],
        round=0,
        last_completed_stage=completed_stage,
        semantics=current_semantics,
        operations=operations,
        program_source=verification.source if verification is not None else None,
        verification=verification,
    )


def test_every_reasoning_stage_accepts_its_expected_deliverable() -> None:
    validate_deliverable(
        SemanticSubmission(
            deliverable=_semantics(),
            responses=[_response("ticket_initial", PipelineStage.SEMANTICS)],
        ),
        _snapshot(None),
    )
    validate_deliverable(
        OperationSubmission(
            deliverable=_operations(),
            responses=[_response("ticket_initial", PipelineStage.OPERATIONS)],
        ),
        _snapshot(PipelineStage.SEMANTICS),
    )
    validate_deliverable(
        CodingSubmission(
            deliverable=None,
            responses=[_response("ticket_initial", PipelineStage.CODING)],
        ),
        _snapshot(PipelineStage.OPERATIONS),
        verification=VerifyOutputResult(status=ExecutionStatus.REJECTED),
    )


@pytest.mark.parametrize(
    ("responses", "message"),
    [
        (
            [_response("ticket_one", PipelineStage.SEMANTICS)],
            "missing.*ticket_two",
        ),
        (
            [
                _response("ticket_one", PipelineStage.SEMANTICS),
                _response("ticket_unknown", PipelineStage.SEMANTICS),
            ],
            "unknown.*ticket_unknown",
        ),
        (
            [
                _response("ticket_one", PipelineStage.SEMANTICS),
                _response("ticket_one", PipelineStage.SEMANTICS),
            ],
            "duplicate.*ticket_one",
        ),
        (
            [
                _response("ticket_one", PipelineStage.OPERATIONS),
                _response("ticket_two", PipelineStage.OPERATIONS),
            ],
            "must belong to semantics",
        ),
    ],
)
def test_ticket_responses_must_cover_the_current_snapshot_exactly_once(
    responses: list[TicketResponse],
    message: str,
) -> None:
    snapshot = _snapshot(
        None,
        tickets=[_ticket("ticket_one"), _ticket("ticket_two")],
    )
    submission = SemanticSubmission(
        deliverable=_semantics(),
        responses=responses,
    )

    with pytest.raises(DeliverableValidationError, match=message):
        validate_deliverable(submission, snapshot)


def test_a_stage_answers_its_assigned_tickets_and_only_those() -> None:
    snapshot = _snapshot(
        None,
        tickets=[
            _ticket("ticket_one"),
            _ticket("ticket_two", assigned=(PipelineStage.CODING,)),
        ],
    )

    validate_deliverable(
        SemanticSubmission(
            deliverable=_semantics(),
            responses=[_response("ticket_one", PipelineStage.SEMANTICS)],
        ),
        snapshot,
    )

    with pytest.raises(DeliverableValidationError, match="not assigned.*ticket_two"):
        validate_deliverable(
            SemanticSubmission(
                deliverable=_semantics(),
                responses=[
                    _response("ticket_one", PipelineStage.SEMANTICS),
                    _response("ticket_two", PipelineStage.SEMANTICS),
                ],
            ),
            snapshot,
        )


def test_a_stage_assigned_nothing_answers_nothing() -> None:
    snapshot = _snapshot(
        None,
        tickets=[_ticket("ticket_one", assigned=(PipelineStage.CODING,))],
    )

    validate_deliverable(
        SemanticSubmission(deliverable=_semantics(), responses=[]),
        snapshot,
    )


def test_the_current_snapshot_decides_which_deliverable_type_is_valid() -> None:
    operations = OperationSubmission(
        deliverable=_operations(),
        responses=[_response("ticket_initial", PipelineStage.SEMANTICS)],
    )

    with pytest.raises(DeliverableValidationError, match="SemanticHypothesis"):
        validate_deliverable(operations, _snapshot(None))


def test_operations_must_cover_only_current_semantic_features() -> None:
    submission = OperationSubmission(
        deliverable=_operations(semantics=["sem_absent"]),
        responses=[_response("ticket_initial", PipelineStage.OPERATIONS)],
    )

    with pytest.raises(DeliverableValidationError, match="sem_feature_1"):
        validate_deliverable(submission, _snapshot(PipelineStage.SEMANTICS))


def _validate_plan(
    plan: OperationPlan,
    semantics: SemanticHypothesis,
) -> None:
    validate_deliverable(
        OperationSubmission(
            deliverable=plan,
            responses=[_response("ticket_initial", PipelineStage.OPERATIONS)],
        ),
        _snapshot(PipelineStage.SEMANTICS, semantics=semantics),
    )


def test_operation_validation_names_both_missing_and_invented_features() -> None:
    semantics = hypothesis("base", "bore")

    with pytest.raises(DeliverableValidationError) as caught:
        _validate_plan(_plan_for(["sem_feature_1", "sem_absent"]), semantics)

    message = str(caught.value)
    assert "no operation in the plan builds" in message
    assert "sem_feature_2" in message
    assert "the hypothesis does not contain" in message
    assert "sem_absent" in message


def _measured_blend() -> SemanticHypothesis:
    return hypothesis(
        proposal=[
            feature(
                "sem_shoulder_blend",
                "shoulder blend",
                geometry=[
                    geometry(
                        "torus",
                        name="geo_blend_torus",
                        major_radius=11.31245992416,
                        tube_radius=3.39440063713,
                    )
                ],
            )
        ]
    )


def test_operation_validation_rejects_a_copied_high_precision_number() -> None:
    with pytest.raises(DeliverableValidationError) as caught:
        _validate_plan(
            _plan_for(
                ["sem_shoulder_blend"],
                detail="Sweep a blend of radius 11.31245992416.",
            ),
            _measured_blend(),
        )

    assert "11.31245992416" in str(caught.value)
    assert "sem_<feature>.geo_<claim>.<parameter>" in str(caught.value)


def test_operation_validation_accepts_derived_and_short_numbers() -> None:
    _validate_plan(
        _plan_for(
            ["sem_shoulder_blend"],
            detail=(
                "Cut 5.65622996208 deep, half of "
                "sem_shoulder_blend.geo_blend_torus.major_radius."
            ),
        ),
        _measured_blend(),
    )
    _validate_plan(
        _plan_for(
            ["sem_boss"],
            detail="Extrude 25 mm.",
        ),
        hypothesis(
            proposal=[
                feature(
                    "sem_boss",
                    "boss",
                    geometry=[geometry("sphere", radius=25.0)],
                )
            ]
        ),
    )


def test_operation_validation_rejects_a_nonexistent_parameter_address() -> None:
    address = "sem_shoulder_blend.geo_blend_torus.height"

    with pytest.raises(DeliverableValidationError, match=address):
        _validate_plan(
            _plan_for(
                ["sem_shoulder_blend"],
                detail=f"Sweep {address} along +z.",
            ),
            _measured_blend(),
        )


def test_many_copied_numbers_produce_one_bounded_validation_message() -> None:
    radii = [11.31245992416 + number for number in range(40)]
    semantics = hypothesis(
        proposal=[
            feature(
                "sem_blend",
                "blend",
                geometry=[
                    geometry(
                        "sphere",
                        name=f"geo_sphere_{identifier}",
                        radius=radius,
                    )
                    for identifier, radius in enumerate(radii, start=1)
                ],
            )
        ]
    )

    with pytest.raises(DeliverableValidationError) as caught:
        _validate_plan(
            _plan_for(
                ["sem_blend"],
                detail=" ".join(f"{radius:.11f}" for radius in radii),
            ),
            semantics,
        )

    assert "and 37 more" in str(caught.value)
    assert len(str(caught.value)) < 400


def test_only_coding_accepts_a_separate_terminal_verification() -> None:
    semantic_submission = SemanticSubmission(
        deliverable=_semantics(),
        responses=[_response("ticket_initial", PipelineStage.SEMANTICS)],
    )
    coding_submission = CodingSubmission(
        deliverable=None,
        responses=[_response("ticket_initial", PipelineStage.CODING)],
    )

    with pytest.raises(DeliverableValidationError, match="must not submit"):
        validate_deliverable(
            semantic_submission,
            _snapshot(None),
            verification=VerifyOutputResult(status=ExecutionStatus.REJECTED),
        )
    with pytest.raises(DeliverableValidationError, match="requires"):
        validate_deliverable(coding_submission, _snapshot(PipelineStage.OPERATIONS))
    with pytest.raises(DeliverableValidationError, match="must be terminal"):
        validate_deliverable(
            coding_submission,
            _snapshot(PipelineStage.OPERATIONS),
            verification=VerifyOutputResult(),
        )


def test_coding_checks_the_submitted_program_against_current_round_operations() -> None:
    submission = CodingSubmission(
        deliverable=None,
        responses=[_response("ticket_initial", PipelineStage.CODING)],
    )

    with pytest.raises(DeliverableValidationError, match="missing.*op_base"):
        validate_deliverable(
            submission,
            _snapshot(PipelineStage.OPERATIONS),
            verification=VerifyOutputResult(
                status=ExecutionStatus.REJECTED,
                source="ret_other = object()\nresult = ret_other\n",
            ),
        )


def test_coding_keeps_a_terminal_unreadable_program_auditable() -> None:
    submission = CodingSubmission(
        deliverable=None,
        responses=[_response("ticket_initial", PipelineStage.CODING)],
    )

    validate_deliverable(
        submission,
        _snapshot(PipelineStage.OPERATIONS),
        verification=VerifyOutputResult(status=ExecutionStatus.REJECTED),
    )


def test_completed_coding_accepts_only_an_audit_report() -> None:
    submission = CodingSubmission(
        deliverable=None,
        responses=[_response("ticket_initial", PipelineStage.CODING)],
    )

    with pytest.raises(DeliverableValidationError, match="only an AuditReport"):
        validate_deliverable(
            submission,
            _snapshot(PipelineStage.CODING),
            verification=VerifyOutputResult(status=ExecutionStatus.REJECTED),
        )
