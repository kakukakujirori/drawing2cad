"""Contextual validation shared by all reconstruction stages."""

from collections.abc import Sequence
from typing import cast

import pytest

from tests.zeroshot.contracts import feature, geometry, hypothesis, replacing
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
from zeroshot.pipeline.messages.contracts.stages import REASONING_STAGES, next_stage
from zeroshot.pipeline.verification import ExecutionStatus, VerifyOutputResult
from zeroshot.pipeline.workflow.merge_submission import merge_submission
from zeroshot.pipeline.workflow.validate_submission import (
    SubmissionValidationError,
    validate_submission,
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


def _merge_and_validate(
    output: SemanticSubmission | OperationSubmission | CodingSubmission,
    snapshot: ReconstructionSnapshot,
    *,
    verification: VerifyOutputResult | None = None,
) -> None:
    """Merge the revision and validate the result, as the pipeline does.

    Every snapshot here belongs to a first round, so there is no preceding
    artifact for the edits to apply to.
    """
    stage = next_stage(snapshot.last_completed_stage)
    if stage not in REASONING_STAGES:
        validate_submission(output, snapshot, verification=verification)
        return
    validate_submission(
        output,
        snapshot,
        deliverable=merge_submission(output, None, stage),
        verification=verification,
    )


def test_every_reasoning_stage_accepts_its_expected_deliverable() -> None:
    _merge_and_validate(
        SemanticSubmission(
            **replacing(_semantics()),
            responses=[_response("ticket_initial", PipelineStage.SEMANTICS)],
        ),
        _snapshot(None),
    )
    _merge_and_validate(
        OperationSubmission(
            **replacing(_operations()),
            responses=[_response("ticket_initial", PipelineStage.OPERATIONS)],
        ),
        _snapshot(PipelineStage.SEMANTICS),
    )
    _merge_and_validate(
        CodingSubmission(
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
        **replacing(_semantics()),
        responses=responses,
    )

    with pytest.raises(SubmissionValidationError, match=message):
        _merge_and_validate(submission, snapshot)


def test_a_stage_answers_its_assigned_tickets_and_only_those() -> None:
    snapshot = _snapshot(
        None,
        tickets=[
            _ticket("ticket_one"),
            _ticket("ticket_two", assigned=(PipelineStage.CODING,)),
        ],
    )

    _merge_and_validate(
        SemanticSubmission(
            **replacing(_semantics()),
            responses=[_response("ticket_one", PipelineStage.SEMANTICS)],
        ),
        snapshot,
    )

    with pytest.raises(SubmissionValidationError, match="not assigned.*ticket_two"):
        _merge_and_validate(
            SemanticSubmission(
                **replacing(_semantics()),
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

    _merge_and_validate(
        SemanticSubmission(**replacing(_semantics()), responses=[]),
        snapshot,
    )


def test_the_current_snapshot_decides_which_deliverable_type_is_valid() -> None:
    operations = OperationSubmission(
        **replacing(_operations()),
        responses=[_response("ticket_initial", PipelineStage.SEMANTICS)],
    )

    with pytest.raises(SubmissionValidationError, match="SemanticSubmission"):
        _merge_and_validate(operations, _snapshot(None))


def test_operations_must_cover_only_current_semantic_features() -> None:
    submission = OperationSubmission(
        **replacing(_operations(semantics=["sem_absent"])),
        responses=[_response("ticket_initial", PipelineStage.OPERATIONS)],
    )

    with pytest.raises(SubmissionValidationError, match="sem_feature_1"):
        _merge_and_validate(submission, _snapshot(PipelineStage.SEMANTICS))


def _validate_plan(
    plan: OperationPlan,
    semantics: SemanticHypothesis,
) -> None:
    _merge_and_validate(
        OperationSubmission(
            **replacing(plan),
            responses=[_response("ticket_initial", PipelineStage.OPERATIONS)],
        ),
        _snapshot(PipelineStage.SEMANTICS, semantics=semantics),
    )


def test_operation_validation_names_both_missing_and_invented_features() -> None:
    semantics = hypothesis("base", "bore")

    with pytest.raises(SubmissionValidationError) as caught:
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
    with pytest.raises(SubmissionValidationError) as caught:
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

    with pytest.raises(SubmissionValidationError, match=address):
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

    with pytest.raises(SubmissionValidationError) as caught:
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
        **replacing(_semantics()),
        responses=[_response("ticket_initial", PipelineStage.SEMANTICS)],
    )
    coding_submission = CodingSubmission(
        responses=[_response("ticket_initial", PipelineStage.CODING)],
    )

    with pytest.raises(SubmissionValidationError, match="must not submit"):
        _merge_and_validate(
            semantic_submission,
            _snapshot(None),
            verification=VerifyOutputResult(status=ExecutionStatus.REJECTED),
        )
    with pytest.raises(SubmissionValidationError, match="requires"):
        _merge_and_validate(coding_submission, _snapshot(PipelineStage.OPERATIONS))
    with pytest.raises(SubmissionValidationError, match="must be terminal"):
        _merge_and_validate(
            coding_submission,
            _snapshot(PipelineStage.OPERATIONS),
            verification=VerifyOutputResult(),
        )


def test_coding_checks_the_submitted_program_against_current_round_operations() -> None:
    submission = CodingSubmission(
        responses=[_response("ticket_initial", PipelineStage.CODING)],
    )

    with pytest.raises(SubmissionValidationError, match="missing.*op_base"):
        _merge_and_validate(
            submission,
            _snapshot(PipelineStage.OPERATIONS),
            verification=VerifyOutputResult(
                status=ExecutionStatus.REJECTED,
                source="ret_other = object()\nresult = ret_other\n",
            ),
        )


def test_coding_keeps_a_terminal_unreadable_program_auditable() -> None:
    submission = CodingSubmission(
        responses=[_response("ticket_initial", PipelineStage.CODING)],
    )

    _merge_and_validate(
        submission,
        _snapshot(PipelineStage.OPERATIONS),
        verification=VerifyOutputResult(status=ExecutionStatus.REJECTED),
    )


def test_completed_coding_accepts_only_an_audit_report() -> None:
    submission = CodingSubmission(
        responses=[_response("ticket_initial", PipelineStage.CODING)],
    )

    with pytest.raises(SubmissionValidationError, match="only an AuditReport"):
        _merge_and_validate(
            submission,
            _snapshot(PipelineStage.CODING),
            verification=VerifyOutputResult(status=ExecutionStatus.REJECTED),
        )
