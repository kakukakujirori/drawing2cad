import pytest
from pydantic import ValidationError

from tests.zeroshot.contracts import hypothesis
from zeroshot.pipeline.messages.contracts import (
    Operation,
    OperationPlan,
    OperationVerb,
)
from zeroshot.pipeline.messages.contracts.audit import (
    AuditFinding,
    Backtrace,
    RevisionRequest,
    StageOutputRef,
)
from zeroshot.pipeline.messages.contracts.reconstruction import (
    BootstrapWork,
    CodingSubmission,
    OperationSubmission,
    ReconstructionRun,
    ReconstructionSnapshot,
    SemanticSubmission,
    Ticket,
    TicketResponse,
)
from zeroshot.pipeline.verification import ExecutionStatus, VerifyOutputResult


def _semantics():
    return hypothesis("a base body")


def _operations() -> OperationPlan:
    return OperationPlan(
        proposal=[
            Operation(
                name="op_base",
                verb=OperationVerb.EXTRUDE,
                detail="Extrude the base body.",
                depends_on=[],
                semantics=["sem_feature_1"],
            )
        ],
        rationale="The base body is one extrusion.",
    )


def _finding() -> AuditFinding:
    target = StageOutputRef(stage="semantics", name="sem_feature_1")
    return AuditFinding(
        name="finding_wrong_base",
        observation="The reconstructed base is too wide.",
        evidence=["render_3d/hlg_front.png"],
        backtraces=[
            Backtrace(
                hops=[],
                revision_request=RevisionRequest(
                    action="modify",
                    targets=[target],
                    instruction="Correct the interpreted base width.",
                    proposed_names=[],
                ),
            )
        ],
    )


def _responses(ticket_id: str, *stages: str) -> list[TicketResponse]:
    return [
        TicketResponse(
            ticket_id=ticket_id,
            stage=stage,  # type: ignore[arg-type]
            summary=f"Reviewed sem_feature_1 for {stage}.",
        )
        for stage in stages
    ]


def _ticket(
    ticket_id: str = "ticket_bootstrap",
    *,
    subject: BootstrapWork | AuditFinding | None = None,
    stages: tuple[str, ...] = (),
) -> Ticket:
    return Ticket(
        ticket_id=ticket_id,
        subject=subject or BootstrapWork(instruction="Reconstruct the part."),
        responses=_responses(ticket_id, *stages),
    )


def _snapshot(
    *,
    round: int = 0,
    ticket: Ticket | None = None,
    last_completed_stage: str | None = None,
    verification: VerifyOutputResult | None = None,
) -> ReconstructionSnapshot:
    semantics = (
        _semantics()
        if last_completed_stage in {"semantics", "operations", "coding"}
        else None
    )
    operations = (
        _operations() if last_completed_stage in {"operations", "coding"} else None
    )
    return ReconstructionSnapshot(
        open_tickets=[ticket or _ticket()],
        round=round,
        last_completed_stage=last_completed_stage,  # type: ignore[arg-type]
        semantics=semantics,
        operations=operations,
        program_source=verification.source if verification is not None else None,
        verification=verification,
    )


def test_semantics_and_operations_accept_their_concrete_deliverables() -> None:
    semantics = _semantics()
    operations = _operations()

    semantic_submission = SemanticSubmission(
        deliverable=semantics,
        responses=_responses("ticket_bootstrap", "semantics"),
    )
    operation_submission = OperationSubmission(
        deliverable=operations,
        responses=_responses("ticket_bootstrap", "operations"),
    )

    assert semantic_submission.deliverable is semantics
    assert operation_submission.deliverable is operations


def test_coding_requires_an_explicit_null_deliverable() -> None:
    responses = _responses("ticket_bootstrap", "coding")

    submission = CodingSubmission(deliverable=None, responses=responses)

    assert submission.deliverable is None
    with pytest.raises(ValidationError):
        CodingSubmission.model_validate({"responses": responses})
    with pytest.raises(ValidationError):
        CodingSubmission(deliverable="model.py", responses=responses)


def test_a_stage_submission_rejects_empty_responses_and_extra_fields() -> None:
    with pytest.raises(ValidationError, match="at least 1 item"):
        SemanticSubmission(deliverable=_semantics(), responses=[])

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SemanticSubmission.model_validate(
            {
                "deliverable": _semantics(),
                "responses": _responses("ticket_bootstrap", "semantics"),
                "commentary": "not part of the submission contract",
            }
        )


def test_each_stage_submission_exposes_its_concrete_json_schema() -> None:
    semantic_schema = SemanticSubmission.model_json_schema()
    operation_schema = OperationSubmission.model_json_schema()
    coding_schema = CodingSubmission.model_json_schema()

    assert semantic_schema["properties"]["deliverable"]["$ref"].endswith(
        "/SemanticHypothesis"
    )
    assert operation_schema["properties"]["deliverable"]["$ref"].endswith(
        "/OperationPlan"
    )
    assert coding_schema["properties"]["deliverable"]["type"] == "null"


def test_a_round_checkpoint_requires_every_ticket_response_in_stage_order() -> None:
    ticket = _ticket(stages=("semantics",))

    with pytest.raises(ValidationError, match="responses must be"):
        _snapshot(
            ticket=ticket,
            last_completed_stage="operations",
        )


def test_a_ticket_rejects_a_response_for_another_ticket() -> None:
    with pytest.raises(ValidationError, match="containing ticket"):
        Ticket(
            ticket_id="ticket_one",
            subject=BootstrapWork(instruction="Reconstruct the part."),
            responses=_responses("ticket_other", "semantics"),
        )


def test_a_failed_verification_is_a_valid_completed_coding_checkpoint() -> None:
    report = VerifyOutputResult(
        status=ExecutionStatus.REJECTED,
        executor_error="model.py was not found",
    )
    ticket = _ticket(stages=("semantics", "operations", "coding"))

    snapshot = _snapshot(
        ticket=ticket,
        last_completed_stage="coding",
        verification=report,
    )

    assert snapshot.program_source is None
    assert snapshot.verification is report


def test_an_uninitialized_verification_does_not_complete_coding() -> None:
    ticket = _ticket(stages=("semantics", "operations", "coding"))

    with pytest.raises(ValidationError, match="must be completed"):
        _snapshot(
            ticket=ticket,
            last_completed_stage="coding",
            verification=VerifyOutputResult(),
        )


@pytest.mark.parametrize(
    ("completed_stage", "field", "value"),
    [
        (None, "semantics", _semantics()),
        ("semantics", "operations", _operations()),
        ("operations", "program_source", "result = object()\n"),
    ],
)
def test_snapshot_rejects_an_artifact_from_an_unfinished_stage(
    completed_stage: str | None,
    field: str,
    value,
) -> None:
    completed_stages = {
        None: (),
        "semantics": ("semantics",),
        "operations": ("semantics", "operations"),
    }[completed_stage]
    snapshot = _snapshot(
        ticket=_ticket(stages=completed_stages),
        last_completed_stage=completed_stage,
    )
    data = snapshot.model_dump()
    data[field] = value

    with pytest.raises(ValidationError, match="unfinished stage artifacts"):
        ReconstructionSnapshot.model_validate(data)


def test_round_zero_requires_exactly_one_bootstrap_ticket() -> None:
    first = _snapshot()
    second_ticket = _ticket("ticket_second_bootstrap")
    invalid_first = first.model_copy(
        update={"open_tickets": [*first.open_tickets, second_ticket]}
    )

    with pytest.raises(ValidationError, match="exactly one bootstrap"):
        ReconstructionRun(
            schema_version=1,
            run_id="run_example",
            snapshots=[invalid_first],
        )


def test_round_zero_rejects_a_finding_in_place_of_bootstrap_work() -> None:
    first = _snapshot(
        ticket=_ticket("ticket_wrong_base", subject=_finding()),
    )

    with pytest.raises(ValidationError, match="exactly one bootstrap"):
        ReconstructionRun(
            schema_version=1,
            run_id="run_example",
            snapshots=[first],
        )


def test_later_rounds_reject_bootstrap_tickets() -> None:
    first = _snapshot(
        ticket=_ticket(stages=("semantics", "operations", "coding")),
        last_completed_stage="coding",
        verification=VerifyOutputResult(status=ExecutionStatus.REJECTED),
    )
    second = _snapshot(
        round=1,
        ticket=_ticket("ticket_late_bootstrap"),
    )

    with pytest.raises(ValidationError, match="only in round 0"):
        ReconstructionRun(
            schema_version=1,
            run_id="run_example",
            snapshots=[first, second],
        )


def test_round_numbers_follow_snapshot_order() -> None:
    with pytest.raises(ValidationError, match="snapshot rounds"):
        ReconstructionRun(
            schema_version=1,
            run_id="run_example",
            snapshots=[_snapshot(round=1)],
        )


def test_a_run_round_trips_bootstrap_findings_and_verification_as_json() -> None:
    first = _snapshot(
        ticket=_ticket(stages=("semantics", "operations", "coding")),
        last_completed_stage="coding",
        verification=VerifyOutputResult(
            verification_id="000",
            status=ExecutionStatus.VERIFIED,
            source="result = object()",
            returncode=0,
        ),
    )
    second = ReconstructionSnapshot(
        open_tickets=[
            _ticket(
                "ticket_wrong_base",
                subject=_finding(),
            )
        ],
        round=1,
        last_completed_stage=None,
        semantics=None,
        operations=None,
        program_source=None,
        verification=None,
    )
    run = ReconstructionRun(
        schema_version=1,
        run_id="run_example",
        snapshots=[first, second],
    )

    restored = ReconstructionRun.model_validate_json(run.model_dump_json())

    assert restored == run
    assert isinstance(restored.snapshots[0].verification, VerifyOutputResult)
    assert restored.snapshots[0].verification is not None
    assert restored.snapshots[0].verification.status is ExecutionStatus.VERIFIED
    assert isinstance(restored.snapshots[1].open_tickets[0].subject, AuditFinding)
