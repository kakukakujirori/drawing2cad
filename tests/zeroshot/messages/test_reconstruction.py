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
    ReconstructionRun,
    ReconstructionSnapshot,
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
        _operations()
        if last_completed_stage in {"operations", "coding"}
        else None
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
        semantics=first.semantics,
        operations=first.operations,
        program_source=first.program_source,
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
