"""Cross-validation and persistence at the workflow boundary."""

import pytest

from tests.zeroshot.contracts import hypothesis
from zeroshot.pipeline.messages.contracts import (
    Operation,
    OperationPlan,
    OperationVerb,
)
from zeroshot.pipeline.messages.contracts.audit import (
    AuditFinding,
    AuditReport,
    Backtrace,
    CausalHop,
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
from zeroshot.pipeline.workflow import reconstruction as reconstruction_module
from zeroshot.pipeline.workflow.reconstruction import (
    AuditCrossValidationError,
    check_audit_report,
    load_reconstruction,
    open_next_round,
    replace_current_snapshot,
    save_reconstruction,
    start_reconstruction,
)

_SOURCE = "ret_base = object()\nret_hole = ret_base\nresult = ret_hole\n"


def _ref(stage: str, name: str | None) -> StageOutputRef:
    return StageOutputRef(stage=stage, name=name)  # type: ignore[arg-type]


def _operations() -> OperationPlan:
    return OperationPlan(
        proposal=[
            Operation(
                name="op_base",
                verb=OperationVerb.EXTRUDE,
                detail="Extrude the base.",
                depends_on=[],
                semantics=["sem_feature_1"],
            ),
            Operation(
                name="op_hole",
                verb=OperationVerb.HOLE,
                detail="Cut the hole through the base.",
                depends_on=["op_base"],
                semantics=["sem_feature_2"],
            ),
        ],
        rationale="The hole follows the base.",
    )


def _snapshot(source: str | None = _SOURCE) -> ReconstructionSnapshot:
    ticket_id = "ticket_initial"
    responses = [
        TicketResponse(
            ticket_id=ticket_id,
            stage=stage,  # type: ignore[arg-type]
            summary=f"Reviewed the ticket during {stage}.",
        )
        for stage in ("semantics", "operations", "coding")
    ]
    verification = VerifyOutputResult(
        status=(
            ExecutionStatus.VERIFIED if source is not None else ExecutionStatus.REJECTED
        ),
        source=source,
        returncode=0 if source is not None else None,
    )
    return ReconstructionSnapshot(
        open_tickets=[
            Ticket(
                ticket_id=ticket_id,
                subject=BootstrapWork(instruction="Reconstruct the part."),
                responses=responses,
            )
        ],
        round=0,
        last_completed_stage="coding",
        semantics=hypothesis("the base", "the hole"),
        operations=_operations(),
        program_source=source,
        verification=verification,
    )


def _advance_snapshot(
    current: ReconstructionSnapshot,
    stage: str,
) -> ReconstructionSnapshot:
    tickets = [
        Ticket(
            ticket_id=ticket.ticket_id,
            subject=ticket.subject,
            responses=[
                *ticket.responses,
                TicketResponse(
                    ticket_id=ticket.ticket_id,
                    stage=stage,  # type: ignore[arg-type]
                    summary=f"Reviewed the ticket during {stage}.",
                ),
            ],
        )
        for ticket in current.open_tickets
    ]
    return ReconstructionSnapshot(
        open_tickets=tickets,
        round=current.round,
        last_completed_stage=stage,  # type: ignore[arg-type]
        semantics=(
            hypothesis("the base", "the hole")
            if stage == "semantics"
            else current.semantics
        ),
        operations=_operations() if stage == "operations" else current.operations,
        program_source=_SOURCE if stage == "coding" else current.program_source,
        verification=(
            VerifyOutputResult(
                status=ExecutionStatus.VERIFIED,
                source=_SOURCE,
                returncode=0,
            )
            if stage == "coding"
            else None
        ),
    )


def _completed_run() -> ReconstructionRun:
    run = start_reconstruction("run_example", "Reconstruct the part.")
    for stage in ("semantics", "operations", "coding"):
        snapshot = _advance_snapshot(run.snapshots[-1], stage)
        run = replace_current_snapshot(run, snapshot)
    return run


def _hop(
    effect_stage: str,
    effect_name: str,
    cause_stage: str,
    cause_name: str,
) -> CausalHop:
    return CausalHop(
        effect=_ref(effect_stage, effect_name),
        cause=_ref(cause_stage, cause_name),
        rationale="The named cause produces the named effect.",
    )


def _report(*hops: CausalHop, target: StageOutputRef | None = None) -> AuditReport:
    revision_target = target or hops[-1].cause
    return AuditReport(
        accepted=False,
        findings=[
            AuditFinding(
                name="finding_shape_mismatch",
                observation="The rendered shape differs from the drawing.",
                evidence=["render_3d/hlg_front.png"],
                backtraces=[
                    Backtrace(
                        hops=list(hops),
                        revision_request=RevisionRequest(
                            action="modify",
                            targets=[revision_target],
                            instruction="Correct the source of the mismatch.",
                            proposed_names=[],
                        ),
                    )
                ],
            )
        ],
    )


def test_audit_cross_validation_accepts_supported_backtrace_hops() -> None:
    report = _report(
        _hop("coding", "result", "coding", "ret_hole"),
        _hop("coding", "ret_hole", "operations", "op_hole"),
        _hop("operations", "op_hole", "semantics", "sem_feature_2"),
    )

    check_audit_report(report, _snapshot())


@pytest.mark.parametrize(
    ("hop", "message"),
    [
        (
            _hop("coding", "ret_base", "operations", "op_hole"),
            "must use 'ret_hole'",
        ),
        (
            _hop("operations", "op_base", "operations", "op_hole"),
            "op_base.depends_on",
        ),
        (
            _hop("operations", "op_hole", "semantics", "sem_feature_1"),
            "op_hole.semantics",
        ),
    ],
)
def test_audit_cross_validation_rejects_unsupported_contract_links(
    hop: CausalHop,
    message: str,
) -> None:
    with pytest.raises(AuditCrossValidationError, match=message):
        check_audit_report(_report(hop), _snapshot())


def test_audit_cross_validation_rejects_a_missing_revision_target() -> None:
    target = _ref("semantics", "sem_absent")
    report = _report(target=target)

    with pytest.raises(AuditCrossValidationError, match="does not exist"):
        check_audit_report(report, _snapshot())


def test_named_code_references_require_parseable_source() -> None:
    source = "ret_base = (\n"
    report = _report(_hop("coding", "ret_base", "operations", "op_base"))

    with pytest.raises(AuditCrossValidationError, match="invalid syntax"):
        check_audit_report(report, _snapshot(source))


def test_snapshot_replacement_advances_one_stage_without_mutating_the_run() -> None:
    initial = start_reconstruction("run_example", "Reconstruct the part.")
    original_json = initial.model_dump_json()

    semantics = _advance_snapshot(initial.snapshots[-1], "semantics")
    after_semantics = replace_current_snapshot(initial, semantics)
    operations = _advance_snapshot(after_semantics.snapshots[-1], "operations")
    after_operations = replace_current_snapshot(after_semantics, operations)
    coding = _advance_snapshot(after_operations.snapshots[-1], "coding")
    completed = replace_current_snapshot(after_operations, coding)

    assert initial.model_dump_json() == original_json
    assert completed.snapshots[-1].last_completed_stage == "coding"


def test_snapshot_replacement_rejects_a_skipped_stage() -> None:
    run = start_reconstruction("run_example", "Reconstruct the part.")

    with pytest.raises(ValueError, match="must advance"):
        replace_current_snapshot(run, _snapshot())


def test_snapshot_replacement_preserves_ticket_subjects() -> None:
    run = start_reconstruction("run_example", "Reconstruct the part.")
    semantics = _advance_snapshot(run.snapshots[-1], "semantics")
    original_ticket = semantics.open_tickets[0]
    changed_ticket = Ticket(
        ticket_id=original_ticket.ticket_id,
        subject=BootstrapWork(instruction="A different task."),
        responses=original_ticket.responses,
    )
    semantics = semantics.model_copy(update={"open_tickets": [changed_ticket]})

    with pytest.raises(ValueError, match="subject must not change"):
        replace_current_snapshot(run, semantics)


def test_snapshot_replacement_preserves_prior_responses() -> None:
    run = start_reconstruction("run_example", "Reconstruct the part.")
    semantics = _advance_snapshot(run.snapshots[-1], "semantics")
    run = replace_current_snapshot(run, semantics)
    operations = _advance_snapshot(run.snapshots[-1], "operations")
    ticket = operations.open_tickets[0]
    rewritten = TicketResponse(
        ticket_id=ticket.ticket_id,
        stage="semantics",
        summary="Rewrote the earlier response.",
    )
    changed_ticket = Ticket(
        ticket_id=ticket.ticket_id,
        subject=ticket.subject,
        responses=[rewritten, ticket.responses[-1]],
    )
    operations = operations.model_copy(update={"open_tickets": [changed_ticket]})

    with pytest.raises(ValueError, match="without rewriting prior responses"):
        replace_current_snapshot(run, operations)


def test_snapshot_replacement_preserves_artifacts_owned_by_other_stages() -> None:
    run = start_reconstruction("run_example", "Reconstruct the part.")
    semantics = _advance_snapshot(run.snapshots[-1], "semantics")
    run = replace_current_snapshot(run, semantics)
    operations = _advance_snapshot(run.snapshots[-1], "operations")
    operations = operations.model_copy(
        update={"semantics": hypothesis("a replacement from the wrong stage")}
    )

    with pytest.raises(ValueError, match="must preserve the current semantics"):
        replace_current_snapshot(run, operations)


def test_rejected_audit_opens_a_fresh_round_without_mutating_history() -> None:
    run = _completed_run()
    original_json = run.model_dump_json()
    report = _report(
        _hop("coding", "ret_hole", "operations", "op_hole"),
        _hop("operations", "op_hole", "semantics", "sem_feature_2"),
    )

    updated = open_next_round(run, report)
    current = updated.snapshots[-1]

    assert run.model_dump_json() == original_json
    assert len(updated.snapshots) == 2
    assert current.round == 1
    assert current.last_completed_stage is None
    assert current.semantics == run.snapshots[-1].semantics
    assert current.operations == run.snapshots[-1].operations
    assert current.program_source == run.snapshots[-1].program_source
    assert current.verification is None
    assert current.open_tickets[0].ticket_id == "ticket_001_shape_mismatch"
    assert current.open_tickets[0].subject == report.findings[0]
    assert current.open_tickets[0].responses == []


def test_accepted_or_invalid_audit_does_not_open_a_round() -> None:
    run = _completed_run()
    original_json = run.model_dump_json()
    accepted = AuditReport(accepted=True, findings=[])
    invalid = _report(_hop("coding", "ret_base", "operations", "op_hole"))

    with pytest.raises(ValueError, match="accepted audit"):
        open_next_round(run, accepted)
    with pytest.raises(AuditCrossValidationError, match="must use 'ret_hole'"):
        open_next_round(run, invalid)

    assert run.model_dump_json() == original_json


def test_reconstruction_save_round_trips_the_validated_run(tmp_path) -> None:
    path = tmp_path / "nested" / "reconstruction.json"
    run = start_reconstruction("run_example", "Reconstruct the part.")

    save_reconstruction(path, run)

    assert load_reconstruction(path) == run
    assert path.read_bytes().endswith(b"\n")


def test_failed_atomic_save_preserves_the_previous_file(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "reconstruction.json"
    original = start_reconstruction("run_original", "Original task.")
    replacement = start_reconstruction("run_replacement", "Replacement task.")
    save_reconstruction(path, original)
    original_bytes = path.read_bytes()

    def fail_replace(source, destination) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(reconstruction_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        save_reconstruction(path, replacement)

    assert path.read_bytes() == original_bytes
    assert list(tmp_path.glob(".reconstruction.json.*")) == []
