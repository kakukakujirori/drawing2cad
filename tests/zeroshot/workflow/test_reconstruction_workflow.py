"""Cross-validation and persistence at the workflow boundary."""

import pytest

from tests.zeroshot.contracts import feature, hypothesis, replacing, unchanged
from zeroshot.pipeline.messages.contracts import (
    Operation,
    OperationPlan,
    OperationVerb,
)
from zeroshot.pipeline.messages.contracts.audit import (
    AuditFinding,
    AuditReport,
    CausalHop,
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
from zeroshot.pipeline.workflow import reconstruction as reconstruction_module
from zeroshot.pipeline.workflow.reconstruction import (
    advance_reconstruction,
    load_reconstruction,
    open_next_round,
    save_reconstruction,
    start_reconstruction,
)
from zeroshot.pipeline.workflow.validate_submission import (
    SubmissionValidationError,
    validate_submission,
)

_SOURCE = "ret_base = object()\nret_hole = ret_base.cut(object())\nresult = ret_hole\n"


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
                assigned_stages=["semantics", "operations", "coding"],
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
            assigned_stages=ticket.assigned_stages,
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


def _stage_responses(
    run: ReconstructionRun,
    stage: str,
) -> list[TicketResponse]:
    return [
        TicketResponse(
            ticket_id=ticket.ticket_id,
            stage=stage,  # type: ignore[arg-type]
            summary=f"Reviewed the ticket during {stage}.",
        )
        for ticket in run.snapshots[-1].open_tickets
        if stage in ticket.assigned_stages
    ]


def _completed_run(
    run: ReconstructionRun | None = None,
    verification: VerifyOutputResult | None = None,
) -> ReconstructionRun:
    run = run or start_reconstruction("run_example", "Reconstruct the part.")
    run = advance_reconstruction(
        run,
        SemanticSubmission(
            **replacing(hypothesis("the base", "the hole")),
            responses=_stage_responses(run, "semantics"),
        ),
    )
    run = advance_reconstruction(
        run,
        OperationSubmission(
            **replacing(_operations()),
            responses=_stage_responses(run, "operations"),
        ),
    )
    verification = verification or VerifyOutputResult(
        status=ExecutionStatus.VERIFIED,
        source=_SOURCE,
        returncode=0,
    )
    run = advance_reconstruction(
        run,
        CodingSubmission(
            **unchanged(),
            responses=_stage_responses(run, "coding"),
        ),
        verification=verification,
    )
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
                backtrace=list(hops),
                revision_request=RevisionRequest(
                    action="modify",
                    targets=[revision_target],
                    instruction="Correct the source of the mismatch.",
                    proposed_names=[],
                ),
            )
        ],
    )


def test_audit_cross_validation_accepts_supported_backtrace_hops() -> None:
    report = _report(
        _hop("coding", "ret_hole", "coding", "ret_base"),
        _hop("coding", "ret_base", "operations", "op_base"),
        _hop("operations", "op_base", "semantics", "sem_feature_1"),
    )

    validate_submission(report, _snapshot())


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
    with pytest.raises(SubmissionValidationError, match=message):
        validate_submission(_report(hop), _snapshot())


def test_audit_cross_validation_rejects_a_missing_revision_target() -> None:
    target = _ref("semantics", "sem_absent")
    report = _report(target=target)

    with pytest.raises(SubmissionValidationError, match="does not exist"):
        validate_submission(report, _snapshot())


def test_named_code_references_require_parseable_source() -> None:
    source = "ret_base = (\n"
    report = _report(_hop("coding", "ret_base", "operations", "op_base"))

    with pytest.raises(SubmissionValidationError, match="invalid syntax"):
        validate_submission(report, _snapshot(source))


def test_advance_reconstruction_integrates_each_stage_without_mutating_the_run() -> (
    None
):
    initial = start_reconstruction("run_example", "Reconstruct the part.")
    original_json = initial.model_dump_json()

    completed = _completed_run(initial)

    assert initial.model_dump_json() == original_json
    assert completed.snapshots[-1].last_completed_stage == "coding"
    assert completed.snapshots[-1].program_source == _SOURCE


def test_coding_stores_the_program_once_and_clips_long_logs() -> None:
    noisy = VerifyOutputResult(
        status=ExecutionStatus.VERIFIED,
        source=_SOURCE,
        returncode=0,
        stdout="x" * 10_000,
        stderr="short",
    )

    snapshot = _completed_run(verification=noisy).snapshots[-1]

    assert snapshot.program_source == _SOURCE
    assert snapshot.verification is not None
    assert snapshot.verification.source is None
    assert "characters omitted" in snapshot.verification.stdout
    assert len(snapshot.verification.stdout) < len(noisy.stdout)
    assert snapshot.verification.stderr == "short"


def test_advance_reconstruction_rejects_before_mutating_the_run() -> None:
    run = start_reconstruction("run_example", "Reconstruct the part.")
    original_json = run.model_dump_json()
    wrong_stage = OperationSubmission(
        **replacing(_operations()),
        responses=_stage_responses(run, "semantics"),
    )

    with pytest.raises(SubmissionValidationError, match="SemanticSubmission"):
        advance_reconstruction(run, wrong_stage)

    assert run.model_dump_json() == original_json


def test_advance_reconstruction_matches_responses_by_ticket_id() -> None:
    completed = _completed_run()
    first_finding = _report(target=_ref("semantics", "sem_feature_1")).findings[0]
    second_finding = first_finding.model_copy(
        update={"name": "finding_second_mismatch"}
    )
    run = open_next_round(
        completed,
        AuditReport(
            accepted=False,
            findings=[first_finding, second_finding],
        ),
    )
    current = run.snapshots[-1]
    original_ticket_ids = [ticket.ticket_id for ticket in current.open_tickets]
    responses = [
        TicketResponse(
            ticket_id=ticket.ticket_id,
            stage="semantics",
            summary=f"Addressed {ticket.ticket_id}.",
        )
        for ticket in reversed(current.open_tickets)
    ]

    advanced = advance_reconstruction(
        run,
        SemanticSubmission(
            **replacing(hypothesis("the base", "the hole")),
            responses=responses,
        ),
    )

    updated_tickets = advanced.snapshots[-1].open_tickets
    assert [ticket.ticket_id for ticket in updated_tickets] == original_ticket_ids
    assert [ticket.responses[-1].ticket_id for ticket in updated_tickets] == (
        original_ticket_ids
    )
    assert [ticket.responses[-1].summary for ticket in updated_tickets] == [
        f"Addressed {ticket_id}." for ticket_id in original_ticket_ids
    ]


def test_snapshot_commit_rejects_a_skipped_stage() -> None:
    run = start_reconstruction("run_example", "Reconstruct the part.")

    with pytest.raises(ValueError, match="must advance"):
        reconstruction_module._commit_snapshot(run, _snapshot())


def test_snapshot_commit_preserves_ticket_subjects() -> None:
    run = start_reconstruction("run_example", "Reconstruct the part.")
    semantics = _advance_snapshot(run.snapshots[-1], "semantics")
    original_ticket = semantics.open_tickets[0]
    changed_ticket = Ticket(
        ticket_id=original_ticket.ticket_id,
        subject=BootstrapWork(instruction="A different task."),
        assigned_stages=original_ticket.assigned_stages,
        responses=original_ticket.responses,
    )
    semantics = semantics.model_copy(update={"open_tickets": [changed_ticket]})

    with pytest.raises(ValueError, match="subject must not change"):
        reconstruction_module._commit_snapshot(run, semantics)


def test_snapshot_commit_preserves_prior_responses() -> None:
    run = start_reconstruction("run_example", "Reconstruct the part.")
    semantics = _advance_snapshot(run.snapshots[-1], "semantics")
    run = reconstruction_module._commit_snapshot(run, semantics)
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
        assigned_stages=ticket.assigned_stages,
        responses=[rewritten, ticket.responses[-1]],
    )
    operations = operations.model_copy(update={"open_tickets": [changed_ticket]})

    with pytest.raises(ValueError, match="without rewriting prior responses"):
        reconstruction_module._commit_snapshot(run, operations)


def test_snapshot_commit_preserves_artifacts_owned_by_other_stages() -> None:
    run = start_reconstruction("run_example", "Reconstruct the part.")
    semantics = _advance_snapshot(run.snapshots[-1], "semantics")
    run = reconstruction_module._commit_snapshot(run, semantics)
    operations = _advance_snapshot(run.snapshots[-1], "operations")
    operations = operations.model_copy(
        update={"semantics": hypothesis("a replacement from the wrong stage")}
    )

    with pytest.raises(ValueError, match="must preserve the current semantics"):
        reconstruction_module._commit_snapshot(run, operations)


@pytest.mark.parametrize(
    ("root", "member", "expected"),
    [
        ("semantics", "sem_feature_2", ["semantics", "operations", "coding"]),
        ("operations", "op_hole", ["operations", "coding"]),
        ("coding", "ret_hole", ["coding"]),
    ],
)
def test_a_ticket_is_assigned_from_its_revision_root_downstream(
    root: str,
    member: str,
    expected: list[str],
) -> None:
    report = _report(target=_ref(root, member))

    run = open_next_round(_completed_run(), report)

    assert run.snapshots[-1].open_tickets[0].assigned_stages == expected


def test_one_request_over_several_members_assigns_their_shared_stage() -> None:
    finding = _report(target=_ref("operations", "op_hole")).findings[0]
    two_targets = finding.model_copy(
        update={
            "revision_request": RevisionRequest(
                action="modify",
                targets=[_ref("operations", "op_hole"), _ref("operations", "op_base")],
                instruction="Correct both operations.",
                proposed_names=[],
            )
        }
    )
    report = AuditReport(accepted=False, findings=[two_targets])

    run = open_next_round(_completed_run(), report)

    assert run.snapshots[-1].open_tickets[0].assigned_stages == ["operations", "coding"]


def test_an_unassigned_stage_leaves_the_ticket_untouched() -> None:
    run = open_next_round(_completed_run(), _report(target=_ref("coding", "ret_hole")))

    run = advance_reconstruction(
        run,
        SemanticSubmission(
            **replacing(hypothesis("the base", "the hole")),
            responses=[],
        ),
    )
    ticket = run.snapshots[-1].open_tickets[0]

    assert ticket.assigned_stages == ["coding"]
    assert ticket.responses == []


def test_a_revision_round_carries_the_untouched_hypothesis_forward() -> None:
    run = open_next_round(
        _completed_run(),
        _report(target=_ref("semantics", "sem_feature_2")),
    )
    previous = run.snapshots[-2].semantics
    assert previous is not None

    run = advance_reconstruction(
        run,
        SemanticSubmission(
            edits=[feature("sem_feature_1", "the base, corrected")],
            deleted=[],
            rationale=None,
            responses=_stage_responses(run, "semantics"),
        ),
    )
    current = run.snapshots[-1].semantics
    assert current is not None

    assert [entry.name for entry in current.proposal] == [
        "sem_feature_1",
        "sem_feature_2",
    ]
    assert current.proposal[0].description == "the base, corrected"
    assert current.proposal[1] == previous.proposal[1]
    assert current.rationale == previous.rationale


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
    assert current.semantics is None
    assert current.operations is None
    assert current.program_source is None
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
    with pytest.raises(SubmissionValidationError, match="must use 'ret_hole'"):
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
