"""Cross-validation and persistence at the workflow boundary."""

import pytest

from tests.zeroshot.contracts import (
    bootstrap_review,
    drawing,
    interpretation,
    interpreted_feature,
)
from zeroshot.pipeline.stages.audit.contracts import (
    AuditFinding,
    AuditReport,
    CausalHop,
    ConcernReview,
    RevisionRequest,
    StageOutputRef,
    TicketReview,
)
from zeroshot.pipeline.stages.coding.verify import VerifyOutputResult
from zeroshot.pipeline.stages.contracts import (
    ReconstructionHistory,
    ReconstructionSnapshot,
)
from zeroshot.pipeline.stages.operations.contracts import (
    Operation,
    OperationPlan,
    OperationVerb,
)
from zeroshot.pipeline.stages.tickets.contracts import (
    BootstrapWork,
    StageReport,
    Ticket,
    TicketAnswers,
    TicketResponse,
)
from zeroshot.pipeline.stages.types import REASONING_STAGES, PipelineStage
from zeroshot.pipeline.stages.validate import (
    SubmissionValidationError,
    validate_submission,
)
from zeroshot.pipeline.verification import ExecutionStatus
from zeroshot.pipeline.workflow import lifecycle as lifecycle_module
from zeroshot.pipeline.workflow.lifecycle import (
    advance_reconstruction,
    interpretation_baseline,
    load_reconstruction,
    open_next_round,
    save_reconstruction,
    start_reconstruction,
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
                semantics=["sem_feature_1"],
            ),
            Operation(
                name="op_hole",
                verb=OperationVerb.HOLE,
                detail="Cut the hole through the base.",
                semantics=["sem_feature_2"],
            ),
        ],
        rationale="The hole follows the base.",
    )


def _snapshot(
    source: str | None = _SOURCE, *, ticket_id: str = "ticket_initial"
) -> ReconstructionSnapshot:
    responses = [
        TicketResponse(
            ticket_id=ticket_id,
            stage=stage,
            summary=f"Reviewed the ticket during {stage}.",
        )
        for stage in REASONING_STAGES
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
                assigned_stages=list(REASONING_STAGES),
                responses=responses,
            )
        ],
        round=0,
        last_completed_stage=PipelineStage.CODING,
        interpretation=interpretation("the base", "the hole"),
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
        interpretation=(
            interpretation("the base", "the hole")
            if stage == "interpretation"
            else current.interpretation
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
    history: ReconstructionHistory,
    stage: str,
) -> list[TicketResponse]:
    return [
        TicketResponse(
            ticket_id=ticket.ticket_id,
            stage=stage,  # type: ignore[arg-type]
            summary=f"Reviewed the ticket during {stage}.",
        )
        for ticket in history.snapshots[-1].open_tickets
        if stage in ticket.assigned_stages
    ]


def _reread(history: ReconstructionHistory) -> ReconstructionHistory:
    return advance_reconstruction(
        history,
        TicketAnswers(responses=_stage_responses(history, "interpretation")),
        workspace_output=interpretation_baseline(history)
        or interpretation("the base", "the hole"),
    )


def _completed_run(
    history: ReconstructionHistory | None = None,
    verification: VerifyOutputResult | None = None,
) -> ReconstructionHistory:
    history = history or start_reconstruction(
        "run_example", "Reconstruct the part.", drawing()
    )
    history = advance_reconstruction(
        history,
        TicketAnswers(responses=_stage_responses(history, "interpretation")),
        workspace_output=interpretation("the base", "the hole"),
    )
    history = advance_reconstruction(
        history,
        TicketAnswers(responses=_stage_responses(history, "operations")),
        workspace_output=_operations(),
    )
    verification = verification or VerifyOutputResult(
        status=ExecutionStatus.VERIFIED,
        source=_SOURCE,
        returncode=0,
    )
    history = advance_reconstruction(
        history,
        TicketAnswers(
            responses=_stage_responses(history, "coding"),
            stage_report=StageReport(dimension_checks={}),
        ),
        workspace_output=verification,
    )
    return history


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


def _hop_from(effect: StageOutputRef, cause: StageOutputRef) -> CausalHop:
    return CausalHop(
        effect=effect,
        cause=cause,
        rationale="The named cause produces the named effect.",
    )


def _report(
    *hops: CausalHop,
    target: StageOutputRef | None = None,
    ticket_reviews: list[TicketReview] | None = None,
    related_ticket_ids: list[str] | None = None,
) -> AuditReport:
    revision_target = target or hops[-1].cause
    return AuditReport(
        concern_reviews=[],
        accepted=False,
        ticket_reviews=[bootstrap_review()]
        if ticket_reviews is None
        else ticket_reviews,
        findings=[
            AuditFinding(
                name="find_shape_mismatch",
                observation="The rendered shape differs from the drawing.",
                evidence=["render_3d/hlg_front.png"],
                backtrace=list(hops),
                related_ticket_ids=related_ticket_ids or [],
                revision_request=RevisionRequest(
                    action="modify",
                    targets=[revision_target],
                    instruction="Correct the source of the mismatch.",
                    proposed_names=[],
                ),
            )
        ],
    )


def _concerned_snapshot() -> ReconstructionSnapshot:
    snapshot = _snapshot()
    snapshot.stage_reports = {
        PipelineStage.INTERPRETATION: StageReport(
            concerns={"concern_web_thickness": "The web thickness is estimated."}
        )
    }
    return snapshot


def test_a_concern_left_without_a_disposition_is_refused() -> None:
    with pytest.raises(SubmissionValidationError, match="concern_web_thickness"):
        validate_submission(
            AuditReport(
                concern_reviews=[],
                accepted=True,
                ticket_reviews=[bootstrap_review()],
                findings=[],
            ),
            _concerned_snapshot(),
        )


def test_a_settled_concern_lets_an_audit_accept_without_findings() -> None:
    validate_submission(
        AuditReport(
            concern_reviews=[
                ConcernReview(
                    concern="interpretation.concern_web_thickness",
                    finding_name=None,
                    disposition="The front view confirms the estimate.",
                )
            ],
            accepted=True,
            ticket_reviews=[bootstrap_review()],
            findings=[],
        ),
        _concerned_snapshot(),
    )


def test_a_concern_may_be_escalated_against_another_stage() -> None:
    """The prefix names the reporter; the finding may target any stage."""
    snapshot = _snapshot()
    snapshot.stage_reports = {
        PipelineStage.CODING: StageReport(
            concerns={"concern_bore_diameter": "sem_bore looks too wide."},
            dimension_checks={},
        )
    }
    report = _report(_hop("operations", "op_base", "interpretation", "sem_feature_1"))
    report.concern_reviews = [
        ConcernReview(
            concern="coding.concern_bore_diameter",
            finding_name="find_shape_mismatch",
            disposition="Confirmed; the interpretation is the root.",
        )
    ]
    validate_submission(report, snapshot)


def test_a_disposition_for_a_concern_nobody_raised_is_refused() -> None:
    report = _report(_hop("operations", "op_base", "interpretation", "sem_feature_1"))
    report.concern_reviews = [
        ConcernReview(
            concern="coding.concern_invented",
            finding_name="find_shape_mismatch",
            disposition="The mismatch finding covers it.",
        )
    ]
    with pytest.raises(SubmissionValidationError, match="concern_invented"):
        validate_submission(report, _snapshot())


def test_audit_cross_validation_accepts_supported_backtrace_hops() -> None:
    report = _report(
        _hop("coding", "ret_hole", "coding", "ret_base"),
        _hop("coding", "ret_base", "operations", "op_base"),
        _hop("operations", "op_base", "interpretation", "sem_feature_1"),
    )

    validate_submission(report, _snapshot())


@pytest.mark.parametrize(
    "status", [s for s in ExecutionStatus if s is not ExecutionStatus.VERIFIED]
)
def test_audit_cannot_accept_without_a_verified_solid(status: ExecutionStatus) -> None:
    snapshot = _snapshot()
    snapshot.verification = VerifyOutputResult(
        status=status, source=_SOURCE, returncode=1
    )
    with pytest.raises(SubmissionValidationError, match="without a verified solid"):
        validate_submission(
            AuditReport(
                concern_reviews=[],
                accepted=True,
                ticket_reviews=[bootstrap_review()],
                findings=[],
            ),
            snapshot,
        )

    # A diagnostic finding remains valid for the same failing program.
    validate_submission(_report(target=_ref("coding", "ret_base")), snapshot)


@pytest.mark.parametrize(
    ("hop", "message"),
    [
        (
            _hop("coding", "ret_base", "operations", "op_hole"),
            "must use 'ret_hole'",
        ),
        (
            _hop("operations", "op_base", "operations", "op_hole"),
            "must name an operation listed before op_base",
        ),
        (
            _hop("operations", "op_hole", "interpretation", "sem_feature_1"),
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


def test_whole_stage_reference_does_not_count_as_a_named_prefix_hop() -> None:
    report = _report(
        _hop_from(_ref("coding", None), _ref("coding", "ret_hole")),
        _hop("coding", "ret_hole", "coding", "ret_base"),
    )
    validate_submission(report, _snapshot())


def test_audit_cross_validation_accepts_one_step_inside_each_stage() -> None:
    """One naming hop in coding and one in operations, on one path."""
    report = _report(
        _hop_from(_ref("coding", None), _ref("coding", "ret_hole")),
        _hop("coding", "ret_hole", "operations", "op_hole"),
        _hop("operations", "op_hole", "operations", "op_base"),
        _hop("operations", "op_base", "interpretation", "sem_feature_1"),
    )

    validate_submission(report, _snapshot())


def test_audit_cross_validation_rejects_a_missing_revision_target() -> None:
    target = _ref("interpretation", "sem_absent")
    report = _report(target=target)

    with pytest.raises(SubmissionValidationError, match="does not exist"):
        validate_submission(report, _snapshot())


@pytest.mark.parametrize(
    ("action", "targets", "proposed", "collision"),
    [
        ("add", [None], ["op_new"], False),
        ("add", [None], ["op_base"], True),
        ("rename", ["op_base"], ["op_hole"], True),
        ("split", ["op_base"], ["op_base", "op_new"], False),
        ("split", ["op_base"], ["op_base", "op_hole"], True),
        ("merge", ["op_base", "op_hole"], ["op_base"], False),
        ("merge", ["op_base", "op_hole"], ["op_new"], False),
    ],
)
def test_audit_new_names_only_reuse_their_own_split_or_merge_targets(
    action, targets, proposed, collision
) -> None:
    finding = _report(target=_ref("operations", targets[0])).findings[0]
    finding.revision_request = RevisionRequest(
        action=action,
        targets=[_ref("operations", name) for name in targets],
        instruction="Correct the operation identities.",
        proposed_names=proposed,
    )
    report = AuditReport(
        concern_reviews=[],
        accepted=False,
        ticket_reviews=[bootstrap_review()],
        findings=[finding],
    )
    if collision:
        with pytest.raises(SubmissionValidationError, match="already exists"):
            validate_submission(report, _snapshot())
    else:
        validate_submission(report, _snapshot())


def test_named_code_references_require_parseable_source() -> None:
    source = "ret_base = (\n"
    report = _report(_hop("coding", "ret_base", "operations", "op_base"))

    with pytest.raises(SubmissionValidationError, match="invalid syntax"):
        validate_submission(report, _snapshot(source))


def test_whole_coding_reference_keeps_invalid_source_auditable() -> None:
    validate_submission(
        _report(target=_ref("coding", None)),
        _snapshot("ret_base = (\n"),
    )


def test_audit_can_address_an_unplanned_code_output() -> None:
    validate_submission(
        _report(target=_ref("coding", "ret_unplanned")),
        _snapshot(_SOURCE + "ret_unplanned = ret_base\n"),
    )


def test_repeated_missing_audit_reference_is_reported_once() -> None:
    report = _report(_hop("coding", "ret_base", "operations", "op_absent"))
    with pytest.raises(SubmissionValidationError) as caught:
        validate_submission(report, _snapshot())
    assert str(caught.value).count("operations member 'op_absent' does not exist") == 1


def test_advance_reconstruction_integrates_each_stage_without_mutating_the_run() -> (
    None
):
    initial = start_reconstruction("run_example", "Reconstruct the part.", drawing())
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
    assert noisy.source == _SOURCE
    assert noisy.stdout == "x" * 10_000


def test_coding_without_readable_source_still_completes_the_round() -> None:
    failed = VerifyOutputResult(
        status=ExecutionStatus.REJECTED,
        source=None,
        stderr="model.py is missing",
    )

    run = _completed_run(verification=failed)
    snapshot = run.snapshots[-1]

    assert snapshot.last_completed_stage is PipelineStage.CODING
    assert snapshot.program_source is None
    assert snapshot.verification is not None
    assert snapshot.verification.status is ExecutionStatus.REJECTED
    assert snapshot.verification.stderr == failed.stderr
    assert snapshot.open_tickets[0].responses[-1].stage is PipelineStage.CODING
    assert ReconstructionHistory.model_validate_json(run.model_dump_json()) == run


def test_advance_reconstruction_rejects_before_mutating_the_run() -> None:
    run = start_reconstruction("run_example", "Reconstruct the part.", drawing())
    original_json = run.model_dump_json()
    answers = TicketAnswers(responses=_stage_responses(run, "interpretation"))

    with pytest.raises(
        SubmissionValidationError, match="verified DrawingInterpretation"
    ):
        advance_reconstruction(run, answers, workspace_output=_operations())

    assert run.model_dump_json() == original_json


def test_advance_reconstruction_matches_responses_by_ticket_id() -> None:
    completed = _completed_run()
    first_finding = _report(target=_ref("interpretation", "sem_feature_1")).findings[0]
    second_finding = first_finding.model_copy(update={"name": "find_second_mismatch"})
    run = open_next_round(
        completed,
        AuditReport(
            concern_reviews=[],
            accepted=False,
            ticket_reviews=[bootstrap_review()],
            findings=[first_finding, second_finding],
        ),
    )
    current = run.snapshots[-1]
    original_ticket_ids = [ticket.ticket_id for ticket in current.open_tickets]
    responses = [
        TicketResponse(
            ticket_id=ticket.ticket_id,
            stage=PipelineStage.INTERPRETATION,
            summary=f"Addressed {ticket.ticket_id}.",
        )
        for ticket in reversed(current.open_tickets)
    ]

    advanced = advance_reconstruction(
        run,
        TicketAnswers(
            responses=responses,
        ),
        workspace_output=interpretation("the base", "the hole"),
    )

    updated_tickets = advanced.snapshots[-1].open_tickets
    assert [ticket.ticket_id for ticket in updated_tickets] == original_ticket_ids
    assert [ticket.responses[-1].ticket_id for ticket in updated_tickets] == (
        original_ticket_ids
    )
    assert [ticket.responses[-1].summary for ticket in updated_tickets] == [
        f"Addressed {ticket_id}." for ticket_id in original_ticket_ids
    ]


def test_integration_resolves_the_references_in_what_it_stores() -> None:
    """Every later reader opens reconstruction.json rather than the prompt that
    carried the plan, so the numbers have to be in it."""
    completed = _completed_run()
    run = open_next_round(
        completed,
        AuditReport(
            concern_reviews=[],
            accepted=False,
            ticket_reviews=[bootstrap_review()],
            findings=[
                _report(target=_ref("interpretation", "sem_feature_1")).findings[0]
            ],
        ),
    )
    current = run.snapshots[-1]
    held = interpretation(
        features=[
            interpreted_feature(1, "the base", parameters={"radius": 4.25}),
            interpreted_feature(2, "the hole"),
        ]
    )

    advanced = advance_reconstruction(
        run,
        TicketAnswers(
            responses=[
                TicketResponse(
                    ticket_id=ticket.ticket_id,
                    stage="interpretation",  # type: ignore[arg-type]
                    summary="Restated sem_feature_1.radius.",
                )
                for ticket in current.open_tickets
            ],
        ),
        workspace_output=held,
    )

    stored = advanced.snapshots[-1]
    assert stored.open_tickets[0].responses[-1].summary == (
        "Restated sem_feature_1.radius (= 4.25)."
    )


def test_snapshot_commit_rejects_a_skipped_stage() -> None:
    run = start_reconstruction("run_example", "Reconstruct the part.", drawing())

    with pytest.raises(ValueError, match="must advance"):
        lifecycle_module._commit_snapshot(run, _snapshot())


def test_snapshot_commit_preserves_ticket_subjects() -> None:
    run = start_reconstruction("run_example", "Reconstruct the part.", drawing())
    run = lifecycle_module._commit_snapshot(
        run, _advance_snapshot(run.snapshots[-1], "interpretation")
    )
    operations = _advance_snapshot(run.snapshots[-1], "operations")
    original_ticket = operations.open_tickets[0]
    changed_ticket = Ticket(
        ticket_id=original_ticket.ticket_id,
        subject=BootstrapWork(instruction="A different task."),
        assigned_stages=original_ticket.assigned_stages,
        responses=original_ticket.responses,
    )
    operations = operations.model_copy(update={"open_tickets": [changed_ticket]})

    with pytest.raises(ValueError, match="subject must not change"):
        lifecycle_module._commit_snapshot(run, operations)


def test_snapshot_commit_preserves_prior_responses() -> None:
    run = start_reconstruction("run_example", "Reconstruct the part.", drawing())
    run = lifecycle_module._commit_snapshot(
        run, _advance_snapshot(run.snapshots[-1], "interpretation")
    )
    operations = _advance_snapshot(run.snapshots[-1], "operations")
    ticket = operations.open_tickets[0]
    rewritten = TicketResponse(
        ticket_id=ticket.ticket_id,
        stage=PipelineStage.INTERPRETATION,
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
        lifecycle_module._commit_snapshot(run, operations)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("interpretation", interpretation("a replacement from the wrong stage")),
        ("verification", VerifyOutputResult(status=ExecutionStatus.REJECTED)),
    ],
)
def test_snapshot_commit_preserves_artifacts_owned_by_other_stages(
    field: str, value: object
) -> None:
    run = start_reconstruction("run_example", "Reconstruct the part.", drawing())
    run = lifecycle_module._commit_snapshot(
        run, _advance_snapshot(run.snapshots[-1], "interpretation")
    )
    operations = _advance_snapshot(run.snapshots[-1], "operations")
    operations = operations.model_copy(update={field: value})

    with pytest.raises(ValueError, match=f"must preserve the current {field}"):
        lifecycle_module._commit_snapshot(run, operations)


@pytest.mark.parametrize(
    ("root", "member", "expected"),
    [
        ("interpretation", "sem_feature_2", ["interpretation", "operations", "coding"]),
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
    report = AuditReport(
        concern_reviews=[],
        accepted=False,
        ticket_reviews=[bootstrap_review()],
        findings=[two_targets],
    )

    run = open_next_round(_completed_run(), report)

    assert run.snapshots[-1].open_tickets[0].assigned_stages == ["operations", "coding"]


def test_an_unassigned_stage_leaves_the_ticket_untouched() -> None:
    run = _reread(
        open_next_round(_completed_run(), _report(target=_ref("coding", "ret_hole")))
    )
    ticket = run.snapshots[-1].open_tickets[0]
    assert ticket.assigned_stages == ["coding"]
    assert ticket.responses == []


def test_a_revision_round_replaces_the_complete_interpretation() -> None:
    run = open_next_round(
        _completed_run(), _report(target=_ref("interpretation", "sem_feature_1"))
    )
    previous = interpretation_baseline(run)
    assert previous is not None
    revised = previous.model_copy(
        update={
            "features": [
                interpreted_feature("sem_feature_1", "the base, corrected"),
                previous.features[1],
            ]
        }
    )
    advanced = advance_reconstruction(
        run,
        TicketAnswers(responses=_stage_responses(run, "interpretation")),
        workspace_output=revised,
    )
    current = advanced.snapshots[-1].interpretation
    assert current == revised
    assert current.features[1] == previous.features[1]
    assert previous.features[0].description == "the base"


def test_rejected_audit_opens_a_fresh_round_without_mutating_history() -> None:
    run = _completed_run()
    original_json = run.model_dump_json()
    report = _report(
        _hop("coding", "ret_hole", "operations", "op_hole"),
        _hop("operations", "op_hole", "interpretation", "sem_feature_2"),
    )

    updated = open_next_round(run, report)
    current = updated.snapshots[-1]

    assert run.model_dump_json() == original_json
    assert len(updated.snapshots) == 2
    assert current.round == 1
    assert current.last_completed_stage is None
    # The accepted reading remains in the preceding immutable snapshot; the
    # new round has not completed its interpretation stage yet.
    assert current.interpretation is None
    assert current.operations is None
    assert current.program_source is None
    assert current.verification is None
    assert current.stage_reports == {}
    assert updated.snapshots[-2].stage_reports == run.snapshots[-1].stage_reports
    assert set(updated.snapshots[-2].stage_reports) == set(REASONING_STAGES)
    assert current.open_tickets[0].ticket_id == "ticket_001_shape_mismatch"
    assert current.open_tickets[0].subject == report.findings[0]
    assert current.open_tickets[0].responses == []


def test_accepted_or_invalid_audit_does_not_open_a_round() -> None:
    run = _completed_run()
    original_json = run.model_dump_json()
    accepted = AuditReport(
        concern_reviews=[],
        accepted=True,
        ticket_reviews=[bootstrap_review()],
        findings=[],
    )
    invalid = _report(_hop("coding", "ret_base", "operations", "op_hole"))

    with pytest.raises(ValueError, match="accepted audit"):
        open_next_round(run, accepted)
    with pytest.raises(SubmissionValidationError, match="must use 'ret_hole'"):
        open_next_round(run, invalid)

    assert run.model_dump_json() == original_json


def test_reconstruction_save_round_trips_the_validated_run(tmp_path) -> None:
    path = tmp_path / "nested" / "reconstruction.json"
    run = start_reconstruction("run_example", "Reconstruct the part.", drawing())

    save_reconstruction(path, run)

    assert load_reconstruction(path) == run
    assert path.read_bytes().endswith(b"\n")


def test_failed_atomic_save_preserves_the_previous_file(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "reconstruction.json"
    original = start_reconstruction("run_original", "Original task.", drawing())
    replacement = start_reconstruction(
        "run_replacement", "Replacement task.", drawing()
    )
    save_reconstruction(path, original)
    original_bytes = path.read_bytes()

    def fail_replace(source, destination) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(lifecycle_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        save_reconstruction(path, replacement)

    assert path.read_bytes() == original_bytes
    assert list(tmp_path.glob(".reconstruction.json.*")) == []


@pytest.mark.parametrize("ticket_id", ["ticket_initial", "ticket_other_bootstrap"])
def test_bootstrap_is_reviewed_like_any_other_open_ticket(ticket_id: str) -> None:
    """Round 0's audit is the only one whose findings become work, so it disposes."""
    snapshot = _snapshot(ticket_id=ticket_id)
    validate_submission(
        _report(
            target=_ref("coding", "ret_hole"),
            ticket_reviews=[
                TicketReview(
                    ticket_id=ticket_id,
                    summary="The reconstruction does not answer the order.",
                    solved=False,
                )
            ],
            related_ticket_ids=[ticket_id],
        ),
        snapshot,
    )

    with pytest.raises(SubmissionValidationError, match=f"missing=.*{ticket_id}"):
        validate_submission(
            _report(target=_ref("coding", "ret_hole"), ticket_reviews=[]), snapshot
        )


@pytest.mark.parametrize(
    ("reviewed", "valid"),
    [
        (["ticket_001_shape_mismatch"], True),
        ([], False),
        (["ticket_initial"], False),
        (["ticket_absent"], False),
        (["ticket_001_shape_mismatch", "ticket_absent"], False),
    ],
)
def test_audit_reviews_cover_every_open_ticket_even_on_acceptance(
    reviewed: list[str], valid: bool
) -> None:
    run = _completed_run(
        open_next_round(_completed_run(), _report(target=_ref("coding", "ret_hole")))
    )
    report = AuditReport(
        concern_reviews=[],
        accepted=True,
        findings=[],
        ticket_reviews=[
            TicketReview(ticket_id=name, summary="The hole is restored.", solved=True)
            for name in reviewed
        ],
    )
    if valid:
        validate_submission(report, run.snapshots[-1])
    else:
        previous = run.model_dump_json()
        with pytest.raises(
            SubmissionValidationError, match="missing=.*unexpected="
        ) as caught:
            open_next_round(run, report)
        assert any(
            name in str(caught.value)
            for name in [*reviewed, "ticket_001_shape_mismatch"]
        )
        assert run.model_dump_json() == previous


@pytest.mark.parametrize("solved_second", [True, False])
def test_current_findings_replace_old_tickets_and_choose_the_new_revision_root(
    solved_second: bool,
) -> None:
    first = _report(target=_ref("coding", "ret_hole")).findings[0]
    second = first.model_copy(update={"name": "find_second_mismatch"})
    run = _completed_run(
        open_next_round(
            _completed_run(),
            AuditReport(
                concern_reviews=[],
                accepted=False,
                ticket_reviews=[bootstrap_review()],
                findings=[first, second],
            ),
        )
    )
    old_ids = [ticket.ticket_id for ticket in run.snapshots[-1].open_tickets]
    reviews = [
        TicketReview(
            ticket_id=name,
            summary="Checked the current solid.",
            solved=solved_second if index else False,
        )
        for index, name in enumerate(old_ids)
    ]
    report = _report(
        _hop("coding", "ret_hole", "operations", "op_hole"),
        _hop("operations", "op_hole", "interpretation", "sem_feature_2"),
        ticket_reviews=reviews,
        related_ticket_ids=[
            review.ticket_id for review in reviews if not review.solved
        ],
    )
    previous = run.model_dump_json()
    updated = open_next_round(run, report)
    assert run.model_dump_json() == previous
    current = updated.snapshots[-1]
    assert current.round == 2
    assert (
        len(current.open_tickets) == 1
    )  # Never copy old unsolved tickets alongside the finding.
    ticket = current.open_tickets[0]
    assert ticket.ticket_id == "ticket_002_shape_mismatch"
    assert ticket.subject == report.findings[0]
    assert ticket.assigned_stages == list(REASONING_STAGES)
    assert ticket.responses == []
    assert not any(
        isinstance(ticket.subject, BootstrapWork) for ticket in current.open_tickets
    )
    assert "audit_report" not in current.model_dump()
    assert "ticket_reviews" not in current.model_dump()
    assert (
        ReconstructionHistory.model_validate_json(updated.model_dump_json()) == updated
    )
