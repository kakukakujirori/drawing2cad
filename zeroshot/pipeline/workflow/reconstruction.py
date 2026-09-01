"""Lifecycle transitions and persistence for reconstruction runs."""

import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import cast

from zeroshot.pipeline.messages.contracts import OperationPlan, SemanticHypothesis
from zeroshot.pipeline.messages.contracts.audit import (
    AuditFinding,
    AuditReport,
)
from zeroshot.pipeline.messages.contracts.reconstruction import (
    BootstrapWork,
    CodingSubmission,
    OperationSubmission,
    ReconstructionRun,
    ReconstructionSnapshot,
    SemanticSubmission,
    Ticket,
)
from zeroshot.pipeline.messages.contracts.stages import (
    REASONING_STAGES,
    PipelineStage,
    ReasoningStage,
    next_stage,
)
from zeroshot.pipeline.verification import VerifyOutputResult
from zeroshot.pipeline.workflow.merge_submission import merge_submission
from zeroshot.pipeline.workflow.validate_submission import validate_submission

type ReasoningSubmission = SemanticSubmission | OperationSubmission | CodingSubmission

_LOG_LIMIT = 4000

# ---------------------------------------------------------------------------
# Pure lifecycle transitions
# ---------------------------------------------------------------------------


def start_reconstruction(
    run_id: str,
    instruction: str,
) -> ReconstructionRun:
    snapshot = ReconstructionSnapshot(
        open_tickets=[
            Ticket(
                ticket_id="ticket_initial",
                subject=BootstrapWork(instruction=instruction),
                assigned_stages=list(REASONING_STAGES),
                responses=[],
            )
        ],
        round=0,
        last_completed_stage=None,
        semantics=None,
        operations=None,
        program_source=None,
        verification=None,
    )
    return ReconstructionRun(
        schema_version=1,
        run_id=run_id,
        snapshots=[snapshot],
    )


def open_next_round(
    run: ReconstructionRun,
    report: AuditReport,
) -> ReconstructionRun:
    """Create the next round from a rejected, cross-validated audit report."""
    current = run.snapshots[-1]
    validate_submission(report, current)

    if report.accepted:
        raise ValueError("an accepted audit does not open another round")

    next_round = current.round + 1
    tickets = [_ticket_from_finding(next_round, finding) for finding in report.findings]
    snapshot = ReconstructionSnapshot(
        open_tickets=tickets,
        round=next_round,
        last_completed_stage=None,
        semantics=None,
        operations=None,
        program_source=None,
        verification=None,
    )
    return ReconstructionRun(
        schema_version=run.schema_version,
        run_id=run.run_id,
        snapshots=[*run.snapshots, snapshot],
    )


def _ticket_from_finding(
    round_number: int,
    finding: AuditFinding,
) -> Ticket:
    suffix = finding.name.removeprefix("finding_")
    return Ticket(
        ticket_id=f"ticket_{round_number:03d}_{suffix}",
        subject=finding,
        assigned_stages=_assigned_stages(finding),
        responses=[],
    )


def _assigned_stages(finding: AuditFinding) -> list[ReasoningStage]:
    """The earliest stage the revision targets, and everything after it."""
    root = min(
        REASONING_STAGES.index(target.stage)
        for target in finding.revision_request.targets
    )
    return list(REASONING_STAGES[root:])


def advance_reconstruction(
    run: ReconstructionRun,
    submission: ReasoningSubmission,
    *,
    verification: VerifyOutputResult | None = None,
) -> ReconstructionRun:
    """Validate and atomically integrate one reasoning-stage submission."""
    current = run.snapshots[-1]
    stage = next_stage(current.last_completed_stage)
    if stage not in REASONING_STAGES:
        raise ValueError("a completed coding snapshot cannot advance again")

    preceding = run.snapshots[-2] if len(run.snapshots) > 1 else None
    deliverable = merge_submission(submission, preceding, stage)
    validate_submission(
        submission,
        current,
        deliverable=deliverable,
        verification=verification,
    )

    responses_by_ticket = {
        response.ticket_id: response for response in submission.responses
    }
    tickets = [
        (
            Ticket(
                ticket_id=ticket.ticket_id,
                subject=ticket.subject,
                assigned_stages=ticket.assigned_stages,
                responses=[
                    *ticket.responses,
                    responses_by_ticket[ticket.ticket_id],
                ],
            )
            if stage in ticket.assigned_stages
            else ticket
        )
        for ticket in current.open_tickets
    ]

    semantics = current.semantics
    operations = current.operations
    program_source = current.program_source
    integrated_verification = current.verification
    match stage:
        case PipelineStage.SEMANTICS:
            semantics = cast(SemanticHypothesis, deliverable)
        case PipelineStage.OPERATIONS:
            operations = cast(OperationPlan, deliverable)
        case PipelineStage.CODING:
            terminal = cast(VerifyOutputResult, verification)
            integrated_verification = _durable_verification(terminal)
            program_source = terminal.source

    candidate = ReconstructionSnapshot(
        open_tickets=tickets,
        round=current.round,
        last_completed_stage=stage,
        semantics=semantics,
        operations=operations,
        program_source=program_source,
        verification=integrated_verification,
    )
    return _commit_snapshot(run, candidate)


def _clip_log(log: str) -> str:
    if len(log) <= _LOG_LIMIT:
        return log
    half = _LOG_LIMIT // 2
    omitted = len(log) - 2 * half
    return f"{log[:half]}\n...[{omitted} characters omitted]...\n{log[-half:]}"


def _durable_verification(verification: VerifyOutputResult) -> VerifyOutputResult:
    """Strip what the snapshot already keeps, and what it need not keep whole."""
    return replace(
        verification,
        source=None,  # logged in `program_source` already
        stdout=_clip_log(verification.stdout),
        stderr=_clip_log(verification.stderr),
    )


def _commit_snapshot(
    run: ReconstructionRun,
    snapshot: ReconstructionSnapshot,
) -> ReconstructionRun:
    """Commit one structurally valid current-round stage transition."""
    current = run.snapshots[-1]

    if current.last_completed_stage is PipelineStage.CODING:
        raise ValueError("a completed coding snapshot is immutable")
    if snapshot.round != current.round:
        raise ValueError("replacement must belong to the current round")

    expected_stage = next_stage(current.last_completed_stage)
    if expected_stage not in REASONING_STAGES:
        raise ValueError("a completed coding snapshot cannot be replaced")
    if snapshot.last_completed_stage != expected_stage:
        raise ValueError(
            f"current round must advance from {current.last_completed_stage!r} "
            f"to {expected_stage!r}"
        )

    _require_ticket_progress(current, snapshot, expected_stage)
    _require_only_stage_artifact_changed(current, snapshot, expected_stage)

    return ReconstructionRun(
        schema_version=run.schema_version,
        run_id=run.run_id,
        snapshots=[*run.snapshots[:-1], snapshot],
    )


def _require_ticket_progress(
    current: ReconstructionSnapshot,
    replacement: ReconstructionSnapshot,
    stage: ReasoningStage,
) -> None:
    """Keep ticket identity and prior responses fixed within one round."""
    current_ids = [ticket.ticket_id for ticket in current.open_tickets]
    replacement_ids = [ticket.ticket_id for ticket in replacement.open_tickets]
    if replacement_ids != current_ids:
        raise ValueError("open tickets must not change within a round")

    for previous, updated in zip(
        current.open_tickets,
        replacement.open_tickets,
        strict=True,
    ):
        if updated.subject != previous.subject:
            raise ValueError(
                f"{previous.ticket_id} subject must not change within a round"
            )
        if updated.assigned_stages != previous.assigned_stages:
            raise ValueError(
                f"{previous.ticket_id} assignment must not change within a round"
            )
        if stage not in updated.assigned_stages:
            if updated.responses != previous.responses:
                raise ValueError(
                    f"{previous.ticket_id} is not assigned to {stage} and must "
                    "keep its responses unchanged"
                )
        elif (
            len(updated.responses) != len(previous.responses) + 1
            or updated.responses[:-1] != previous.responses
        ):
            raise ValueError(
                f"{previous.ticket_id} must append one response without "
                "rewriting prior responses"
            )


def _require_only_stage_artifact_changed(
    current: ReconstructionSnapshot,
    replacement: ReconstructionSnapshot,
    stage: ReasoningStage,
) -> None:
    """A stage may replace its own artifact but not an upstream/downstream one."""
    owned_artifact = {
        PipelineStage.SEMANTICS: "semantics",
        PipelineStage.OPERATIONS: "operations",
        PipelineStage.CODING: "program_source",
    }[stage]
    for artifact in ("semantics", "operations", "program_source"):
        if artifact == owned_artifact:
            continue
        if getattr(replacement, artifact) != getattr(current, artifact):
            raise ValueError(f"{stage} must preserve the current {artifact} artifact")


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------


def load_reconstruction(path: Path) -> ReconstructionRun:
    """Load and validate the complete reconstruction history at ``path``."""
    return ReconstructionRun.model_validate_json(path.read_text(encoding="utf-8"))


def save_reconstruction(path: Path, run: ReconstructionRun) -> None:
    """Atomically replace ``path`` with the complete reconstruction history."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = run.model_dump_json(indent=2) + "\n"

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)

    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
