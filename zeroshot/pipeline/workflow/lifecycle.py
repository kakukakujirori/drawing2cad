"""Lifecycle transitions and persistence for reconstruction runs."""

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

from zeroshot.pipeline.stages.audit.contracts import (
    AuditFinding,
    AuditReport,
)
from zeroshot.pipeline.stages.contracts import (
    ReconstructionHistory,
    ReconstructionSnapshot,
)
from zeroshot.pipeline.stages.interpretation.contracts import (
    UNDECIDED,
    DrawingInterpretation,
    DrawingView,
)
from zeroshot.pipeline.stages.operations.contracts import OperationPlan
from zeroshot.pipeline.stages.resolve_refs import resolve_references
from zeroshot.pipeline.stages.snapshot_update import (
    WorkspaceOutput,
    build_snapshot_update,
)
from zeroshot.pipeline.stages.tickets.contracts import (
    BootstrapWork,
    Ticket,
    TicketAnswers,
)
from zeroshot.pipeline.stages.types import (
    REASONING_STAGES,
    STAGE_ARTIFACT_FIELDS,
    PipelineStage,
    ReasoningStage,
    next_stage,
)
from zeroshot.pipeline.stages.validate import validate_submission

# ---------------------------------------------------------------------------
# Pure lifecycle transitions
# ---------------------------------------------------------------------------


def start_reconstruction(
    run_id: str,
    instruction: str,
    drawings: list[DrawingView],
) -> ReconstructionHistory:
    """Start a run whose first round has not produced a drawing reading yet."""
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
        interpretation=None,
        operations=None,
        program_source=None,
        verification=None,
    )
    return ReconstructionHistory(
        run_id=run_id,
        input_drawings=drawings,
        snapshots=[snapshot],
    )


def open_next_round(
    history: ReconstructionHistory,
    report: AuditReport,
    evidence_renders: Mapping[str, list[str]] | None = None,
) -> ReconstructionHistory:
    """Create the next round from a rejected, cross-validated audit report.

    Evidence paths refer to the already reviewed, immutable audit attempt.
    Omit them only for transitions that do not involve workspace artifacts.
    """
    current = history.snapshots[-1]
    validate_submission(report, current)

    if not report.findings:
        raise ValueError("an accepted audit does not open another round")

    if evidence_renders is not None and (
        set(evidence_renders) != {finding.name for finding in report.findings}
        or any(
            len(evidence_renders[finding.name]) != len(finding.evidence)
            for finding in report.findings
        )
    ):
        raise ValueError(
            "evidence_renders must contain one path per region of every finding"
        )

    # As for a reasoning stage: the tickets this opens carry the finding's own
    # words into the next round, so the addresses in them are resolved here.
    report = resolve_references(
        report,
        current.interpretation,
    )

    next_round = current.round + 1
    tickets = [
        _ticket_from_finding(
            next_round,
            finding,
            evidence_renders[finding.name] if evidence_renders is not None else [],
        )
        for finding in report.findings
    ]
    snapshot = ReconstructionSnapshot(
        open_tickets=tickets,
        round=next_round,
        last_completed_stage=None,
        interpretation=None,
        operations=None,
        program_source=None,
        verification=None,
    )
    return ReconstructionHistory(
        run_id=history.run_id,
        input_drawings=history.input_drawings,
        snapshots=[*history.snapshots, snapshot],
    )


def interpretation_baseline(history: ReconstructionHistory) -> DrawingInterpretation:
    """The interpretation accepted last round, or the input already registered."""
    accepted = (
        history.snapshots[-2].interpretation if len(history.snapshots) > 1 else None
    )
    return accepted or DrawingInterpretation(
        datum=UNDECIDED,
        views=list(history.input_drawings),
        features=[],
    )


def operations_baseline(history: ReconstructionHistory) -> OperationPlan | None:
    """The accepted operation plan from the preceding round, if any."""
    return history.snapshots[-2].operations if len(history.snapshots) > 1 else None


def _ticket_from_finding(
    round_number: int,
    finding: AuditFinding,
    evidence_renders: list[str],
) -> Ticket:
    ticket_id = f"ticket_{round_number:03d}_{finding.name.removeprefix('find_')}"
    return Ticket(
        ticket_id=ticket_id,
        subject=finding,
        assigned_stages=_assigned_stages(finding),
        responses=[],
        evidence_renders=evidence_renders,
    )


def _assigned_stages(finding: AuditFinding) -> list[ReasoningStage]:
    """The earliest stage the revision targets, and everything after it."""
    root = min(
        REASONING_STAGES.index(target.stage)
        for target in finding.revision_request.targets
    )
    return list(REASONING_STAGES[root:])


def advance_reconstruction(
    history: ReconstructionHistory,
    submission: TicketAnswers,
    *,
    workspace_output: WorkspaceOutput | None = None,
) -> ReconstructionHistory:
    """Validate and atomically integrate one reasoning-stage result.

    Every reasoning stage revises its artifact in the workspace; the verified
    result arrives here alongside the ticket answers and stage report.
    """
    current = history.snapshots[-1]
    stage = next_stage(current.last_completed_stage)
    if stage not in REASONING_STAGES:
        raise ValueError("a completed coding snapshot cannot advance again")

    update = build_snapshot_update(
        submission,
        history,
        stage,
        workspace_output=workspace_output,
    )
    responses_by_ticket = {
        response.ticket_id: response for response in update.responses
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
                evidence_renders=ticket.evidence_renders,
            )
            if stage in ticket.assigned_stages
            else ticket
        )
        for ticket in current.open_tickets
    ]

    # Preserve the model instances and validate the new snapshot. model_copy
    # alone would bypass the checkpoint's structural invariants.
    candidate = ReconstructionSnapshot.model_validate(
        {
            **dict(current),
            **update.artifacts,
            "open_tickets": tickets,
            "last_completed_stage": stage,
            "stage_reports": {**current.stage_reports, stage: update.report},
        }
    )
    return _commit_snapshot(history, candidate)


def _commit_snapshot(
    history: ReconstructionHistory,
    snapshot: ReconstructionSnapshot,
) -> ReconstructionHistory:
    """Commit one structurally valid current-round stage transition."""
    current = history.snapshots[-1]

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

    return ReconstructionHistory(
        run_id=history.run_id,
        input_drawings=history.input_drawings,
        snapshots=[*history.snapshots[:-1], snapshot],
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
    owned_artifacts = STAGE_ARTIFACT_FIELDS[stage]
    for artifacts in STAGE_ARTIFACT_FIELDS.values():
        for artifact in artifacts:
            if artifact in owned_artifacts:
                continue
            if getattr(replacement, artifact) != getattr(current, artifact):
                raise ValueError(
                    f"{stage} must preserve the current {artifact} artifact"
                )
    for other in REASONING_STAGES:
        if other != stage and replacement.stage_reports.get(
            other
        ) != current.stage_reports.get(other):
            raise ValueError(f"{stage} must preserve the current {other} stage report")


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------


def load_reconstruction(path: Path) -> ReconstructionHistory:
    """Load and validate the complete reconstruction history at ``path``."""
    return ReconstructionHistory.model_validate_json(path.read_text(encoding="utf-8"))


def save_reconstruction(path: Path, history: ReconstructionHistory) -> None:
    """Atomically replace ``path`` with the complete reconstruction history."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = history.model_dump_json(indent=2) + "\n"

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
