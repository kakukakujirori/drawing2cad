"""Build validated artifact, ticket-response and report updates for a snapshot."""

from dataclasses import dataclass, replace
from functools import partial
from typing import cast

from zeroshot.pipeline.stages._base.validate import raise_together
from zeroshot.pipeline.stages.coding.verify import VerifyOutputResult
from zeroshot.pipeline.stages.contracts import ReconstructionHistory
from zeroshot.pipeline.stages.interpretation.contracts import DrawingInterpretation
from zeroshot.pipeline.stages.operations.contracts import OperationPlan
from zeroshot.pipeline.stages.resolve_refs import resolve_references
from zeroshot.pipeline.stages.tickets.contracts import (
    StageReport,
    TicketAnswers,
    TicketResponse,
)
from zeroshot.pipeline.stages.tickets.validate import validate_revision_scope
from zeroshot.pipeline.stages.types import ArtifactField, PipelineStage, ReasoningStage
from zeroshot.pipeline.stages.validate import validate_submission

type WorkspaceOutput = DrawingInterpretation | OperationPlan | VerifyOutputResult

_LOG_LIMIT = 4000


@dataclass(frozen=True)
class SnapshotUpdate:
    # Only supplied fields are replaced; None is an explicit replacement.
    # The complete candidate is validated as a ReconstructionSnapshot by the caller.
    artifacts: dict[ArtifactField, object]
    responses: list[TicketResponse]
    report: StageReport


def build_snapshot_update(
    submission: TicketAnswers,
    history: ReconstructionHistory,
    stage: ReasoningStage,
    *,
    workspace_output: WorkspaceOutput | None = None,
) -> SnapshotUpdate:
    """Validate the verified workspace artifact against the current round.

    This function neither runs verifiers nor saves state.
    """
    current = history.snapshots[-1]
    deliverable = workspace_output
    artifact = (
        (
            deliverable.exec_report.source
            if deliverable.exec_report is not None
            else None
        )
        if isinstance(deliverable, VerifyOutputResult)
        else deliverable
    )
    raise_together(
        partial(validate_submission, submission, current, deliverable=deliverable),
        partial(
            validate_revision_scope,
            submission.stage_report,
            history,
            artifact,
        ),
    )

    # Validate the addresses before annotating them. A revised artifact is
    # cited against its own new contents, not the preceding round's values.
    interpretation = (
        deliverable
        if isinstance(deliverable, DrawingInterpretation)
        else current.interpretation
    )
    if isinstance(deliverable, OperationPlan):
        deliverable = resolve_references(deliverable, interpretation)
    submission = resolve_references(submission, interpretation)

    artifacts: dict[ArtifactField, object]
    match stage:
        case PipelineStage.INTERPRETATION:
            artifacts = {"interpretation": deliverable}
        case PipelineStage.OPERATIONS:
            artifacts = {"operations": deliverable}
        case PipelineStage.CODING:
            terminal = cast(VerifyOutputResult, deliverable)
            assert terminal.exec_report is not None
            artifacts = {
                "program_source": terminal.exec_report.source,
                "verification": _verification_for_snapshot(terminal),
            }

    return SnapshotUpdate(
        artifacts=artifacts,
        # The stage is stamped here rather than asked for: this is the stage
        # that ran, and a submitted one could only ever agree or be rejected.
        responses=[
            TicketResponse(ticket_id=ticket_id, stage=stage, summary=summary)
            for ticket_id, summary in submission.responses.items()
        ],
        report=submission.stage_report,
    )


def _verification_for_snapshot(verification: VerifyOutputResult) -> VerifyOutputResult:
    """Keep source in program_source only, and bound the persisted logs."""
    exec_report = verification.exec_report
    assert exec_report is not None

    return replace(
        verification,
        exec_report=replace(
            exec_report,
            source=None,
            stdout=_clip_log(exec_report.stdout),
            stderr=_clip_log(exec_report.stderr),
        ),
    )


def _clip_log(log: str) -> str:
    if len(log) <= _LOG_LIMIT:
        return log
    half = _LOG_LIMIT // 2
    omitted = len(log) - 2 * half
    return f"{log[:half]}\n...[{omitted} characters omitted]...\n{log[-half:]}"
