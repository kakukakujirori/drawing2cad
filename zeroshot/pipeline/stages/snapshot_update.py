"""Build validated artifact, ticket-response and report updates for a snapshot."""

from dataclasses import dataclass, replace
from typing import cast

from zeroshot.pipeline.messages.tickets import (
    StageReport,
    TicketAnswers,
    TicketResponse,
)
from zeroshot.pipeline.stages.contracts import ReconstructionSnapshot
from zeroshot.pipeline.stages.interpretation.contracts import DrawingInterpretation
from zeroshot.pipeline.stages.operations.contracts import OperationPlan
from zeroshot.pipeline.stages.resolve_refs import resolve_references
from zeroshot.pipeline.stages.types import ArtifactField, PipelineStage, ReasoningStage
from zeroshot.pipeline.stages.validate import validate_submission
from zeroshot.pipeline.verification import VerifyOutputResult

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
    current: ReconstructionSnapshot,
    stage: ReasoningStage,
    *,
    workspace_output: WorkspaceOutput | None = None,
) -> SnapshotUpdate:
    """Validate the verified workspace artifact against the current round.

    This function neither runs verifiers nor saves state.
    """
    deliverable = workspace_output
    validate_submission(submission, current, deliverable=deliverable)

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
            artifacts = {
                "program_source": terminal.source,
                "verification": _verification_for_snapshot(terminal),
            }

    return SnapshotUpdate(
        artifacts=artifacts,
        responses=submission.responses,
        report=submission.stage_report,
    )


def _verification_for_snapshot(verification: VerifyOutputResult) -> VerifyOutputResult:
    """Keep source in program_source only, and bound the persisted logs."""
    return replace(
        verification,
        source=None,
        stdout=_clip_log(verification.stdout),
        stderr=_clip_log(verification.stderr),
    )


def _clip_log(log: str) -> str:
    if len(log) <= _LOG_LIMIT:
        return log
    half = _LOG_LIMIT // 2
    omitted = len(log) - 2 * half
    return f"{log[:half]}\n...[{omitted} characters omitted]...\n{log[-half:]}"
