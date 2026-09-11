"""Build validated artifact and ticket-response updates for a snapshot."""

from dataclasses import dataclass, replace
from typing import cast

from zeroshot.pipeline.messages.tickets import TicketResponse
from zeroshot.pipeline.stages._base.validate import SubmissionValidationError
from zeroshot.pipeline.stages.coding.submission import CodingSubmission
from zeroshot.pipeline.stages.contracts import ReconstructionSnapshot
from zeroshot.pipeline.stages.drawings.contracts import DrawingSource
from zeroshot.pipeline.stages.drawings.submission import DrawingSubmission
from zeroshot.pipeline.stages.merge import merge_submission
from zeroshot.pipeline.stages.operations.contracts import OperationPlan
from zeroshot.pipeline.stages.operations.submission import OperationSubmission
from zeroshot.pipeline.stages.resolve_refs import resolve_references
from zeroshot.pipeline.stages.semantics.contracts import SemanticHypothesis
from zeroshot.pipeline.stages.semantics.submission import SemanticSubmission
from zeroshot.pipeline.stages.types import ArtifactField, PipelineStage, ReasoningStage
from zeroshot.pipeline.stages.validate import validate_submission
from zeroshot.pipeline.verification import VerifyOutputResult

type ReasoningSubmission = (
    DrawingSubmission | SemanticSubmission | OperationSubmission | CodingSubmission
)
type WorkspaceOutput = DrawingSource | VerifyOutputResult

_LOG_LIMIT = 4000


@dataclass(frozen=True)
class SnapshotUpdate:
    # Only supplied fields are replaced; None is an explicit replacement.
    # The complete candidate is validated as a ReconstructionSnapshot by the caller.
    artifacts: dict[ArtifactField, object]
    responses: list[TicketResponse]


def build_snapshot_update(
    submission: ReasoningSubmission,
    current: ReconstructionSnapshot,
    previous: ReconstructionSnapshot,
    stage: ReasoningStage,
    *,
    workspace_output: WorkspaceOutput | None = None,
) -> SnapshotUpdate:
    """Merge against the previous round and validate against the current one.

    Verified artifacts are used directly; revision submissions are merged.
    This function neither runs verifiers nor saves state.
    """
    if stage in {PipelineStage.DRAWINGS, PipelineStage.CODING}:
        deliverable = workspace_output
    else:
        if workspace_output is not None:
            raise SubmissionValidationError(
                f"{stage} does not accept a workspace output"
            )
        deliverable = merge_submission(submission, previous, stage)
    validate_submission(submission, current, deliverable=deliverable)

    # Validate the addresses before annotating them. A revised artifact is
    # cited against its own new contents, not the preceding round's values.
    cited_hypothesis = (
        deliverable
        if isinstance(deliverable, SemanticHypothesis)
        else current.semantics
    )
    cited_drawing = cast(
        DrawingSource,
        deliverable if isinstance(deliverable, DrawingSource) else current.drawings,
    )
    if isinstance(deliverable, (DrawingSource, SemanticHypothesis, OperationPlan)):
        deliverable = resolve_references(deliverable, cited_hypothesis, cited_drawing)
    submission = resolve_references(submission, cited_hypothesis, cited_drawing)

    artifacts: dict[ArtifactField, object]
    match stage:
        case PipelineStage.DRAWINGS:
            artifacts = {"drawings": deliverable}
        case PipelineStage.SEMANTICS:
            artifacts = {"semantics": deliverable}
        case PipelineStage.OPERATIONS:
            artifacts = {"operations": deliverable}
        case PipelineStage.CODING:
            terminal = cast(VerifyOutputResult, deliverable)
            artifacts = {
                "program_source": terminal.source,
                "verification": _verification_for_snapshot(terminal),
            }

    return SnapshotUpdate(artifacts=artifacts, responses=submission.responses)


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
