"""Merge one reasoning stage's revision onto the artifact it revises."""

from pydantic import ValidationError

from zeroshot.pipeline.messages.tickets import TicketAnswers
from zeroshot.pipeline.stages._base.validate import SubmissionValidationError
from zeroshot.pipeline.stages.contracts import ReconstructionSnapshot
from zeroshot.pipeline.stages.operations.contracts import OperationPlan
from zeroshot.pipeline.stages.operations.merge import merge_operations
from zeroshot.pipeline.stages.operations.submission import OperationSubmission
from zeroshot.pipeline.stages.semantics.contracts import SemanticHypothesis
from zeroshot.pipeline.stages.semantics.merge import merge_semantics
from zeroshot.pipeline.stages.semantics.submission import SemanticSubmission
from zeroshot.pipeline.stages.types import PipelineStage, ReasoningStage

type StageArtifact = SemanticHypothesis | OperationPlan


def merge_submission(
    submission: TicketAnswers,
    previous: ReconstructionSnapshot,
    stage: ReasoningStage,
) -> StageArtifact:
    """Apply a revision to the preceding round's artifact."""
    try:
        match stage, submission:
            case PipelineStage.SEMANTICS, SemanticSubmission():
                return merge_semantics(submission, previous.semantics)

            case PipelineStage.OPERATIONS, OperationSubmission():
                return merge_operations(submission, previous.operations)

            case PipelineStage.SEMANTICS, _:
                raise SubmissionValidationError(
                    "semantics must submit a SemanticSubmission"
                )

            case PipelineStage.OPERATIONS, _:
                raise SubmissionValidationError(
                    "operations must submit an OperationSubmission"
                )

            case _:
                raise SubmissionValidationError(
                    f"{stage} uses a workspace output, not a structured revision"
                )

    # Pydantic validation errors may be raised from the merge function outputs
    except ValidationError as error:
        raise SubmissionValidationError(
            f"the revised artifact is not valid: {error}"
        ) from error
