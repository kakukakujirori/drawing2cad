"""Contextual validation of one stage's submission against its snapshot.

A reasoning stage submits a revision rather than a whole artifact, so what
is measured against the snapshot is the artifact that revision merges to.
`merge_submission` produces it and hands it here as `deliverable`.
"""

from typing import cast

from zeroshot.pipeline.messages.contracts import (
    DrawingSource,
    OperationPlan,
    SemanticHypothesis,
)
from zeroshot.pipeline.messages.contracts.audit import AuditReport
from zeroshot.pipeline.messages.contracts.reconstruction import (
    CodingSubmission,
    DrawingSubmission,
    OperationSubmission,
    ReconstructionSnapshot,
    SemanticSubmission,
    TicketAnswers,
)
from zeroshot.pipeline.stages._base.validate import (
    SubmissionValidationError,
    validate_ticket_responses,
)
from zeroshot.pipeline.stages.audit.validate import validate_audit_report
from zeroshot.pipeline.stages.coding.validate import validate_coding
from zeroshot.pipeline.stages.operations.validate import validate_operations
from zeroshot.pipeline.stages.semantics.validate import validate_semantics
from zeroshot.pipeline.stages.types import (
    REASONING_STAGES,
    PipelineStage,
    ReasoningStage,
    next_stage,
)
from zeroshot.pipeline.verification import VerifyOutputResult

type Submission = (
    DrawingSubmission
    | SemanticSubmission
    | OperationSubmission
    | CodingSubmission
    | AuditReport
)
type StageDeliverable = (
    DrawingSource | SemanticHypothesis | OperationPlan | VerifyOutputResult
)


def validate_submission(
    submission: Submission,
    snapshot: ReconstructionSnapshot,
    *,
    deliverable: StageDeliverable | None = None,
) -> None:
    """Reject a submission that contradicts the round it belongs to.

    `deliverable` is the complete stage output: merged for proposal stages and
    obtained from workspace verification for drawing and coding. An audit has none.
    """
    if isinstance(submission, AuditReport):
        if deliverable is not None:
            raise SubmissionValidationError("audit does not accept a deliverable")
        validate_audit_report(submission, snapshot)
        return

    if not isinstance(submission, TicketAnswers):
        raise TypeError(f"unsupported submission type: {type(submission).__name__}")

    stage = _next_reasoning_stage(snapshot.last_completed_stage)
    expected_submission = {
        PipelineStage.DRAWINGS: DrawingSubmission,
        PipelineStage.SEMANTICS: SemanticSubmission,
        PipelineStage.OPERATIONS: OperationSubmission,
        PipelineStage.CODING: CodingSubmission,
    }[stage]
    if not isinstance(submission, expected_submission):
        raise SubmissionValidationError(
            f"{stage} must submit a {expected_submission.__name__}"
        )
    validate_ticket_responses(
        submission.responses,
        snapshot.open_tickets,
        expected_stage=stage,
    )
    match stage:
        case PipelineStage.DRAWINGS:
            if not isinstance(deliverable, DrawingSource):
                raise SubmissionValidationError("drawing must revise a DrawingSource")
        case PipelineStage.SEMANTICS:
            if not isinstance(deliverable, SemanticHypothesis):
                raise SubmissionValidationError(
                    "semantics must revise a SemanticHypothesis"
                )
            validate_semantics(deliverable, cast(DrawingSource, snapshot.drawings))
        case PipelineStage.OPERATIONS:
            if not isinstance(deliverable, OperationPlan):
                raise SubmissionValidationError(
                    "operations must revise an OperationPlan"
                )
            validate_operations(deliverable, snapshot)
        case PipelineStage.CODING:
            if not isinstance(deliverable, VerifyOutputResult):
                raise SubmissionValidationError(
                    "coding requires a terminal verification result"
                )
            validate_coding(snapshot, deliverable)
        case _:
            raise SubmissionValidationError(f"unexpected reasoning stage: {stage}")


def _next_reasoning_stage(
    completed_stage: ReasoningStage | None,
) -> ReasoningStage:
    stage = next_stage(completed_stage)
    if stage not in REASONING_STAGES:
        raise SubmissionValidationError(
            "a completed coding snapshot accepts only an AuditReport"
        )
    return stage
