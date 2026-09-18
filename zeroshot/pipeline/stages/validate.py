"""Contextual validation of one stage's submission against its snapshot.

Every reasoning stage revises its artifact in the workspace and answers with
ticket responses and a stage report, so what is measured against the snapshot is the
verified artifact the pipeline read back, handed here as `deliverable`.
"""

from functools import partial

from zeroshot.pipeline.stages._base.validate import (
    SubmissionValidationError,
    raise_together,
)
from zeroshot.pipeline.stages.audit.contracts import AuditReport
from zeroshot.pipeline.stages.audit.validate import validate_audit_report
from zeroshot.pipeline.stages.coding.validate import (
    validate_coding,
)
from zeroshot.pipeline.stages.coding.verify import VerifyOutputResult
from zeroshot.pipeline.stages.contracts import ReconstructionSnapshot
from zeroshot.pipeline.stages.interpretation.contracts import DrawingInterpretation
from zeroshot.pipeline.stages.operations.contracts import OperationPlan
from zeroshot.pipeline.stages.operations.validate import validate_operations
from zeroshot.pipeline.stages.tickets.contracts import TicketAnswers
from zeroshot.pipeline.stages.tickets.validate import validate_ticket_answers
from zeroshot.pipeline.stages.types import (
    REASONING_STAGES,
    PipelineStage,
    ReasoningStage,
    next_stage,
)

type Submission = TicketAnswers | AuditReport
type StageDeliverable = DrawingInterpretation | OperationPlan | VerifyOutputResult


def validate_submission(
    submission: Submission,
    snapshot: ReconstructionSnapshot,
    *,
    deliverable: StageDeliverable | None = None,
) -> None:
    """Reject a submission that contradicts the round it belongs to.

    `deliverable` is the complete stage output obtained from workspace
    verification. An audit has none.
    """
    if isinstance(submission, AuditReport):
        if deliverable is not None:
            raise SubmissionValidationError("audit does not accept a deliverable")
        validate_audit_report(submission, snapshot)
        return

    if not isinstance(submission, TicketAnswers):
        raise TypeError(f"unsupported submission type: {type(submission).__name__}")

    stage = next_stage(snapshot.last_completed_stage)
    if stage not in REASONING_STAGES:
        raise SubmissionValidationError(
            "a completed coding snapshot accepts only an AuditReport"
        )
    raise_together(
        partial(validate_ticket_answers, submission, snapshot),
        partial(_validate_deliverable, stage, snapshot, deliverable),
    )


def _validate_deliverable(
    stage: ReasoningStage,
    snapshot: ReconstructionSnapshot,
    deliverable: StageDeliverable | None,
) -> None:
    match stage:
        case PipelineStage.INTERPRETATION:
            if not isinstance(deliverable, DrawingInterpretation):
                raise SubmissionValidationError(
                    "interpretation requires a verified DrawingInterpretation"
                )
        case PipelineStage.OPERATIONS:
            if not isinstance(deliverable, OperationPlan):
                raise SubmissionValidationError(
                    "operations requires a verified OperationPlan"
                )
            validate_operations(deliverable, snapshot.interpretation)
        case PipelineStage.CODING:
            if not isinstance(deliverable, VerifyOutputResult):
                raise SubmissionValidationError(
                    "coding requires a terminal verification result"
                )
            validate_coding(snapshot, deliverable)
        case _:
            raise SubmissionValidationError(f"unexpected reasoning stage: {stage}")
