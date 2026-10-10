"""Contextual validation of one stage's submission against its snapshot.

Coding revises its artifacts in the workspace and answers with ticket
responses and a stage report, so what is measured against the snapshot is the
verified output the pipeline read back, handed here as `deliverable`.
"""

from functools import partial

from zeroshot.pipeline.stages._base.validate import (
    SubmissionValidationError,
    raise_together,
)
from zeroshot.pipeline.stages.audit.contracts import AuditReport
from zeroshot.pipeline.stages.audit.validate import validate_audit_report
from zeroshot.pipeline.stages.coding.validate import CodingOutput, validate_coding
from zeroshot.pipeline.stages.contracts import ReconstructionSnapshot
from zeroshot.pipeline.stages.tickets.contracts import TicketAnswers
from zeroshot.pipeline.stages.tickets.validate import validate_ticket_answers
from zeroshot.pipeline.stages.types import (
    REASONING_STAGES,
    next_stage,
)

type Submission = TicketAnswers | AuditReport


def validate_submission(
    submission: Submission,
    snapshot: ReconstructionSnapshot,
    *,
    deliverable: CodingOutput | None = None,
) -> None:
    """Reject a submission that contradicts the round it belongs to.

    `deliverable` is the coding output obtained from workspace verification.
    An audit has none.
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
        partial(_validate_deliverable, deliverable),
    )


def _validate_deliverable(deliverable: CodingOutput | None) -> None:
    if not isinstance(deliverable, CodingOutput):
        raise SubmissionValidationError(
            "coding requires a verified interpretation and a terminal build"
        )
    validate_coding(deliverable)
