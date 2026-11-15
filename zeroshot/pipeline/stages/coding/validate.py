from dataclasses import dataclass

from zeroshot.pipeline.stages._base.validate import SubmissionValidationError
from zeroshot.pipeline.stages.coding.verify import VerifyOutputResult
from zeroshot.pipeline.stages.interpretation.contracts import DrawingInterpretation
from zeroshot.pipeline.stages.tickets.contracts import TicketAnswers
from zeroshot.pipeline.verification import ExecutionStatus

UNCHANGED_PROGRAM_CONCERN = "concern_program_unchanged"
UNCHANGED_PROGRAM_MESSAGE = (
    "The submitted CadQuery program is unchanged from the previous round. "
    "If you have not yet incorporated identified improvements into model.py, "
    "do so. Otherwise, have you built and compared geometric hypotheses against "
    "the drawing, beyond merely taking measurements? If not, do so before resubmitting.\n\n"
    "Retain the program only with concrete evidence: source-view measurements "
    "that refute the reported defect, or actual candidate builds and projection "
    "comparisons that justify rejecting the tested changes and keeping the "
    "current program. In that case, include "
    f"`{UNCHANGED_PROGRAM_CONCERN}` in stage_report.concerns and cite the specific "
    "views, measurements, build artifacts, and comparison results supporting "
    "that decision. Explain each ticket's outcome in its response. "
    "Time spent measuring, running out of turns, or errors in measurement helpers "
    "are not valid justifications for leaving model.py unchanged."
)


def validate_program_revision(
    answers: TicketAnswers, source: str | None, previous_source: str | None
) -> None:
    """An unchanged revision must be explicit, without forcing arbitrary edits."""
    # Round 0 has no previous program to retain.
    if (
        previous_source is not None
        and source == previous_source
        and not any(
            concern.id == UNCHANGED_PROGRAM_CONCERN
            for concern in answers.stage_report.concerns
        )
    ):
        raise SubmissionValidationError(UNCHANGED_PROGRAM_MESSAGE)


@dataclass(frozen=True)
class CodingOutput:
    """The verified interpretation.json and the terminal build of model.py."""

    interpretation: DrawingInterpretation
    verification: VerifyOutputResult


def validate_coding(output: CodingOutput) -> None:
    """A failed build stays auditable; only an unfinished one is refused."""
    exec_report = output.verification.exec_report
    if exec_report is None:
        raise SubmissionValidationError("coding verification is not complete")
    if exec_report.status is ExecutionStatus.UNINITIALIZED:
        raise SubmissionValidationError("coding verification must be terminal")
