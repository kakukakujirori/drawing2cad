from dataclasses import dataclass

from zeroshot.pipeline.stages._base.validate import SubmissionValidationError
from zeroshot.pipeline.stages.coding.verify import VerifyOutputResult
from zeroshot.pipeline.stages.interpretation.contracts import DrawingInterpretation
from zeroshot.pipeline.verification import ExecutionStatus


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
