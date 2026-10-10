from zeroshot.pipeline.stages._base.validate import SubmissionValidationError
from zeroshot.pipeline.stages.coding.verify import VerifyOutputResult
from zeroshot.pipeline.verification import ExecutionStatus


def validate_coding(verification: VerifyOutputResult) -> None:
    """A failed build stays auditable; only an unfinished one is refused."""
    exec_report = verification.exec_report
    if exec_report is None:
        raise SubmissionValidationError("coding verification is not complete")
    if exec_report.status is ExecutionStatus.UNINITIALIZED:
        raise SubmissionValidationError("coding verification must be terminal")
