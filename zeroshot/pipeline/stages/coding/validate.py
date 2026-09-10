from zeroshot.pipeline.stages._base.validate import SubmissionValidationError
from zeroshot.pipeline.stages.contracts import ReconstructionSnapshot
from zeroshot.pipeline.verification import (
    ExecutionStatus,
    VerifyOutputResult,
    check_program,
)


def validate_coding(
    snapshot: ReconstructionSnapshot,
    verification: VerifyOutputResult,
) -> None:
    if verification.status is ExecutionStatus.UNINITIALIZED:
        raise SubmissionValidationError("coding verification must be terminal")
    if snapshot.operations is None:
        raise SubmissionValidationError("coding requires an integrated OperationPlan")

    # A missing or syntactically invalid source is already represented by a
    # terminal verification failure and must remain auditable. When readable
    # source exists, reject plan-to-code identity drift as early as possible.
    if verification.source is None:
        return
    try:
        program_check = check_program(verification.source, snapshot.operations)
    except SyntaxError:
        return
    if program_check.faults:
        raise SubmissionValidationError("\n".join(program_check.faults))
