from collections.abc import Mapping

from zeroshot.pipeline.stages._base.validate import SubmissionValidationError
from zeroshot.pipeline.stages.contracts import ReconstructionSnapshot
from zeroshot.pipeline.stages.interpretation.contracts import DrawingInterpretation
from zeroshot.pipeline.verification import (
    ExecutionStatus,
    VerifyOutputResult,
    check_program,
)


def validate_dimension_checks(
    checks: Mapping[str, str] | None,
    interpretation: DrawingInterpretation | None,
) -> None:
    """Require coverage of the drawing's dimensions, not proof of their geometry."""
    if checks is None:
        raise SubmissionValidationError(
            "coding requires dimension_checks; use {} when there are no dimensions"
        )
    if interpretation is None:
        raise SubmissionValidationError("coding requires an integrated interpretation")
    dimensions = {dimension.name for dimension in interpretation.all_dimensions}
    missing, unknown = (
        sorted(dimensions - checks.keys()),
        sorted(checks.keys() - dimensions),
    )
    if missing or unknown:
        raise SubmissionValidationError(
            "dimension_checks must cover every current dimension exactly once; "
            f"missing: {missing}; unknown: {unknown}"
            + (
                ". Unknown IDs are not interpretation dimensions; report a "
                "printed figure the interpretation lacks in remark instead"
                if unknown
                else ""
            )
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
