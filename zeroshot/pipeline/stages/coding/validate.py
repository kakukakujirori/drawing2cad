from collections.abc import Mapping

from zeroshot.pipeline.stages._base.validate import SubmissionValidationError
from zeroshot.pipeline.stages.coding.verify import VerifyOutputResult
from zeroshot.pipeline.stages.interpretation.contracts import DrawingInterpretation
from zeroshot.pipeline.verification import ExecutionStatus


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
                "printed figure the interpretation lacks in concerns instead"
                if unknown
                else ""
            )
        )


def validate_coding(verification: VerifyOutputResult) -> None:
    """A failed build stays auditable; only an unfinished one is refused."""
    exec_report = verification.exec_report
    if exec_report is None:
        raise SubmissionValidationError("coding verification is not complete")
    if exec_report.status is ExecutionStatus.UNINITIALIZED:
        raise SubmissionValidationError("coding verification must be terminal")
