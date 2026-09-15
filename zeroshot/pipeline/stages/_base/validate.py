from collections.abc import Callable


class SubmissionValidationError(ValueError):
    """A stage's submission contradicts the current reconstruction snapshot."""


def raise_together(*checks: Callable[[], None]) -> None:
    """Report every contradiction at once, so one re-ask can fix them together."""
    errors = []
    for check in checks:
        try:
            check()
        except SubmissionValidationError as error:
            errors.append(str(error))
    if errors:
        raise SubmissionValidationError("\n".join(errors))
