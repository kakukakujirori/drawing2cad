from collections.abc import Callable, Sequence
from typing import Self

# pointer to a JSON element, e.g.
# key_location = ("views", 0, "dimensions", 1)
# means data["views"][0]["dimensions"][1]
type KeyLocation = tuple[str | int, ...]


class SubmissionValidationError(ValueError):
    """A stage's submission contradicts the current reconstruction snapshot."""


class LocatedError(SubmissionValidationError):
    """Semantic errors a check could place in the submitted JSON.

    Pydantic knows where its own errors came from; a check that runs after
    parsing does not, so it says here. Turning a KeyLocation into a line
    belongs to whoever holds the file the model wrote.
    """

    def __init__(self, found: Sequence[tuple[KeyLocation, str]]) -> None:
        super().__init__("\n".join(message for _, message in found))
        self.found = tuple(found)

    @classmethod
    def at(cls, location: KeyLocation, message: str) -> Self:
        return cls([(location, message)])


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
