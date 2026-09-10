from typing import Self

from zeroshot.pipeline.messages.tickets import TicketAnswers


class CodingSubmission(TicketAnswers):
    """Your ticket responses. The program itself is read from the workspace.

    This ends the coding stage: give it once, after the program in the
    workspace is the one you mean to submit.
    """

    @classmethod
    def unchanged(cls) -> Self:
        """The answer a coding stage with no ticket of its own would give."""
        return cls(responses=[])
