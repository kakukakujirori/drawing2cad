from typing import Self

from zeroshot.pipeline.messages.tickets import TicketAnswers


class InterpretationSubmission(TicketAnswers):
    """Ticket responses for the complete interpretation written to JSON."""

    @classmethod
    def unchanged(cls) -> Self:
        return cls(responses=[])
