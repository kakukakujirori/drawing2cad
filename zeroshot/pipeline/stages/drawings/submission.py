from typing import Self

from zeroshot.pipeline.messages.tickets import TicketAnswers


class DrawingSubmission(TicketAnswers):
    """Ticket responses for the DrawingSource written to ``drawing.json``."""

    @classmethod
    def unchanged(cls) -> Self:
        """The answer when this round assigned no ticket to drawings."""
        return cls(responses=[])
