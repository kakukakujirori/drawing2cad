"""Check a stage's ticket answers against the round they answer."""

from langchain_core.messages.content import ContentBlock, create_text_block

from zeroshot.pipeline.messages.tickets import TicketAnswers
from zeroshot.pipeline.stages._base.validate import SubmissionValidationError
from zeroshot.pipeline.stages.contracts import ReconstructionSnapshot
from zeroshot.pipeline.stages.validate import validate_ticket_answers


class TicketVerifier:
    """Explain why ticket responses or a stage report contradict their round.

    Integration applies the same rules; this lets the agent fix them in its turn.
    """

    def __init__(self) -> None:
        self._snapshot: ReconstructionSnapshot | None = None

    def reset(self, snapshot: ReconstructionSnapshot) -> None:
        self._snapshot = snapshot

    def feedback(self, answers: TicketAnswers) -> list[ContentBlock]:
        """Nothing when the answers fit the round."""
        if self._snapshot is None:
            raise RuntimeError("the ticket round is not prepared")
        try:
            validate_ticket_answers(answers, self._snapshot)
        except SubmissionValidationError as error:
            return [
                create_text_block(
                    "Your TicketAnswers are not ready to submit. Correct them and "
                    f"submit again:\n{error}"
                )
            ]
        return []
