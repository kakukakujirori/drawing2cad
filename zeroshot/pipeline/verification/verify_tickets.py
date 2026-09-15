"""Check a stage's ticket answers against the reconstruction history."""

from collections.abc import Callable
from functools import partial

from langchain_core.messages.content import ContentBlock, create_text_block

from zeroshot.pipeline.stages._base.validate import (
    SubmissionValidationError,
    raise_together,
)
from zeroshot.pipeline.stages.contracts import ReconstructionHistory
from zeroshot.pipeline.stages.tickets.contracts import TicketAnswers
from zeroshot.pipeline.stages.tickets.validate import (
    StageArtifact,
    validate_revision_scope,
    validate_ticket_answers,
)


class TicketVerifier:
    """Explain why ticket answers contradict their round or its revision scope.

    Integration applies the same rules; this lets the agent fix them in its turn.
    """

    def __init__(self, artifact: Callable[[], StageArtifact | None]) -> None:
        # The stage's confirmed artifact, or None while it is not confirmed.
        self.artifact = artifact
        self._history: ReconstructionHistory | None = None

    def reset(self, history: ReconstructionHistory) -> None:
        self._history = history

    def feedback(self, answers: TicketAnswers) -> list[ContentBlock]:
        """Nothing when the answers fit the round."""
        if self._history is None:
            raise RuntimeError("the ticket round is not prepared")
        try:
            raise_together(
                partial(validate_ticket_answers, answers, self._history.snapshots[-1]),
                partial(
                    validate_revision_scope,
                    answers.stage_report,
                    self._history,
                    self.artifact(),
                ),
            )
        except SubmissionValidationError as error:
            return [
                create_text_block(
                    "Your TicketAnswers are not ready to submit. Correct them and "
                    f"submit again:\n{error}"
                )
            ]
        return []
