"""Verify the coder's two files: a valid interpretation.json gates the build."""

from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

from langchain_core.messages.content import ContentBlock, create_text_block

from zeroshot.pipeline.stages.coding.progress import ProgressOutputVerifier
from zeroshot.pipeline.stages.interpretation.contracts import DrawingInterpretation
from zeroshot.pipeline.stages.interpretation.verify import InterpretationVerifier

_GATED = (
    "{program} is built only after {interpretation} validates, because the "
    "build projects the solid into the views {interpretation} registers."
)
_NO_FEATURES = (
    "{interpretation} lists no features. Record the part's features before answering."
)


def _build_inputs(interpretation: DrawingInterpretation) -> list[dict[str, Any]]:
    """What a build reads from the views: files, roles, axes and scales."""
    return [view.model_dump(exclude={"dimensions"}) for view in interpretation.views]


class CodingWorkspaceVerifier:
    """Report both files after a write, and build model.py on a valid reading."""

    def __init__(
        self, interpretation: InterpretationVerifier, output: ProgressOutputVerifier
    ) -> None:
        self.interpretation = interpretation
        self.output = output
        self.ticket_ids: Sequence[str] = ()
        self.interpretation_ready = False  # valid once in this invocation
        self._interpretation_reported: str | None = None
        self._output_reported: str | None = None

    @property
    def source_path(self) -> Path:
        """The program; `source_digest` also watches interpretation.json."""
        return self.output.source_path

    def source_digest(self) -> str:
        both = f"{self.interpretation.source_digest()}|{self.output.source_digest()}"
        return sha256(both.encode()).hexdigest()

    def reset(self, baseline: DrawingInterpretation, ticket_ids: Sequence[str]) -> None:
        """Seed interpretation.json and forget every build of the last invocation."""
        self.interpretation.reset(baseline)
        self.output.reset()
        self.output.interpretation = None
        self.ticket_ids = ticket_ids
        self.interpretation_ready = False
        self._interpretation_reported = None
        self._output_reported = None

    @property
    def blockers(self) -> list[str]:
        accepted = self.interpretation.accepted_interpretation
        if accepted is None:
            return [*self.interpretation.blockers, self._format(_GATED)]
        missing = [] if accepted.features else [self._format(_NO_FEATURES)]
        return missing + self.output.blockers

    def feedback(self) -> list[ContentBlock]:
        blocks: list[ContentBlock] = []
        if self.interpretation.source_digest() != self._interpretation_reported:
            blocks += self.interpretation.feedback()
            # Validation fills derived fields, so read the digest afterwards.
            self._interpretation_reported = self.interpretation.source_digest()
        accepted = self.interpretation.accepted_interpretation
        if accepted is None:
            blocks.append(create_text_block("[Build skipped]\n" + self._format(_GATED)))
        else:
            self.interpretation_ready = True
            built_on = self.output.interpretation
            if built_on is None or _build_inputs(built_on) != _build_inputs(accepted):
                self.output.reset()
                self._output_reported = None
            self.output.interpretation = accepted
            if self.output.source_digest() != self._output_reported:
                blocks += self.output.feedback()
                self._output_reported = self.output.source_digest()
        return [*blocks, create_text_block(self._pending())]

    def _pending(self) -> str:
        items = (
            [
                self._format(
                    "{interpretation} must validate; {program} is built after that."
                )
            ]
            if self.interpretation.accepted_interpretation is None
            else self.blockers
        )
        items.append(
            "TicketAnswers.responses: one answer for each open ticket: "
            + (", ".join(self.ticket_ids) or "none")
        )
        return "[Pending answer requirements]\n" + "".join(f"- {i}\n" for i in items)

    def _format(self, text: str) -> str:
        return text.format(
            program=self.output.source_filename,
            interpretation=self.interpretation.source_filename,
        )
