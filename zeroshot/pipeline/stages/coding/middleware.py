"""Open the coder's conversation, report its workspace writes and prompt trials."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, override

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.messages.content import ContentBlock, create_text_block
from langgraph.runtime import Runtime

from zeroshot.pipeline.stages.coding.workspace import CodingWorkspaceVerifier
from zeroshot.pipeline.workflow.middleware import VerifyOnWriteMiddleware
from zeroshot.pipeline.workflow.middleware.turn_budget import TurnBudgetState

if TYPE_CHECKING:
    from zeroshot.pipeline.stages._base.prompt import StageInstructions
    from zeroshot.pipeline.workflow.state import ReconstructionState

INSTRUCTIONS_NAME = "coding_stage_instructions"
CHECKPOINT_NAME = "coding_verified_checkpoint"


class CodingMiddleware(VerifyOnWriteMiddleware):
    """Verify the workspace after each write, and open the stage with its existing build."""

    verifier: CodingWorkspaceVerifier

    def report_existing(self) -> list[ContentBlock]:
        """Report an existing model.py before the first response, without refusing it later."""
        if not self.verifier.source_path.is_file():
            return []
        report = self.verifier.feedback()
        self._last_seen = self._reported = self._digest()
        return report

    def opening(
        self,
        state: ReconstructionState,
        instructions: StageInstructions,
        instruction: HumanMessage,
        *,
        retry: bool,
    ) -> list[HumanMessage]:
        """The messages that start an invocation; a validation retry adds no report."""
        del state, instructions
        report = [] if retry else self.report_existing()
        return [instruction, *([self.report_message(report)] if report else [])]

    def report_message(self, report: list[ContentBlock]) -> HumanMessage:
        return HumanMessage(content_blocks=report)

    @override
    def before_model(self, state, runtime):
        update = super().before_model(state, runtime)
        if update is not None:
            (message,) = update["messages"]
            update["messages"] = [self.report_message(message.content_blocks)]
        return update


class FreshCodingMiddleware(CodingMiddleware):
    """Drop pre-checkpoint messages only from a model request, after valid builds."""

    def reset(self) -> None:
        super().reset()
        self._baseline_candidate: dict[str, Any] | None = None

    @override
    def opening(
        self,
        state: ReconstructionState,
        instructions: StageInstructions,
        instruction: HumanMessage,
        *,
        retry: bool,
    ) -> list[HumanMessage]:
        self._set_context(state, instructions)
        instruction = instruction.model_copy(
            update={
                "name": INSTRUCTIONS_NAME,
                "additional_kwargs": {
                    **instruction.additional_kwargs,
                    "coding_validation_error": retry,
                },
            }
        )
        return super().opening(state, instructions, instruction, retry=retry)

    def _set_context(
        self, state: ReconstructionState, instructions: StageInstructions
    ) -> None:
        history = state["reconstruction"]
        snapshot = history.snapshots[-1]
        self._inputs = instructions.create_artifact_message().content_blocks
        self._context = {
            "round": snapshot.round,
            "open_tickets": [
                ticket.model_dump(mode="json") for ticket in snapshot.open_tickets
            ],
            "current_concerns": {
                stage: report.concerns
                for stage, report in snapshot.stage_reports.items()
            },
            "previous_concerns": {
                stage: report.concerns
                for stage, report in (
                    history.snapshots[-2].stage_reports.items()
                    if len(history.snapshots) > 1
                    else ()
                )
            },
        }
        output = self.verifier.output
        self._known_names = {
            output.source_filename,
            self.verifier.interpretation.source_filename,
            output.attempt_store.root_dirname.name,
            "inputs",
            *(
                path.rsplit("/", 1)[-1]
                for key, path in instructions.prompt_context.items()
                if key.endswith("_path")
            ),
        }

    @override
    def report_message(self, feedback: list[ContentBlock]) -> HumanMessage:
        output = self.verifier.output
        source = output.accepted_source
        report = output.last_report
        interpretation = self.verifier.interpretation.accepted_interpretation
        if source is None or report is None or interpretation is None:
            return HumanMessage(content_blocks=feedback)
        candidate = {
            "verification_id": report.verification_id,
            "program": output.candidate_path(report),
            "source_sha256": output.source_digest(),
        }
        if self._baseline_candidate is None:
            self._baseline_candidate = candidate
        best = output.best_report
        entries = sorted(output.workdir.host_bind_dir.iterdir())
        auxiliary = [
            str(output.workdir.host_to_sandbox_path(path))
            + ("/" if path.is_dir() else "")
            for path in entries
            if path.name not in self._known_names and not path.is_symlink()
        ]
        context = {
            **self._context,
            "interpretation": interpretation.model_dump(mode="json"),
            "current_candidate": candidate,
            "baseline_candidate": self._baseline_candidate,
            "best_eligible_candidate": (
                {
                    "verification_id": best.verification_id,
                    "program": output.candidate_path(best),
                }
                if best is not None
                else None
            ),
            "auxiliary_workspace_entries": auxiliary[:100],
            "auxiliary_listing_truncated": len(auxiliary) > 100,
        }
        return HumanMessage(
            name=CHECKPOINT_NAME,
            additional_kwargs={"coding_candidate": candidate},
            content_blocks=[
                create_text_block(
                    "[Current verified coding checkpoint]\n"
                    "A previous coding pass produced the supplied code and verification. "
                    "Take over from this checkpoint and finish correcting the drawing mismatches. "
                    "Past AI/tool exchanges before this checkpoint are omitted "
                    "from inference; their saved files remain available. "
                    "A verified build does not establish drawing correctness.\n"
                    + json.dumps(context, separators=(",", ":"))
                ),
                *self._inputs,
                create_text_block("[Current model.py]\n```python\n" + source + "\n```"),
                *feedback,
            ],
        )

    @override
    def wrap_model_call(
        self, request: ModelRequest[None], handler: Any
    ) -> ModelResponse:
        messages = request.messages
        anchor = next(
            (
                i
                for i in range(len(messages) - 1, -1, -1)
                if isinstance(messages[i], HumanMessage)
                and messages[i].name == CHECKPOINT_NAME
            ),
            None,
        )
        if anchor is None:
            return super().wrap_model_call(request, handler)
        keep = {anchor}
        for predicate in (
            lambda message: (
                message.name == INSTRUCTIONS_NAME
                and not message.additional_kwargs.get("coding_validation_error")
            ),
            lambda message: message.name == INSTRUCTIONS_NAME,
            lambda message: message.text.startswith("[turn "),
        ):
            index = next(
                (
                    i
                    for i in range(len(messages) - 1, -1, -1)
                    if isinstance(messages[i], HumanMessage) and predicate(messages[i])
                ),
                None,
            )
            if index is not None and index < anchor:
                keep.add(index)
        prefix = [messages[i] for i in sorted(keep) if i < anchor]
        # StageInstructions' first block is the task; the checkpoint supplies
        # the input blocks once, avoiding duplicate copies of the same images.
        prefix = [
            message.model_copy(update={"content": message.content_blocks[:1]})
            if message.name == INSTRUCTIONS_NAME
            else message
            for message in prefix
        ]
        selected = prefix + messages[anchor:]
        request.runtime.stream_writer(
            {
                "prompt": {
                    "role": "coder",
                    "memory_mode": "verified_checkpoint",
                    "anchor_message_id": messages[anchor].id,
                    "candidate": messages[anchor].additional_kwargs["coding_candidate"],
                    "original_message_count": len(messages),
                    "request_message_count": len(selected),
                    "omitted_message_count": len(messages) - len(selected),
                    "retained_ai_messages": sum(
                        isinstance(m, AIMessage) for m in selected
                    ),
                    "retained_tool_messages": sum(
                        isinstance(m, ToolMessage) for m in selected
                    ),
                    "ordinary_tool_count": len(request.tools),
                    "current_turn": request.state.get("current_turn", 0),
                    "total_turns": request.state.get("total_turns", 0),
                }
            }
        )
        return super().wrap_model_call(request.override(messages=selected), handler)


_REMINDER_NAME = "coding_reminder"
_REMINDER = (
    "Rather than long speculation, quickly {action}: it gets you useful "
    "verification feedback from the pipeline."
)
_WRITE = "write a provisional {file}"
_TRY = "try any unresolved geometry as a trial"


class CodingReminderMiddleware(AgentMiddleware[TurnBudgetState, None, Any]):
    """After some turns, push the coder each turn to act rather than speculate.

    The reminder names the first file this round has not written yet; once the
    round has written both, it asks for a trial of any unresolved geometry.
    """

    def __init__(
        self,
        workspace: CodingWorkspaceVerifier,
        *,
        max_turns: int,
        remind_after_turns: int,
    ) -> None:
        super().__init__()
        if remind_after_turns < 1:
            raise ValueError(f"{remind_after_turns=} must be at least 1")
        self.workspace = workspace
        self.max_turns = max_turns
        self.remind_after_turns = remind_after_turns

    @override
    def before_model(
        self, state: TurnBudgetState, runtime: Runtime[None]
    ) -> dict[str, Any] | None:
        del runtime
        turn = state.get("current_turn", 0) + 1
        # Nothing can be written on the answer-only final turn.
        if not self.remind_after_turns < turn < self.max_turns:
            return None
        if _reminded(state["messages"]):
            return None
        file = self.workspace.unwritten_file()
        action = _TRY if file is None else _WRITE.format(file=file)
        text = _REMINDER.format(action=action)
        return {"messages": [HumanMessage(content=text, name=_REMINDER_NAME)]}


def _reminded(messages: list[AnyMessage]) -> bool:
    """Whether a reminder follows the latest response already, as on a replay."""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return False
        if isinstance(message, HumanMessage) and message.name == _REMINDER_NAME:
            return True
    return False
