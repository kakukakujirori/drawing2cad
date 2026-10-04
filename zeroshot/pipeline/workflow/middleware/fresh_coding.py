"""Give inference a verified coding checkpoint while retaining the full state."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, override

from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.content import ContentBlock, create_text_block

from zeroshot.pipeline.stages.coding.progress import CodingProgressMiddleware

if TYPE_CHECKING:
    from zeroshot.pipeline.stages._base.prompt import StageInstructions
    from zeroshot.pipeline.workflow.state import ReconstructionState

INSTRUCTIONS_NAME = "coding_stage_instructions"
CHECKPOINT_NAME = "coding_verified_checkpoint"


class FreshCodingMiddleware(CodingProgressMiddleware):
    """Drop pre-checkpoint messages only from a model request, after valid builds."""

    def reset(self) -> None:
        super().reset()
        self._baseline_candidate: dict[str, Any] | None = None

    def set_context(
        self, state: ReconstructionState, instructions: StageInstructions
    ) -> None:
        history = state["reconstruction"]
        snapshot = history.snapshots[-1]
        self._inputs = instructions.create_artifact_message().content_blocks
        self._context = {
            "round": snapshot.round,
            "interpretation": snapshot.interpretation.model_dump(mode="json"),
            "operations": snapshot.operations.model_dump(mode="json"),
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
        self._known_names = {
            self.verifier.source_filename,
            self.verifier.attempt_store.root_dirname.name,
            "inputs",
            *(
                path.rsplit("/", 1)[-1]
                for key, path in instructions.prompt_context.items()
                if key.endswith("_path")
            ),
        }

    def checkpoint_message(self, feedback: list[ContentBlock]) -> HumanMessage:
        source = self.verifier.accepted_source
        report = self.verifier._last_feedback_report
        if source is None or report is None:
            return HumanMessage(content_blocks=feedback)
        candidate = {
            "verification_id": report.verification_id,
            "program": self.verifier._candidate_path(report),
            "source_sha256": self.verifier.source_digest(),
        }
        if self._baseline_candidate is None:
            self._baseline_candidate = candidate
        best = self.verifier._best_candidate
        entries = sorted(self.verifier.workdir.host_bind_dir.iterdir())
        auxiliary = [
            str(self.verifier.workdir.host_to_sandbox_path(path))
            + ("/" if path.is_dir() else "")
            for path in entries
            if path.name not in self._known_names and not path.is_symlink()
        ]
        context = {
            **self._context,
            "current_candidate": candidate,
            "baseline_candidate": self._baseline_candidate,
            "best_eligible_candidate": (
                {
                    "verification_id": best[1].verification_id,
                    "program": self.verifier._candidate_path(best[1]),
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
    def before_model(self, state, runtime):
        update = super().before_model(state, runtime)
        if update is not None:
            message = update["messages"][0]
            update["messages"] = [self.checkpoint_message(message.content_blocks)]
        return update

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
