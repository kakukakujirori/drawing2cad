"""Prompt the coder to test hypotheses after a completed tool batch."""

from collections.abc import Iterable
from typing import Any, override

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime

from zeroshot.pipeline.workflow.middleware.turn_budget import TurnBudgetState

_REMINDER_NAME = "coding_trial_reminder"
_REMINDER = (
    "If a geometric hypothesis is still unresolved, state the hypothesis, minimal "
    "change and deciding view, then build, render and inspect the latest images "
    "before deciding. Reconsider the same question only after a new measurement "
    "or trial."
)


class CodingTrialMiddleware(AgentMiddleware[TurnBudgetState, None, Any]):
    """Give one short reminder per tool batch, independently of model.py writes."""

    def __init__(self, tool_names: Iterable[str], max_turns: int) -> None:
        super().__init__()
        self.tool_names = frozenset(tool_names)
        self.max_turns = max_turns

    @override
    def before_model(
        self, state: TurnBudgetState, runtime: Runtime[None]
    ) -> dict[str, Any] | None:
        del runtime
        # No old-batch reminder on entry, or new trial on the answer-only turn.
        if not 0 < state.get("current_turn", 0) < self.max_turns - 1:
            return None

        # Inspect only the latest response and its results; the notice itself
        # marks an already-reminded batch, including after a checkpoint resume.
        answered: set[str] = set()
        for message in reversed(state["messages"]):
            if isinstance(message, HumanMessage) and message.name == _REMINDER_NAME:
                return None
            if isinstance(message, ToolMessage):
                answered.add(message.tool_call_id)
            elif isinstance(message, AIMessage):
                calls = message.tool_calls
                if not any(call["name"] in self.tool_names for call in calls):
                    return None
                if not all(call["id"] in answered for call in calls):
                    return None
                return {
                    "messages": [HumanMessage(content=_REMINDER, name=_REMINDER_NAME)]
                }
        return None
