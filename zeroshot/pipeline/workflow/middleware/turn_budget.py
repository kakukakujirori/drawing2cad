"""How many model turns an agent may spend, and what it is told about them."""

from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any, NotRequired, cast, override

from langchain.agents import AgentState as _AgentState
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    hook_config,
)
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime


class StopReason(Enum):
    COMPLETED = "COMPLETED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"


class TurnBudgetState(_AgentState[Any]):
    """State fields owned and required by TurnBudgetMiddleware."""

    current_turn: NotRequired[int]
    total_turns: NotRequired[int]
    stop_reason: NotRequired[StopReason | None]
    # Set by the caller when it re-asks for an answer it refused.
    retry: NotRequired[bool]


class TurnBudgetMiddleware(AgentMiddleware[TurnBudgetState, None, Any]):
    """Turn budget middleware.

    This middleware enforces a turn budget for the agent.
    """

    state_schema = TurnBudgetState

    def __init__(
        self,
        max_turns: int,
        retry_turns: int | None = None,
        announce_turns: bool = True,
        reset_turns_when_reentrant: bool = True,
    ) -> None:
        super().__init__()
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        if retry_turns is not None and retry_turns < 1:
            raise ValueError("retry_turns must be at least 1")
        self.max_turns = max_turns
        self.retry_turns = retry_turns
        self.announce_turns = announce_turns
        self.reset_turns_when_reentrant = reset_turns_when_reentrant

    @override
    def before_agent(
        self, state: TurnBudgetState, runtime: Runtime[None]
    ) -> dict[str, Any] | None:
        """Reset progress that belongs to this invocation.

        No need to set `structured_response: None`.
        Langchain keeps it out of the agent's input schema, so
        no answer survives an invocation to be reset, and writing
        one tells the agent an answer has already been given.
        """
        del runtime
        update: dict[str, Any] = {"stop_reason": None}
        turns = state.get("current_turn", 0)
        if state.get("retry") and self.retry_turns is not None:
            # A retry corrects the refused answer rather than redoing the work:
            # it keeps the count, set back only as far as leaves `retry_turns`.
            resumed = min(turns, max(0, self.max_turns - self.retry_turns))
            if resumed < turns:
                update |= {
                    "messages": [
                        HumanMessage(
                            content=(
                                f"[turn set back to {resumed}/{self.max_turns} "
                                "to correct the refused answer]"
                            )
                        )
                    ],
                    "current_turn": resumed,
                }
            return update
        # `current_turn > 0` checks that this is 're'entrant
        if self.reset_turns_when_reentrant and turns > 0:
            update |= {
                "messages": [
                    HumanMessage(content=f"[turn reset to 0/{self.max_turns}]")
                ],
                "current_turn": 0,
            }
        return update

    @hook_config(can_jump_to=["end"])
    @override
    def before_model(
        self, state: TurnBudgetState, runtime: Runtime[None]
    ) -> dict[str, Any] | None:
        """Check turns, and add turn count to messages.

        The budget-exceeded model must stop here.
        If we stop it after the model call,
        the message history cannot collect ToolMessage.
        """
        del runtime

        turns = state.get("current_turn", 0)

        if turns >= self.max_turns:
            return {
                "jump_to": "end",
                "stop_reason": StopReason.BUDGET_EXHAUSTED,
            }

        if not self.announce_turns:
            return None

        return {"messages": [HumanMessage(content=self._announcement(turns + 1))]}

    def _announcement(self, turn: int) -> str:
        """Announce the turn."""
        countdown = f"[turn {turn}/{self.max_turns}]"
        if turn >= self.max_turns:
            return (
                f"{countdown} Final turn: ordinary tools are unavailable; only "
                "the required answer or structured submission remains. Return "
                "it alone. A response cannot substitute for work missing from "
                "a required file or artifact."
            )
        if turn == self.max_turns - 1:
            return (
                f"{countdown} This is the last ordinary-tool-capable turn. Run "
                "or write now whatever your answer still depends on; the next "
                "and final turn is answer-only."
            )
        return countdown

    def _answer_only(self, request: ModelRequest[None]) -> ModelRequest[None]:
        """Take the tools away on the final turn, leaving the answer.

        The announcement states the same boundary. The answer contract is
        bound separately from the ordinary tools, so it survives.
        """
        turns = cast(TurnBudgetState, request.state).get("current_turn", 0)
        if turns < self.max_turns - 1:
            return request
        return request.override(tools=[])

    @override
    def wrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        return handler(self._answer_only(request))

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        return await handler(self._answer_only(request))

    @hook_config(can_jump_to=["model"])
    @override
    def after_model(
        self, state: TurnBudgetState, runtime: Runtime[None]
    ) -> dict[str, Any]:
        """Count the turn, and end the agent only on the model's own message."""
        del runtime

        last_ai_message = next(
            (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
            None,
        )
        is_tool_call = last_ai_message is not None and bool(last_ai_message.tool_calls)

        progress: dict[str, Any] = {
            "current_turn": state.get("current_turn", 0) + 1,
            "total_turns": state.get("total_turns", 0) + 1,
        }
        # A refusal added after a text answer is for the model to act on.
        if isinstance(state["messages"][-1], HumanMessage):
            return progress | {"jump_to": "model"}
        if is_tool_call and state.get("structured_response") is None:
            return progress
        return progress | {"stop_reason": StopReason.COMPLETED}
