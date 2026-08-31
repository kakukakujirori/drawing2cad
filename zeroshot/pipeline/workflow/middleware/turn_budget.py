"""How many model turns an agent may spend, and what it is told about them."""

from enum import Enum
from typing import Any, NotRequired, override

from langchain.agents import AgentState as _AgentState
from langchain.agents.middleware import AgentMiddleware, hook_config
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


class TurnBudgetMiddleware(AgentMiddleware[TurnBudgetState, None, Any]):
    """Turn budget middleware.

    This middleware enforces a turn budget for the agent.
    """

    state_schema = TurnBudgetState

    def __init__(
        self,
        max_turns: int,
        announce_turns: bool = True,
        reset_turns_when_reentrant: bool = True,
    ) -> None:
        super().__init__()
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        self.max_turns = max_turns
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
        # `current_turn > 0` checks that this is 're'entrant
        if self.reset_turns_when_reentrant and state.get("current_turn", 0) > 0:
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
                f"{countdown} Final turn: a tool call made now still runs, but "
                "nothing comes back to you and no answer follows it. Answer "
                "from what you have, and say in the answer what remains "
                "uncertain."
            )
        if turn == self.max_turns - 1:
            return (
                f"{countdown} One turn remains after this one, and you will not "
                "see what its tool calls return. Run or write now whatever your "
                "answer still depends on."
            )
        return countdown

    @override
    def after_model(
        self, state: TurnBudgetState, runtime: Runtime[None]
    ) -> dict[str, Any]:
        """Count the turn."""
        del runtime

        last_message = state["messages"][-1]
        is_tool_call = isinstance(last_message, AIMessage) and bool(
            last_message.tool_calls
        )

        progress: dict[str, Any] = {
            "current_turn": state.get("current_turn", 0) + 1,
            "total_turns": state.get("total_turns", 0) + 1,
        }
        if is_tool_call:
            return progress
        return progress | {"stop_reason": StopReason.COMPLETED}
