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
        """Reset progress that belongs to this invocation."""
        del runtime
        update: dict[str, Any] = {
            "structured_response": None,
            "stop_reason": None,
        }
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
        """Check turns, and add turn count to messages."""
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
        """Announce the turn, and near the end say what running out costs.

        A budget that only ever reports a number leaves the agent to find the
        edge by going over it: the tool call it spends its last turn on is
        dropped along with the work that call was meant to settle, and a stage
        that never reaches an answer ends the run holding nothing. The warning
        arrives one turn early because the turn that hears it is the last one
        that can still act on it.
        """
        countdown = f"[turn {turn}/{self.max_turns}]"
        if turn >= self.max_turns:
            return (
                f"{countdown} Final turn: a tool call made now is dropped, so "
                "nothing you have not already run or written can still take "
                "effect. Answer from what you have, and say in the answer what "
                "remains uncertain."
            )
        if turn == self.max_turns - 1:
            return (
                f"{countdown} One turn remains after this one, and a tool call "
                "made on it is dropped. Run or write now whatever your answer "
                "still depends on."
            )
        return countdown

    @hook_config(can_jump_to=["end"])
    @override
    def after_model(
        self, state: TurnBudgetState, runtime: Runtime[None]
    ) -> dict[str, Any]:
        """Increment turns."""
        del runtime

        last_message = state["messages"][-1]
        tool_calls = (
            last_message.tool_calls if isinstance(last_message, AIMessage) else []
        )

        progress = {
            "current_turn": state.get("current_turn", 0) + 1,
            "total_turns": state.get("total_turns", 0) + 1,
        }
        if not tool_calls:
            return progress | {"stop_reason": StopReason.COMPLETED}
        if progress["current_turn"] >= self.max_turns:
            return progress | {
                "jump_to": "end",
                "stop_reason": StopReason.BUDGET_EXHAUSTED,
            }
        return progress
