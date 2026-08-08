"""One agent, configured: what Hydra points at, and the policies it must carry.

`create_agent` supplies the ReAct loop itself.  What is left here is the part a
config cannot express on its own: the turn budget -- whose announcement and
whose enforcement have to agree on one number -- and the tool error policy that
decides which failures the model gets to see.  The graph owns topology; this
owns a single agent's conduct.
"""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import NotRequired

import httpx
from langchain.agents import create_agent as _create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRetryMiddleware,
    ToolCallRequest,
    ToolErrorMiddleware,
    hook_config,
)
from langchain.agents.structured_output import ResponseFormat
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool

from zeroshot.pipeline.messages import PromptTemplate
from zeroshot.pipeline.tools import ToolFeedbackError
from zeroshot.pipeline.workflow.state import StopReason


class AgentProgress(AgentState):
    """What an agent reports about its own run, under its own namespace.

    The workflow keeps one run-level stop reason, which with several agents
    would not say which of them ran out.
    """

    turns: NotRequired[int]
    stop_reason: NotRequired[StopReason]


class TurnBudget(AgentMiddleware):
    """A ceiling on model calls, and the agent's only view of it.

    Announcing and enforcing the budget are one object because they must agree
    on one number: an agent told it has 30 turns and stopped at 20 has been
    lied to.  The notice stays in the transcript so the agent reads a rate
    rather than a position: at turn 19 the earlier markers are what say how fast
    it got there.
    """

    state_schema = AgentProgress

    def __init__(self, max_turns: int, *, announce: bool = True) -> None:
        super().__init__()
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        self.max_turns = max_turns
        self.announce = announce

    @staticmethod
    def _turns_taken(messages: Sequence[BaseMessage]) -> int:
        """Model calls so far, which the transcript records one AIMessage each."""
        return sum(isinstance(message, AIMessage) for message in messages)

    def before_model(self, state, runtime) -> dict | None:
        """The output is used for State update."""
        del runtime
        if not self.announce:
            return None
        messages = state["messages"]
        submitted = sum(
            call["name"] == "verify_output"
            for message in messages
            if isinstance(message, AIMessage)
            for call in message.tool_calls
        )
        return {
            "messages": [
                HumanMessage(
                    content=(
                        f"[turn {self._turns_taken(messages) + 1}/{self.max_turns}; "
                        f"candidates submitted: {submitted}]"
                    )
                )
            ]
        }

    @hook_config(can_jump_to=["end"])
    def after_model(self, state, runtime) -> dict | None:
        """The output is generally used for State update.
        However, since the current output key doesn't contain 'messages',
        the main purpose here is not state update but
            1. Jump to END when budget is exhausted;
            2. Inform turns and stop_reason after the whole graph invoke.
        """
        del runtime
        messages = state["messages"]
        last_message = messages[-1]
        turns = self._turns_taken(messages)
        wants_more = isinstance(last_message, AIMessage) and bool(
            last_message.tool_calls
        )
        if not wants_more:
            return {"turns": turns, "stop_reason": StopReason.COMPLETED}
        if turns >= self.max_turns:
            return {
                "turns": turns,
                "stop_reason": StopReason.BUDGET_EXHAUSTED,
                "jump_to": "end",
            }
        return {"turns": turns}


def _handle_tool_error(exception: Exception, request: ToolCallRequest) -> str:
    """Hand back what the model can fix; let a pipeline defect end the run."""
    del request
    if isinstance(exception, ToolFeedbackError):
        return str(exception)
    raise exception


def create_agent(
    role: str,
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    prompt_context: Mapping[str, str] = MappingProxyType({}),
    response_format: ResponseFormat | None = None,
    max_turns: int = 30,
    announce_turn_budget: bool = True,
    model_retries: int = 5,
):
    """An agent, made runnable: `role` says who, the rest is the environment.

    Runs until the model stops asking for tools or the turn budget is spent,
    and hands its whole transcript back.  Deciding what to do with that -- to
    verify it, to feed it to a next role, to stop -- belongs to the graph that
    embeds this.

    `response_format` is passed through as an explicit strategy rather than a
    bare schema: auto-detection reads the model's profile, which would let a
    change of backend silently change when the agent is allowed to stop.
    """
    # Rendered once, here, so a placeholder we forgot to supply fails while the
    # graph is being built rather than partway into a run.
    rendered_prompt = PromptTemplate(role).render(**prompt_context)

    return _create_agent(
        model=model,
        tools=list(tools),
        system_prompt=rendered_prompt,
        response_format=response_format,
        middleware=[
            ToolErrorMiddleware(on_error=_handle_tool_error),
            ModelRetryMiddleware(
                max_retries=model_retries,
                retry_on=(httpx.NetworkError, httpx.ProtocolError),
                on_failure="error",
            ),
            TurnBudget(max_turns, announce=announce_turn_budget),
        ],
        name=role,
    )
