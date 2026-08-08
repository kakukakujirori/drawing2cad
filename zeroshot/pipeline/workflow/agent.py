"""One agent, run to exhaustion: the smallest subgraph template."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from types import MappingProxyType

import httpx
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from zeroshot.pipeline.messages import PromptTemplate
from zeroshot.pipeline.tools import ToolFeedbackError
from zeroshot.pipeline.workflow.state import AgentState, StopReason


@dataclass(frozen=True)
class AgentSpec:
    role: str
    model: BaseChatModel
    prompt: PromptTemplate = field(init=False)
    max_turns: int = 30
    announce_turn_budget: bool = True

    def __post_init__(self) -> None:
        if not self.role or "/" in self.role:
            raise ValueError(f"invalid agent role: {self.role!r}")
        if self.max_turns < 1:
            raise ValueError("max_turns must be at least 1")

        object.__setattr__(self, "prompt", PromptTemplate(self.role))


def _budget_notice(
    messages: Sequence[BaseMessage],
    turn: int,
    budget: int,
) -> HumanMessage:
    """Where the agent stands in a budget it cannot otherwise observe.

    It stays in the transcript so the agent reads a rate rather than a
    position: at turn 19 the earlier markers are what say how fast it got there.
    """

    submitted = sum(
        call["name"] == "verify_output"
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    )
    return HumanMessage(
        content=f"[turn {turn}/{budget}; candidates submitted: {submitted}]"
    )


def create_agent_subgraph(
    spec: AgentSpec,
    tools: Sequence[BaseTool],
    prompt_context: Mapping[str, str] = MappingProxyType({}),
    model_retries: int = 5,
):
    """An agent, made runnable: `spec` says who, the rest is the environment.

    Runs until the model stops asking for tools or the turn budget is spent,
    and hands its whole transcript back.  Deciding what to do with that -- to
    verify it, to feed it to a next role, to stop -- belongs to the graph that
    embeds this.

    The node names `agent` and `tools` are part of the contract: the event log
    reads them to tell a tool the model called from one the workflow called.
    """
    if model_retries < 0:
        raise ValueError("model_retries must not be negative")

    # Rendered once, here, so a placeholder we forgot to supply fails while the
    # graph is being built rather than partway into a run.
    rendered_prompt = spec.prompt.render(**prompt_context)

    def seed(state: AgentState):
        return {"messages": [SystemMessage(content=rendered_prompt), *state["task"]]}

    def should_continue(state: AgentState):
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            raise TypeError("agent node must append an AIMessage before routing")
        if last_message.tool_calls and state["turns"] < spec.max_turns:
            return "tools"
        return "finish"

    def finish(state: AgentState):
        """Say why this agent stopped, which only it is in a position to know.

        It reads the same last turn `should_continue` just routed on: still
        asking for tools means the budget ran out under it, and anything else
        means it decided it was done.
        """
        last_message = state["messages"][-1]
        return {
            "stop_reason": (
                StopReason.BUDGET_EXHAUSTED
                if isinstance(last_message, AIMessage) and last_message.tool_calls
                else StopReason.COMPLETED
            )
        }

    def call_agent(
        state: AgentState,
        agent: Runnable[LanguageModelInput, AIMessage],
    ):
        turn = state.get("turns", 0) + 1
        history = state["messages"]
        notice: list[BaseMessage] = (
            [_budget_notice(history, turn, spec.max_turns)]
            if spec.announce_turn_budget
            else []
        )
        response = agent.invoke([*history, *notice])
        return {"messages": [*notice, response], "turns": turn}

    # The client's own `max_retries` decides from the response status, so a
    # connection that dies mid-body after a 200 is already past it. Timeouts are
    # excluded: they are the one transport failure where waiting again costs the
    # whole request budget rather than a moment.
    agent = spec.model.bind_tools(list(tools)).with_retry(
        retry_if_exception_type=(httpx.NetworkError, httpx.ProtocolError),
        stop_after_attempt=model_retries + 1,
    )

    workflow = StateGraph(state_schema=AgentState)  # type: ignore[type-var]
    workflow.add_node("seed", seed)
    workflow.add_node("agent", partial(call_agent, agent=agent))
    workflow.add_node(
        "tools", ToolNode(list(tools), handle_tool_errors=ToolFeedbackError)
    )

    workflow.add_node("finish", finish)

    workflow.add_edge(START, "seed")
    workflow.add_edge("seed", "agent")
    workflow.add_conditional_edges("agent", should_continue, ["tools", "finish"])
    workflow.add_edge("tools", "agent")
    workflow.add_edge("finish", END)

    return workflow.compile()
