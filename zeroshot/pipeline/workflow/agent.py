"""One agent, configured: what Hydra points at, and the policies it must carry.

`create_agent` supplies the ReAct loop itself.  What is left here is the part a
config cannot express on its own: the turn budget -- whose announcement and
whose enforcement have to agree on one number -- and the tool error policy that
decides which failures the model gets to see.  The graph owns topology; this
owns a single agent's conduct.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, NotRequired

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
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from zeroshot.pipeline.messages import PromptTemplate
from zeroshot.pipeline.tools import ToolFeedbackError
from zeroshot.pipeline.workflow.state import StopReason

# A config binds who an agent is -- role, model, budget -- and leaves the
# environment open.  Whoever builds the graph supplies the rest.
AgentFactory = Callable[..., Any]

StructuredOutput = Literal["provider", "tool"]

_STRATEGIES = {"provider": ProviderStrategy, "tool": ToolStrategy}


class AgentProgress(AgentState):
    """What an agent reports about its own run, under its own namespace.

    The workflow keeps one run-level stop reason, which with several agents
    would not say which of them ran out.
    """

    turns: NotRequired[int]
    stop_reason: NotRequired[StopReason]


class TurnBudget(AgentMiddleware):
    state_schema = AgentProgress

    _NOTICE_KEY = "turn_notice"
    """Marks a notice this wrote. Not sent to the provider; `strip_notices` reads it."""

    def __init__(self, max_turns: int, *, announce: bool = True) -> None:
        super().__init__()
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        self.max_turns = max_turns
        self.announce = announce

    @classmethod
    def strip_notices(cls, messages: Sequence[BaseMessage]) -> list[BaseMessage]:
        """The transcript without the notices addressed to whoever ran it.

        A workflow that hands one agent's messages to the next uses this: the
        notices are a true record of what that agent was shown, but a later
        agent reading `turn 27/30` would be told it is nearly out of a budget
        it never spent.
        """
        return [
            message
            for message in messages
            if not message.additional_kwargs.get(cls._NOTICE_KEY)
        ]

    def before_model(self, state, runtime) -> dict | None:
        """Open the turn with what this agent has spent so far.

        Turns are all this counts.  What an agent has produced with them is
        already in front of it -- a verification result carries its attempt
        number, a tool result carries what it found -- and a budget that also
        tallied those would be reporting on tools it has no business knowing.
        """
        del runtime
        if not self.announce:
            return None
        return {
            "messages": [
                HumanMessage(
                    content=f"[turn {state.get('turns', 0) + 1}/{self.max_turns}]",
                    additional_kwargs={self._NOTICE_KEY: True},
                )
            ]
        }

    @hook_config(can_jump_to=["end"])
    def after_model(self, state, runtime) -> dict:
        """Count the turn just spent, and end the run if it was the last.

        Counted rather than read back off the transcript: an agent that starts
        on an earlier stage's messages would otherwise be charged for the turns
        that produced them.

        Ending here rather than after the tool round leaves the unanswered tool
        calls as the final message, which is the only difference between an
        agent that ran out and one that decided it was done.
        """
        del runtime
        answer = state["messages"][-1]
        tool_calls = answer.tool_calls if isinstance(answer, AIMessage) else []
        progress: dict = {"turns": state.get("turns", 0) + 1}
        if not tool_calls:
            return progress | {"stop_reason": StopReason.COMPLETED}
        if progress["turns"] >= self.max_turns:
            return progress | {
                "stop_reason": StopReason.BUDGET_EXHAUSTED,
                "jump_to": "end",
            }
        return progress


@dataclass(frozen=True)
class AgentRun[T: BaseModel]:
    """What one agent did when the workflow handed it the run so far.

    `messages` is only what this agent added, so a workflow that appends it to
    a shared transcript records each message once however many agents read it.
    """

    messages: list[BaseMessage]
    answer: T | None
    turns: int
    stop_reason: StopReason
    role: str

    @property
    def stop_reasons(self) -> dict[str, StopReason]:
        """How this agent finished, under the name of the role that finished."""
        return {self.role: self.stop_reason}

    @property
    def turns_by_role(self) -> dict[str, int]:
        """What this agent spent, under the name of the role that spent it.

        What a workflow records: with several agents in a run, a bare number is
        one of them and does not say which.
        """
        return {self.role: self.turns}


def run_agent[T: BaseModel](
    agent,
    history: Sequence[BaseMessage],
    instruction: HumanMessage,
    expects: type[T] | None = None,
) -> AgentRun[T]:
    """Give one agent the run so far plus what it is being asked for now.

    `expects` is the typed answer the role owes.  An agent that spent its
    budget mid-investigation has none to give, which the caller routes around;
    one that finished and produced nothing of that shape has broken its
    contract, and the run stops here rather than at whoever reads the artifact.
    """
    context = TurnBudget.strip_notices(history)
    result = agent.invoke({"messages": [*context, instruction]})

    inherited = {message.id for message in context}
    answer = result.get("structured_response")
    stop_reason = result["stop_reason"]
    if expects is not None and not isinstance(answer, expects):
        if stop_reason is not StopReason.BUDGET_EXHAUSTED:
            raise ValueError(
                f"{agent.name} finished without a valid {expects.__name__} "
                f"(stop reason: {stop_reason})"
            )
        answer = None

    return AgentRun(
        messages=[m for m in result["messages"] if m.id not in inherited],
        answer=answer,
        turns=result["turns"],
        stop_reason=stop_reason,
        role=agent.name,
    )


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
    output_schema: type[BaseModel] | None = None,
    structured_output: StructuredOutput = "provider",
    max_turns: int = 30,
    announce_turn_budget: bool = True,
    model_retries: int = 5,
):
    """An agent, made runnable: `role` says who, the rest is the environment.

    Runs until the model stops asking for tools or the turn budget is spent,
    and hands its whole transcript back.  Deciding what to do with that -- to
    verify it, to feed it to a next role, to stop -- belongs to the graph that
    embeds this.

    `output_schema` is the typed answer the role owes, and reaches the model
    twice from this one definition: as the provider's output contract, and as
    the `$output_schema` its prompt explains.  A role prompt that spelled the
    schema out by hand would be a second definition to keep in step.

    `structured_output` names the strategy rather than letting it be inferred:
    auto-detection reads the model's profile, which would let a change of
    backend silently change when the agent is allowed to stop.
    """
    context = dict(prompt_context)
    response_format = None
    if output_schema is not None:
        context["output_schema"] = json.dumps(
            output_schema.model_json_schema(), indent=2
        )
        response_format = _STRATEGIES[structured_output](schema=output_schema)

    # Rendered once, here, so a placeholder we forgot to supply fails while the
    # graph is being built rather than partway into a run.
    rendered_prompt = PromptTemplate(f"roles/{role}").render(**context)

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
