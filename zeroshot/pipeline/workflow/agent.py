"""Extend the ReAct loop with structured outputs, turn budgeting, and other utilities."""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Literal, NotRequired, cast

from langchain.agents import AgentState as _AgentState
from langchain.agents import create_agent as _create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ToolCallRequest,
    ToolErrorMiddleware,
)
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.types import Checkpointer
from pydantic import BaseModel

from zeroshot.pipeline.messages import system_prompt_text
from zeroshot.pipeline.tools import ToolFeedbackError
from zeroshot.pipeline.workflow.middleware import (
    ModelCallRetryMiddleware,
    PromptLogMiddleware,
    StatelessReasoningMiddleware,
    StopReason,
    TurnBudgetMiddleware,
)


class AgentState(_AgentState[Any]):
    """Langchain official AgentState is extended:

    Fields:
        - messages: list[AnyMessage]
        - jump_to: JumpTo | None
        - structured_response: BaseModel
        - current_turn: int
        - total_turns: int
        - stop_reason: StopReason

    The first three are included in _AgentState.
    """

    current_turn: NotRequired[int]
    total_turns: NotRequired[int]
    stop_reason: NotRequired[StopReason | None]


def _build_response_format(
    output_schema: type[BaseModel] | None,
    response_format_strategy: Literal["provider", "tool"],
) -> ProviderStrategy[BaseModel] | ToolStrategy[BaseModel] | None:
    if output_schema is None:
        return None
    if response_format_strategy == "provider":
        return ProviderStrategy(schema=output_schema)
    elif response_format_strategy == "tool":
        return ToolStrategy(schema=output_schema)
    else:
        raise ValueError(
            f"Unknown response format strategy: {response_format_strategy}"
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
    response_format_strategy: Literal["provider", "tool"] = "provider",
    max_turns: int = 30,
    announce_turns: bool = True,
    reset_turns_when_reentrant: bool = True,
    model_retries: int = 5,
    checkpointer: Checkpointer = False,
    extra_middleware: Sequence[AgentMiddleware[Any, None, Any]] = (),
):

    middleware = cast(
        Sequence[AgentMiddleware[AgentState, None, Any]],
        [
            ToolErrorMiddleware(on_error=_handle_tool_error),
            # Ahead of the retry so a prompt is reported once per model node
            # rather than once per attempt: the first handler in this list is
            # the outermost layer around the model call.
            PromptLogMiddleware(role),
            ModelCallRetryMiddleware(max_retries=model_retries, role=role),
            TurnBudgetMiddleware(
                max_turns,
                announce_turns=announce_turns,
                reset_turns_when_reentrant=reset_turns_when_reentrant,
            ),
            # Innermost, so it sees every message the layers above appended and
            # sanitises each retry attempt rather than only the first.
            StatelessReasoningMiddleware(),
            *extra_middleware,
        ],
    )

    # Inform max_turns
    prompt_context = {"max_turns": str(max_turns), **prompt_context}

    agent = _create_agent(
        model=model,
        tools=list(tools),
        system_prompt=system_prompt_text(role, prompt_context, output_schema),
        response_format=_build_response_format(output_schema, response_format_strategy),
        state_schema=AgentState,
        middleware=middleware,
        name=role,
        checkpointer=checkpointer,
    )
    # One tool-bearing turn traverses before_model -> model -> after_model ->
    # tools, plus a node per extra middleware; the headroom covers the agent
    # hooks and the before_model that declares the budget spent.
    steps_per_turn = 4 + len(extra_middleware)
    return agent.with_config(recursion_limit=steps_per_turn * max_turns + 6)
