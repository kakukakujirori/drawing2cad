from collections.abc import Callable, Iterator, Sequence
from typing import Any, cast
from unittest.mock import Mock

import httpx
import pytest
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool, tool

from zeroshot.pipeline.messages import PromptTemplate
from zeroshot.pipeline.tools import ToolFeedbackError
from zeroshot.pipeline.workflow import AgentSpec, StopReason, create_agent_subgraph

PROMPT_CONTEXT = {
    "output_path": "/work/model.py",
    "verification_dir": "/work/attempts",
}


@tool("echo")
def echo(value: str) -> str:
    """Return the supplied value."""
    return value


@tool("verify_output")
def verify_output() -> str:
    """Record one submitted candidate."""
    return "checked"


def _model(
    call: Callable[[LanguageModelInput], AIMessage],
) -> tuple[BaseChatModel, Mock]:
    model_mock = Mock(spec=BaseChatModel)
    model_mock.bind_tools.return_value = RunnableLambda(call)
    return cast(BaseChatModel, model_mock), model_mock


def _scripted_model(
    responses: Sequence[AIMessage],
) -> tuple[BaseChatModel, Mock, list[list[BaseMessage]]]:
    pending: Iterator[AIMessage] = iter(responses)
    seen: list[list[BaseMessage]] = []

    def call(model_input: LanguageModelInput) -> AIMessage:
        assert isinstance(model_input, list)
        assert all(isinstance(message, BaseMessage) for message in model_input)
        seen.append(cast(list[BaseMessage], model_input))
        return next(pending)

    model, model_mock = _model(call)
    return model, model_mock, seen


def _agent(model: BaseChatModel, **overrides: Any) -> AgentSpec:
    return AgentSpec(role="coder", model=model, **overrides)


def _subgraph(
    model: BaseChatModel,
    tools: Sequence[BaseTool] = (echo,),
    **agent_options: Any,
):
    return create_agent_subgraph(
        spec=_agent(model, **agent_options),
        tools=tools,
        prompt_context=PROMPT_CONTEXT,
    )


def test_agent_spec_uses_its_role_as_the_prompt() -> None:
    model, _ = _model(lambda _: AIMessage(content="done"))

    spec = AgentSpec(role="coder", model=model)

    assert spec.prompt == PromptTemplate("coder")
    assert spec.max_turns == 30
    assert spec.announce_turn_budget is True


@pytest.mark.parametrize("role", ["", "roles/coder"])
def test_agent_spec_rejects_an_invalid_role(role: str) -> None:
    model, _ = _model(lambda _: AIMessage(content="done"))

    with pytest.raises(ValueError, match="invalid agent role"):
        AgentSpec(role=role, model=model)


def test_agent_spec_rejects_a_budget_below_one() -> None:
    model, _ = _model(lambda _: AIMessage(content="done"))

    with pytest.raises(ValueError, match="max_turns must be at least 1"):
        AgentSpec(role="coder", model=model, max_turns=0)


def test_agent_returns_its_complete_tool_transcript() -> None:
    model, model_mock, seen = _scripted_model(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "echo",
                        "args": {"value": "hello"},
                        "id": "call-echo",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    task = HumanMessage(content="Use the echo tool")

    result = _subgraph(model, announce_turn_budget=False).invoke({"task": [task]})

    messages = result["messages"]
    assert [type(message) for message in messages] == [
        SystemMessage,
        HumanMessage,
        AIMessage,
        ToolMessage,
        AIMessage,
    ]
    assert messages[0].text == PromptTemplate("coder").render(**PROMPT_CONTEXT)
    assert messages[1] is task
    tool_result = messages[3]
    assert isinstance(tool_result, ToolMessage)
    assert tool_result.tool_call_id == "call-echo"
    assert tool_result.text == "hello"
    assert messages[-1].text == "done"
    assert result["turns"] == 2
    assert result["stop_reason"] is StopReason.COMPLETED

    assert [len(model_input) for model_input in seen] == [2, 4]
    assert seen[1][-1] is tool_result
    model_mock.bind_tools.assert_called_once()
    assert [bound.name for bound in model_mock.bind_tools.call_args.args[0]] == ["echo"]


def test_agent_stops_at_its_turn_budget_and_retains_every_notice() -> None:
    seen: list[list[BaseMessage]] = []
    turns = {"count": 0}

    def call(model_input: LanguageModelInput) -> AIMessage:
        assert isinstance(model_input, list)
        seen.append(cast(list[BaseMessage], model_input))
        turns["count"] += 1
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "verify_output",
                    "args": {},
                    "id": f"call-{turns['count']}",
                    "type": "tool_call",
                }
            ],
        )

    model, _ = _model(call)
    result = _subgraph(model, tools=(verify_output,), max_turns=3).invoke(
        {"task": [HumanMessage(content="go")]}
    )

    assert result["turns"] == 3
    assert result["stop_reason"] is StopReason.BUDGET_EXHAUSTED
    assert [model_input[-1].text for model_input in seen] == [
        f"[turn {turn}/3; candidates submitted: {turn - 1}]" for turn in (1, 2, 3)
    ]
    notices = [
        message.text
        for message in result["messages"]
        if isinstance(message, HumanMessage) and message.text.startswith("[turn ")
    ]
    assert notices == [
        f"[turn {turn}/3; candidates submitted: {turn - 1}]" for turn in (1, 2, 3)
    ]
    assert sum(isinstance(message, ToolMessage) for message in result["messages"]) == 2


def test_agent_can_hide_its_turn_budget() -> None:
    model, _, seen = _scripted_model([AIMessage(content="done")])

    result = _subgraph(model, announce_turn_budget=False).invoke(
        {"task": [HumanMessage(content="go")]}
    )

    assert not [
        message
        for message in seen[0]
        if isinstance(message, HumanMessage) and message.text.startswith("[turn ")
    ]
    assert not [
        message
        for message in result["messages"]
        if isinstance(message, HumanMessage) and message.text.startswith("[turn ")
    ]


def _flaky_model(
    errors: Sequence[Exception],
) -> tuple[BaseChatModel, list[AIMessage], list[int]]:
    pending = iter(errors)
    answers: list[AIMessage] = []
    calls: list[int] = []

    def call(_: LanguageModelInput) -> AIMessage:
        calls.append(len(calls) + 1)
        error = next(pending, None)
        if error is not None:
            raise error
        answers.append(AIMessage(content="done"))
        return answers[-1]

    model, _ = _model(call)
    return model, answers, calls


def test_agent_retries_a_dropped_connection() -> None:
    model, answers, calls = _flaky_model(
        [httpx.RemoteProtocolError("peer closed connection")]
    )
    graph = create_agent_subgraph(
        spec=_agent(model),
        tools=(echo,),
        prompt_context=PROMPT_CONTEXT,
        model_retries=2,
    )

    result = graph.invoke({"task": [HumanMessage(content="go")]})

    assert calls == [1, 2]
    assert [answer.text for answer in answers] == ["done"]
    assert result["turns"] == 1
    assert result["stop_reason"] is StopReason.COMPLETED


@pytest.mark.parametrize(
    "error",
    [httpx.ReadTimeout("too slow"), ValueError("malformed request")],
)
def test_agent_does_not_retry_an_ineligible_failure(error: Exception) -> None:
    model, _, calls = _flaky_model([error])
    graph = create_agent_subgraph(
        spec=_agent(model),
        tools=(echo,),
        prompt_context=PROMPT_CONTEXT,
        model_retries=2,
    )

    with pytest.raises(type(error), match=str(error)):
        graph.invoke({"task": [HumanMessage(content="go")]})

    assert calls == [1]


def test_agent_retries_are_bounded() -> None:
    drops = [httpx.RemoteProtocolError("peer closed connection") for _ in range(3)]
    model, _, calls = _flaky_model(drops)
    graph = create_agent_subgraph(
        spec=_agent(model),
        tools=(echo,),
        prompt_context=PROMPT_CONTEXT,
        model_retries=1,
    )

    with pytest.raises(httpx.RemoteProtocolError):
        graph.invoke({"task": [HumanMessage(content="go")]})

    assert calls == [1, 2]


def test_agent_rejects_a_negative_retry_count() -> None:
    model, _, _ = _scripted_model([AIMessage(content="done")])

    with pytest.raises(ValueError, match="model_retries must not be negative"):
        create_agent_subgraph(
            spec=_agent(model),
            tools=(echo,),
            prompt_context=PROMPT_CONTEXT,
            model_retries=-1,
        )


def _agent_calling(tool_name: str) -> BaseChatModel:
    responses: Iterator[AIMessage] = iter(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": tool_name,
                        "args": {},
                        "id": "call-once",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    model, _ = _model(lambda _: next(responses))
    return model


def _broken_tool(error: Exception) -> BaseTool:
    @tool("broken")
    def broken() -> str:
        """Raise the configured failure."""
        raise error

    return broken


def test_agent_does_not_turn_a_tool_defect_into_model_feedback() -> None:
    graph = _subgraph(
        _agent_calling("broken"),
        tools=(_broken_tool(ValueError("internal invariant broken")),),
        announce_turn_budget=False,
    )

    with pytest.raises(ValueError, match="internal invariant broken"):
        graph.invoke({"task": [HumanMessage(content="go")]})


def test_agent_forwards_a_correctable_tool_error_as_feedback() -> None:
    graph = _subgraph(
        _agent_calling("broken"),
        tools=(_broken_tool(ToolFeedbackError("that is not an image")),),
        announce_turn_budget=False,
    )

    result = graph.invoke({"task": [HumanMessage(content="go")]})

    tool_messages = [
        message for message in result["messages"] if isinstance(message, ToolMessage)
    ]
    assert len(tool_messages) == 1
    assert tool_messages[0].status == "error"
    assert "that is not an image" in tool_messages[0].text
