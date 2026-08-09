from typing import Any

import httpx
import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatResult
from langchain_core.tools import BaseTool, tool
from pydantic import PrivateAttr

from tests.zeroshot.chat_models import ScriptedChatModel, tool_call
from zeroshot.pipeline.messages import PromptTemplate
from zeroshot.pipeline.tools import ToolFeedbackError
from zeroshot.pipeline.workflow import SemanticHypothesis, StopReason
from zeroshot.pipeline.workflow.agent import create_agent

PROMPT_CONTEXT = {
    "output_path": "/work/model.py",
    "verification_dir": "/work/attempts",
}


@tool("echo")
def echo(value: str) -> str:
    """Return the supplied value."""
    return value


class _FlakyChatModel(ScriptedChatModel):
    """Fails a scripted number of times before answering."""

    errors: tuple[Any, ...] = ()

    _attempts: int = PrivateAttr(default=0)

    @property
    def attempts(self) -> int:
        return self._attempts

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._attempts += 1
        if self._attempts <= len(self.errors):
            raise self.errors[self._attempts - 1]
        return super()._generate(messages, stop, run_manager, **kwargs)


def _subgraph(
    model: ScriptedChatModel,
    tools: tuple[BaseTool, ...] = (echo,),
    **agent_options: Any,
):
    return create_agent(
        role="coder",
        model=model,
        tools=tools,
        prompt_context=PROMPT_CONTEXT,
        **agent_options,
    )


def _notices(messages: list[BaseMessage]) -> list[str]:
    return [
        message.text
        for message in messages
        if isinstance(message, HumanMessage) and message.text.startswith("[turn ")
    ]


def test_agent_returns_its_complete_tool_transcript() -> None:
    model = ScriptedChatModel(
        responses=(
            tool_call("echo", {"value": "hello"}, "call-echo"),
            AIMessage(content="done"),
        )
    )
    task = HumanMessage(content="Use the echo tool")

    result = _subgraph(model, announce_turn_budget=False).invoke({"messages": [task]})

    messages = result["messages"]
    assert [type(message) for message in messages] == [
        HumanMessage,
        AIMessage,
        ToolMessage,
        AIMessage,
    ]
    assert messages[0] is task
    tool_result = messages[2]
    assert isinstance(tool_result, ToolMessage)
    assert tool_result.tool_call_id == "call-echo"
    assert tool_result.text == "hello"
    assert messages[-1].text == "done"
    assert (result["turns"], result["stop_reason"]) == (2, StopReason.COMPLETED)

    # The role's prompt opens every call the model sees, without ever becoming a
    # turn of the transcript the workflow hands on.
    seen = model.received_messages
    assert [len(model_input) for model_input in seen] == [2, 4]
    assert all(isinstance(model_input[0], SystemMessage) for model_input in seen)
    assert seen[0][0].text == PromptTemplate("roles/coder").render(**PROMPT_CONTEXT)
    assert seen[1][-1].text == tool_result.text
    assert model.bound_tool_names == ("echo",)


def test_agent_stops_at_its_turn_budget_and_keeps_every_notice() -> None:
    model = ScriptedChatModel(
        responses=tuple(
            tool_call("echo", {"value": "looking"}, f"call-{turn}")
            for turn in range(1, 5)
        )
    )

    result = _subgraph(model, max_turns=3).invoke(
        {"messages": [HumanMessage(content="go")]}
    )

    messages = result["messages"]
    assert (result["turns"], result["stop_reason"]) == (3, StopReason.BUDGET_EXHAUSTED)
    expected = [f"[turn {turn}/3]" for turn in (1, 2, 3)]
    assert [model_input[-1].text for model_input in model.received_messages] == expected
    # Every notice stays: the agent has to see the ladder it climbed, not only
    # the rung it is on, or it spends the whole budget investigating.
    assert _notices(messages) == expected
    assert _notices(model.received_messages[-1]) == expected
    # The turn the budget cut short leaves its tool calls unanswered: two rounds
    # ran, the third was never handed to the tools.
    assert sum(isinstance(message, ToolMessage) for message in messages) == 2


def test_agent_can_hide_its_turn_budget() -> None:
    model = ScriptedChatModel(responses=(AIMessage(content="done"),))

    result = _subgraph(model, announce_turn_budget=False).invoke(
        {"messages": [HumanMessage(content="go")]}
    )

    assert not _notices(model.received_messages[0])
    assert not _notices(result["messages"])


def test_agent_rejects_a_budget_below_one() -> None:
    model = ScriptedChatModel(responses=(AIMessage(content="done"),))

    with pytest.raises(ValueError, match="max_turns must be at least 1"):
        _subgraph(model, max_turns=0)


def test_agent_rejects_a_role_without_a_prompt() -> None:
    model = ScriptedChatModel(responses=(AIMessage(content="done"),))

    with pytest.raises(ValueError, match="prompt not found"):
        create_agent(
            role="nobody",
            model=model,
            tools=(echo,),
            prompt_context=PROMPT_CONTEXT,
        )


def test_agent_reports_the_typed_answer_its_role_owes() -> None:
    model = ScriptedChatModel(
        responses=(AIMessage(content='{"semantics": ["a boss", "a through hole"]}'),)
    )

    result = _subgraph(
        model,
        announce_turn_budget=False,
        output_schema=SemanticHypothesis,
    ).invoke({"messages": [HumanMessage(content="go")]})

    assert result["structured_response"] == SemanticHypothesis(
        semantics=["a boss", "a through hole"]
    )
    # The message the answer came from stays in the transcript for the next role.
    assert result["messages"][-1].text == '{"semantics": ["a boss", "a through hole"]}'


def test_agent_refuses_an_answer_that_breaks_its_output_contract() -> None:
    model = ScriptedChatModel(responses=(AIMessage(content='{"semantics": "a boss"}'),))

    with pytest.raises(Exception, match="SemanticHypothesis"):
        _subgraph(
            model,
            announce_turn_budget=False,
            output_schema=SemanticHypothesis,
        ).invoke({"messages": [HumanMessage(content="go")]})


def test_agent_retries_a_dropped_connection() -> None:
    model = _FlakyChatModel(
        responses=(AIMessage(content="done"),),
        errors=(httpx.RemoteProtocolError("peer closed connection"),),
    )

    result = _subgraph(model, announce_turn_budget=False, model_retries=2).invoke(
        {"messages": [HumanMessage(content="go")]}
    )

    assert model.attempts == 2
    assert (result["turns"], result["stop_reason"]) == (1, StopReason.COMPLETED)


@pytest.mark.parametrize(
    "error",
    [httpx.ReadTimeout("too slow"), ValueError("malformed request")],
)
def test_agent_does_not_retry_an_ineligible_failure(error: Exception) -> None:
    model = _FlakyChatModel(responses=(AIMessage(content="done"),), errors=(error,))

    with pytest.raises(type(error), match=str(error)):
        _subgraph(model, announce_turn_budget=False, model_retries=2).invoke(
            {"messages": [HumanMessage(content="go")]}
        )

    assert model.attempts == 1


def test_agent_retries_are_bounded() -> None:
    model = _FlakyChatModel(
        responses=(AIMessage(content="done"),),
        errors=tuple(
            httpx.RemoteProtocolError("peer closed connection") for _ in range(3)
        ),
    )

    with pytest.raises(httpx.RemoteProtocolError):
        _subgraph(model, announce_turn_budget=False, model_retries=1).invoke(
            {"messages": [HumanMessage(content="go")]}
        )

    assert model.attempts == 2


def test_agent_rejects_a_negative_retry_count() -> None:
    model = ScriptedChatModel(responses=(AIMessage(content="done"),))

    with pytest.raises(ValueError, match="max_retries must be >= 0"):
        _subgraph(model, model_retries=-1)


def _broken_tool(error: Exception) -> BaseTool:
    @tool("broken")
    def broken() -> str:
        """Raise the configured failure."""
        raise error

    return broken


def _calling_broken() -> ScriptedChatModel:
    return ScriptedChatModel(
        responses=(
            tool_call("broken", {}, "call-once"),
            AIMessage(content="done"),
        )
    )


def test_agent_does_not_turn_a_tool_defect_into_model_feedback() -> None:
    graph = _subgraph(
        _calling_broken(),
        tools=(_broken_tool(ValueError("internal invariant broken")),),
        announce_turn_budget=False,
    )

    with pytest.raises(ValueError, match="internal invariant broken"):
        graph.invoke({"messages": [HumanMessage(content="go")]})


def test_agent_forwards_a_correctable_tool_error_as_feedback() -> None:
    graph = _subgraph(
        _calling_broken(),
        tools=(_broken_tool(ToolFeedbackError("that is not an image")),),
        announce_turn_budget=False,
    )

    result = graph.invoke({"messages": [HumanMessage(content="go")]})

    tool_messages = [
        message for message in result["messages"] if isinstance(message, ToolMessage)
    ]
    assert len(tool_messages) == 1
    assert tool_messages[0].status == "error"
    assert tool_messages[0].text == "that is not an image"
