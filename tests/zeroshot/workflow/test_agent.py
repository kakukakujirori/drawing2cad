import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest
from langchain.agents.structured_output import StructuredOutputValidationError
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.content import ContentBlock, create_text_block
from langchain_core.outputs import ChatResult
from langchain_core.tools import BaseTool, tool
from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    LengthFinishReasonError,
)
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, PrivateAttr

from tests.zeroshot.chat_models import (
    ScriptedChatModel,
    tool_call,
    unanswered_tool_calls,
)
from zeroshot.pipeline.messages import PromptTemplate
from zeroshot.pipeline.tools import ToolFeedbackError
from zeroshot.pipeline.workflow import Proposal, StopReason
from zeroshot.pipeline.workflow.components.agent import create_agent
from zeroshot.pipeline.workflow.middleware import VerifyOnWriteMiddleware

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
        # These agents are invoked on their own rather than from inside a graph,
        # and `checkpointer=True` -- what a stage builds them with -- means
        # "inherit the parent's", which a root graph has none of.
        checkpointer=False,
        **agent_options,
    )


def _notices(messages: list[BaseMessage]) -> list[str]:
    return [
        message.text
        for message in messages
        if isinstance(message, HumanMessage) and message.text.startswith("[turn ")
    ]


def _countdowns(messages: list[BaseMessage]) -> list[str]:
    """Each notice's `[turn n/m]` alone, without the landing guidance after it."""
    return [notice.split("]", 1)[0] + "]" for notice in _notices(messages)]


def test_agent_returns_its_complete_tool_transcript() -> None:
    model = ScriptedChatModel(
        responses=(
            tool_call("echo", {"value": "hello"}, "call-echo"),
            AIMessage(content="done"),
        )
    )
    task = HumanMessage(content="Use the echo tool")

    result = _subgraph(model, announce_turns=False).invoke({"messages": [task]})

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
    assert (
        result["current_turn"],
        result["total_turns"],
        result["stop_reason"],
    ) == (2, 2, StopReason.COMPLETED)

    # The role's prompt opens every call the model sees, without ever becoming a
    # turn of the transcript the workflow hands on.
    seen = model.received_messages
    assert [len(model_input) for model_input in seen] == [2, 4]
    assert all(isinstance(model_input[0], SystemMessage) for model_input in seen)
    # `max_turns` is the agent's own setting rather than the caller's, so the
    # role renders with it added to what the workflow supplied.
    assert seen[0][0].text == PromptTemplate("roles/coder").render(
        max_turns="30", **PROMPT_CONTEXT
    )
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
    assert (
        result["current_turn"],
        result["total_turns"],
        result["stop_reason"],
    ) == (3, 3, StopReason.BUDGET_EXHAUSTED)
    expected = [f"[turn {turn}/3]" for turn in (1, 2, 3)]
    assert [
        _countdowns([model_input[-1]])[0] for model_input in model.received_messages
    ] == expected
    # Every notice stays: the agent has to see the ladder it climbed, not only
    # the rung it is on, or it spends the whole budget investigating.
    assert _countdowns(messages) == expected
    assert _countdowns(model.received_messages[-1]) == expected
    # Every turn's calls reach the tools, the last one included; what the budget
    # ends is the reading of that answer, not the call.
    assert sum(isinstance(message, ToolMessage) for message in messages) == 3
    assert isinstance(messages[-1], ToolMessage)


def test_a_transcript_the_budget_cut_short_can_be_handed_back_to_a_model() -> None:
    """A stage sent back re-invokes on what the last ask left behind, and a
    provider rejects an input whose tool call nothing answered."""
    model = ScriptedChatModel(
        responses=(
            tool_call("echo", {"value": "looking"}, "call-1"),
            tool_call("echo", {"value": "still looking"}, "call-2"),
            AIMessage(content="corrected"),
        )
    )
    agent = _subgraph(model, max_turns=2, announce_turns=False)

    first = agent.invoke({"messages": [HumanMessage(content="write it")]})
    assert first["stop_reason"] is StopReason.BUDGET_EXHAUSTED
    assert unanswered_tool_calls(first["messages"]) == []

    second = agent.invoke(
        {
            "messages": [*first["messages"], HumanMessage(content="the audit says no")],
            "current_turn": first["current_turn"],
            "total_turns": first["total_turns"],
        }
    )

    assert second["stop_reason"] is StopReason.COMPLETED
    assert unanswered_tool_calls(model.received_messages[-1]) == []


def test_agent_turn_budget_outlives_langgraphs_default_recursion_limit() -> None:
    model = ScriptedChatModel(
        responses=tuple(
            tool_call("echo", {"value": "looking"}, f"call-{turn}")
            for turn in range(1, 8)
        )
    )

    result = _subgraph(model, max_turns=7, announce_turns=False).invoke(
        {"messages": [HumanMessage(content="go")]}
    )

    # Seven tool-bearing turns traverse 28 agent supersteps plus setup, beyond
    # LangGraph's default limit of 25.  TurnBudget must still own termination.
    assert result["current_turn"] == 7
    assert result["stop_reason"] is StopReason.BUDGET_EXHAUSTED
    assert len(model.received_messages) == 7


def test_a_prompt_report_covers_one_ask_rather_than_the_whole_transcript() -> None:
    """A re-entered agent keeps what it built, and the second ask's report has
    to be the instruction that caused it, not that transcript read back."""
    reports: list[dict[str, Any]] = []
    model = ScriptedChatModel(
        responses=(
            tool_call("echo", {"value": "hello"}, "call-echo"),
            AIMessage(content="first"),
            AIMessage(content="second"),
        )
    )
    agent = _subgraph(model, max_turns=3, announce_turns=False)

    first = agent.invoke({"messages": [HumanMessage(content="propose")]})
    for chunk in agent.stream(
        {
            "messages": [*first["messages"], HumanMessage(content="revise this")],
            "current_turn": first["current_turn"],
            "total_turns": first["total_turns"],
            "reported_message_count": first["reported_message_count"],
        },
        stream_mode="custom",
    ):
        reports.append(chunk["prompt"])

    (second_ask,) = reports
    assert second_ask["role"] == "coder"
    # The system prompt does not change between asks, so it is stated once.
    assert second_ask["system"] == ""
    # The instruction and the budget marker the re-entry itself added, and
    # nothing the first ask left behind.
    assert [message.text for message in second_ask["messages"]] == [
        "revise this",
        "[turn reset to 0/3]",
    ]


def test_a_prompt_report_skips_a_transcript_the_agent_was_handed() -> None:
    """The count travels in state, so an agent continuing someone else's
    transcript -- the fan-out reducer, or a stage continuing the one before it
    -- reports its own ask instead of replaying what it inherited."""
    reports: list[dict[str, Any]] = []
    inherited = [
        HumanMessage(content="analyse the drawing"),
        AIMessage(content="a flanged boss"),
    ]
    model = ScriptedChatModel(responses=(AIMessage(content="written"),))
    agent = _subgraph(model, max_turns=3, announce_turns=False)

    for chunk in agent.stream(
        {
            "messages": [*inherited, HumanMessage(content="now write the code")],
            "reported_message_count": len(inherited),
        },
        stream_mode="custom",
    ):
        reports.append(chunk["prompt"])

    (ask,) = reports
    # The system prompt is stated only to an agent that has been told nothing,
    # and this one was handed a transcript.
    assert ask["system"] == ""
    assert [message.text for message in ask["messages"]] == ["now write the code"]


def test_agent_is_asked_to_land_before_its_budget_runs_out() -> None:
    """The last two turns say what running out costs, one turn apart.

    The warning has to reach the agent while a turn whose results it can still
    read remains, because the final turn's tool call returns to nobody.
    """
    model = ScriptedChatModel(
        responses=tuple(
            tool_call("echo", {"value": "looking"}, f"call-{turn}")
            for turn in range(1, 5)
        )
    )

    result = _subgraph(model, max_turns=4).invoke(
        {"messages": [HumanMessage(content="go")]}
    )

    plain, penultimate, final = (
        _notices(result["messages"])[1],
        _notices(result["messages"])[2],
        _notices(result["messages"])[3],
    )
    assert plain == "[turn 2/4]"
    assert penultimate.startswith("[turn 3/4] One turn remains after this one")
    assert final.startswith("[turn 4/4] Final turn")
    assert "will not see" in penultimate and "nothing comes back" in final
    assert result["stop_reason"] is StopReason.BUDGET_EXHAUSTED


def test_a_single_turn_budget_only_asks_for_the_landing() -> None:
    """With one turn there is no earlier turn to warn, so only the last applies."""
    model = ScriptedChatModel(responses=(AIMessage(content="done"),))

    _subgraph(model, max_turns=1).invoke({"messages": [HumanMessage(content="go")]})

    (notice,) = _notices(model.received_messages[0])
    assert notice.startswith("[turn 1/1] Final turn")


def test_agent_can_hide_its_turn_budget() -> None:
    model = ScriptedChatModel(responses=(AIMessage(content="done"),))

    result = _subgraph(model, announce_turns=False).invoke(
        {"messages": [HumanMessage(content="go")]}
    )

    assert not _notices(model.received_messages[0])
    assert not _notices(result["messages"])


def _asked_twice(model: ScriptedChatModel, **agent_options: Any):
    """Two asks of one agent that keeps its transcript, as a stage re-entry does."""
    agent = _subgraph(model, max_turns=3, **agent_options)
    first = agent.invoke({"messages": [HumanMessage(content="propose")]})
    second = agent.invoke(
        {
            "messages": [*first["messages"], HumanMessage(content="revise")],
            "current_turn": first["current_turn"],
            "total_turns": first["total_turns"],
        }
    )
    return first, second


def test_a_second_ask_starts_a_fresh_budget() -> None:
    """An agent keeps its own transcript between asks now, so the counts it spent
    on the ask before this one are still above it. The budget is what one ask may
    cost, so it starts over; the marker is what keeps the restart from reading as
    the counter silently rewinding mid-conversation."""
    model = ScriptedChatModel(
        responses=(AIMessage(content="first"), AIMessage(content="second"))
    )

    first, second = _asked_twice(model)

    assert (first["current_turn"], second["current_turn"]) == (1, 1)
    assert (first["total_turns"], second["total_turns"]) == (1, 2)
    assert _notices(second["messages"]) == [
        "[turn 1/3]",
        "[turn reset to 0/3]",
        "[turn 1/3]",
    ]


def test_a_second_ask_can_keep_spending_the_first_ones_budget() -> None:
    """A role the workflow asks repeatedly within one stage spends one budget
    across those asks, so nothing is reset and no boundary is drawn."""
    model = ScriptedChatModel(
        responses=(AIMessage(content="first"), AIMessage(content="second"))
    )

    first, second = _asked_twice(model, reset_turns_when_reentrant=False)

    assert (first["current_turn"], second["current_turn"]) == (1, 2)
    assert (first["total_turns"], second["total_turns"]) == (1, 2)
    assert _countdowns(second["messages"]) == ["[turn 1/3]", "[turn 2/3]"]


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


_ANSWER = '{"proposal": ["a boss", "a through hole"], "rationale": "both are turned"}'


def test_agent_reports_the_typed_answer_its_role_owes() -> None:
    model = ScriptedChatModel(responses=(AIMessage(content=_ANSWER),))

    result = _subgraph(
        model,
        announce_turns=False,
        output_schema=Proposal,
    ).invoke({"messages": [HumanMessage(content="go")]})

    assert result["structured_response"] == Proposal(
        proposal=["a boss", "a through hole"], rationale="both are turned"
    )
    # The message the answer came from stays in the transcript for the next role.
    assert result["messages"][-1].text == _ANSWER


def test_agent_refuses_an_answer_that_breaks_its_output_contract() -> None:
    model = ScriptedChatModel(responses=(AIMessage(content='{"proposal": "a boss"}'),))

    with pytest.raises(StructuredOutputValidationError, match="Proposal"):
        _subgraph(
            model,
            announce_turns=False,
            output_schema=Proposal,
            model_retries=0,
        ).invoke({"messages": [HumanMessage(content="go")]})


def test_agent_retries_an_empty_structured_output() -> None:
    model = ScriptedChatModel(
        responses=(AIMessage(content=""), AIMessage(content=_ANSWER))
    )

    result = _subgraph(
        model,
        announce_turns=False,
        output_schema=Proposal,
        model_retries=1,
    ).invoke({"messages": [HumanMessage(content="go")]})

    assert result["structured_response"] == Proposal(
        proposal=["a boss", "a through hole"], rationale="both are turned"
    )
    assert len(model.received_messages) == 2
    retry_instruction = model.received_messages[1][-1].text
    assert "Proposal structured output" in retry_instruction
    assert "Validation error" in retry_instruction
    assert "raw JSON" in retry_instruction
    assert (result["current_turn"], result["total_turns"]) == (1, 1)


def test_agent_returns_invalid_structured_output_to_the_model_for_correction() -> None:
    invalid_answer = '{"proposal": "not a list", "rationale": "wrong type"}'
    model = ScriptedChatModel(
        responses=(AIMessage(content=invalid_answer), AIMessage(content=_ANSWER))
    )

    result = _subgraph(
        model,
        announce_turns=False,
        output_schema=Proposal,
        model_retries=1,
    ).invoke({"messages": [HumanMessage(content="go")]})

    retry_messages = model.received_messages[1]
    assert retry_messages[-2].text == invalid_answer
    assert "validation error" in retry_messages[-1].text.lower()
    assert result["structured_response"] == Proposal(
        proposal=["a boss", "a through hole"], rationale="both are turned"
    )


def test_a_rejected_structured_response_is_preserved_in_the_retry_event() -> None:
    invalid_answer = '{"proposal": "not a list", "rationale": "wrong type"}'
    model = ScriptedChatModel(
        responses=(AIMessage(content=invalid_answer), AIMessage(content=_ANSWER))
    )
    reports: list[dict[str, Any]] = []

    for chunk in _subgraph(
        model,
        announce_turns=False,
        output_schema=Proposal,
        model_retries=1,
    ).stream({"messages": [HumanMessage(content="go")]}, stream_mode="custom"):
        if "model_retry" in chunk:
            reports.append(chunk["model_retry"])

    (report,) = reports
    assert report["error_type"] == "StructuredOutputValidationError"
    assert report["failed_response"] == invalid_answer


def test_agent_bounds_invalid_structured_output_retries() -> None:
    model = ScriptedChatModel(
        responses=(AIMessage(content=""), AIMessage(content="still not JSON"))
    )

    with pytest.raises(StructuredOutputValidationError, match="Proposal"):
        _subgraph(
            model,
            announce_turns=False,
            output_schema=Proposal,
            model_retries=1,
        ).invoke({"messages": [HumanMessage(content="go")]})

    assert len(model.received_messages) == 2


def test_async_agent_retries_invalid_structured_output() -> None:
    model = ScriptedChatModel(
        responses=(AIMessage(content=""), AIMessage(content=_ANSWER))
    )

    async def invoke() -> dict[str, Any]:
        return await _subgraph(
            model,
            announce_turns=False,
            output_schema=Proposal,
            model_retries=1,
        ).ainvoke({"messages": [HumanMessage(content="go")]})

    result = asyncio.run(invoke())

    assert len(model.received_messages) == 2
    assert result["structured_response"] == Proposal(
        proposal=["a boss", "a through hole"], rationale="both are turned"
    )
    assert (result["current_turn"], result["total_turns"]) == (1, 1)


def test_agent_retries_a_dropped_connection() -> None:
    model = _FlakyChatModel(
        responses=(AIMessage(content="done"),),
        errors=(httpx.RemoteProtocolError("peer closed connection"),),
    )

    result = _subgraph(model, announce_turns=False, model_retries=2).invoke(
        {"messages": [HumanMessage(content="go")]}
    )

    assert model.attempts == 2
    assert (
        result["current_turn"],
        result["total_turns"],
        result["stop_reason"],
    ) == (1, 1, StopReason.COMPLETED)


def test_agent_retries_a_generic_stream_api_error() -> None:
    request = httpx.Request("POST", "https://example.invalid/responses")
    model = _FlakyChatModel(
        responses=(AIMessage(content="done"),),
        errors=(APIError("backend overloaded", request, body=None),),
    )

    result = _subgraph(model, announce_turns=False, model_retries=2).invoke(
        {"messages": [HumanMessage(content="go")]}
    )

    assert model.attempts == 2
    assert result["stop_reason"] is StopReason.COMPLETED


def test_agent_retries_an_openai_connection_error() -> None:
    request = httpx.Request("POST", "https://example.invalid/responses")
    model = _FlakyChatModel(
        responses=(AIMessage(content="done"),),
        errors=(APIConnectionError(request=request),),
    )

    result = _subgraph(model, announce_turns=False, model_retries=2).invoke(
        {"messages": [HumanMessage(content="go")]}
    )

    assert model.attempts == 2
    assert result["stop_reason"] is StopReason.COMPLETED


@pytest.mark.parametrize("status_code", [429, 500, 503])
def test_agent_retries_a_transient_api_status_error(status_code: int) -> None:
    request = httpx.Request("POST", "https://example.invalid/responses")
    response = httpx.Response(status_code, request=request)
    error = APIStatusError("temporarily unavailable", response=response, body=None)
    model = _FlakyChatModel(responses=(AIMessage(content="done"),), errors=(error,))

    result = _subgraph(model, announce_turns=False, model_retries=2).invoke(
        {"messages": [HumanMessage(content="go")]}
    )

    assert model.attempts == 2
    assert result["stop_reason"] is StopReason.COMPLETED


def test_agent_does_not_retry_an_api_status_error() -> None:
    request = httpx.Request("POST", "https://example.invalid/responses")
    response = httpx.Response(400, request=request)
    error = APIStatusError("bad request", response=response, body=None)
    model = _FlakyChatModel(responses=(AIMessage(content="unused"),), errors=(error,))

    with pytest.raises(APIStatusError, match="bad request"):
        _subgraph(model, announce_turns=False, model_retries=2).invoke(
            {"messages": [HumanMessage(content="go")]}
        )

    assert model.attempts == 1


def _length_failure() -> LengthFinishReasonError:
    completion = ChatCompletion.model_validate(
        {
            "id": "length-limited",
            "created": 0,
            "model": "local-model",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "length",
                    "message": {"role": "assistant", "content": "incomplete"},
                }
            ],
        }
    )
    return LengthFinishReasonError(completion=completion)


def test_agent_retries_a_length_limited_output_with_concise_feedback() -> None:
    model = _FlakyChatModel(
        responses=(AIMessage(content="done"),),
        errors=(_length_failure(),),
    )

    result = _subgraph(model, announce_turns=False, model_retries=2).invoke(
        {"messages": [HumanMessage(content="go")]}
    )

    assert model.attempts == 2
    retry_instruction = model.received_messages[0][-1].text
    assert "output-token limit" in retry_instruction
    assert "Do not call tools" in retry_instruction
    assert "raw JSON" in retry_instruction
    assert "no explanation" in retry_instruction
    # A failed generation is retried inside the same model node, not charged as
    # a completed agent turn.
    assert (result["current_turn"], result["total_turns"]) == (1, 1)


def test_agent_bounds_length_limited_output_retries() -> None:
    model = _FlakyChatModel(
        responses=(AIMessage(content="unused"),),
        errors=(_length_failure(), _length_failure()),
    )

    with pytest.raises(LengthFinishReasonError):
        _subgraph(model, announce_turns=False, model_retries=1).invoke(
            {"messages": [HumanMessage(content="go")]}
        )

    assert model.attempts == 2


def _generic_api_error() -> APIError:
    request = httpx.Request("POST", "https://example.invalid/responses")
    return APIError("backend overloaded", request, body=None)


def test_a_dropped_stream_does_not_spend_a_correction() -> None:
    """A transport failure says nothing about the answer, so the one retry the
    model is owed for a rejected answer is still there afterwards."""
    model = _FlakyChatModel(
        responses=(AIMessage(content="done"),),
        errors=(_generic_api_error(), _length_failure()),
    )

    result = _subgraph(model, announce_turns=False, model_retries=1).invoke(
        {"messages": [HumanMessage(content="go")]}
    )

    assert model.attempts == 3
    assert result["messages"][-1].text == "done"


def test_async_dropped_stream_does_not_spend_a_correction() -> None:
    model = _FlakyChatModel(
        responses=(AIMessage(content="done"),),
        errors=(_generic_api_error(), _length_failure()),
    )

    async def invoke() -> Any:
        return await _subgraph(model, announce_turns=False, model_retries=1).ainvoke(
            {"messages": [HumanMessage(content="go")]}
        )

    result = asyncio.run(invoke())

    assert model.attempts == 3
    assert result["messages"][-1].text == "done"


def test_transport_retries_are_bounded_on_their_own() -> None:
    """Separating the budgets must not make a broken backend retry forever."""
    model = _FlakyChatModel(
        responses=(AIMessage(content="unused"),),
        errors=(_generic_api_error(), _generic_api_error()),
    )

    with pytest.raises(APIError, match="backend overloaded"):
        _subgraph(model, announce_turns=False, model_retries=1).invoke(
            {"messages": [HumanMessage(content="go")]}
        )

    assert model.attempts == 2


def test_every_abandoned_attempt_reports_itself() -> None:
    """A retry leaves no turn, no message and no event of its own, so without
    this the only trace is an extra HTTP request -- and what it leaves behind,
    a model stream nobody will finish, is worth being able to point at."""
    reports: list[dict[str, Any]] = []
    model = _FlakyChatModel(
        responses=(AIMessage(content="done"),),
        errors=(httpx.RemoteProtocolError("peer closed connection"),),
    )

    for chunk in _subgraph(model, announce_turns=False, model_retries=2).stream(
        {"messages": [HumanMessage(content="go")]}, stream_mode="custom"
    ):
        if "model_retry" in chunk:
            reports.append(chunk["model_retry"])

    (report,) = reports
    assert report["role"] == "coder"
    assert (report["attempt"], report["max_attempts"]) == (1, 3)
    assert report["error_type"] == "RemoteProtocolError"
    assert report["retrying"] is True
    assert report["request_adjusted"] is False


def test_a_budget_that_runs_out_says_it_gave_up() -> None:
    model = _FlakyChatModel(
        responses=(AIMessage(content="done"),),
        errors=(ValueError("malformed request"),),
    )

    reports: list[dict[str, Any]] = []
    with pytest.raises(ValueError, match="malformed request"):
        for chunk in _subgraph(model, announce_turns=False, model_retries=2).stream(
            {"messages": [HumanMessage(content="go")]}, stream_mode="custom"
        ):
            if "model_retry" in chunk:
                reports.append(chunk["model_retry"])

    (report,) = reports
    assert report["retrying"] is False
    assert report["error_type"] == "ValueError"


def test_agent_retries_a_stalled_stream() -> None:
    """A backend that accepts the request and then stops sending is retryable.

    The exception arrives bare rather than as `APITimeoutError` because the
    read that times out belongs to the response stream, not to the request the
    SDK still owns.
    """
    model = _FlakyChatModel(
        responses=(AIMessage(content="done"),),
        errors=(httpx.ReadTimeout("too slow"),),
    )

    result = _subgraph(model, announce_turns=False, model_retries=2).invoke(
        {"messages": [HumanMessage(content="go")]}
    )

    assert model.attempts == 2
    assert (
        result["current_turn"],
        result["total_turns"],
        result["stop_reason"],
    ) == (1, 1, StopReason.COMPLETED)


@pytest.mark.parametrize(
    "error",
    [ValueError("malformed request")],
)
def test_agent_does_not_retry_an_ineligible_failure(error: Exception) -> None:
    model = _FlakyChatModel(responses=(AIMessage(content="done"),), errors=(error,))

    with pytest.raises(type(error), match=str(error)):
        _subgraph(model, announce_turns=False, model_retries=2).invoke(
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
        _subgraph(model, announce_turns=False, model_retries=1).invoke(
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
        announce_turns=False,
    )

    with pytest.raises(ValueError, match="internal invariant broken"):
        graph.invoke({"messages": [HumanMessage(content="go")]})


def test_agent_forwards_a_correctable_tool_error_as_feedback() -> None:
    graph = _subgraph(
        _calling_broken(),
        tools=(_broken_tool(ToolFeedbackError("that is not an image")),),
        announce_turns=False,
    )

    result = graph.invoke({"messages": [HumanMessage(content="go")]})

    tool_messages = [
        message for message in result["messages"] if isinstance(message, ToolMessage)
    ]
    assert len(tool_messages) == 1
    assert tool_messages[0].status == "error"
    assert tool_messages[0].text == "that is not an image"


class _CountingVerifier:
    """A verifier that records when it was asked, and for what.

    `builds` decides the outcome of each build in order, so a test can say the
    program is broken until the model repairs it.
    """

    def __init__(self, source_path: Path, builds: Sequence[bool] = ()) -> None:
        self.source_path = source_path
        self.seen: list[str] = []
        self._builds = list(builds)
        self._sound = False

    @property
    def confirmed_a_solid(self) -> bool:
        return self._sound

    def feedback(self) -> list[ContentBlock]:
        self.seen.append(self.source_path.read_text(encoding="utf-8"))
        self._sound = self._builds.pop(0) if self._builds else True
        return [create_text_block(f"[verified] build {len(self.seen)}")]


def _writing_tool(path: Path) -> BaseTool:
    @tool("write")
    def write(text: str) -> str:
        """Write `text` to the watched file."""
        path.write_text(text, encoding="utf-8")
        return "written"

    return write


def _verifying_agent(
    model: ScriptedChatModel,
    path: Path,
    builds: Sequence[bool] = (),
    **agent_options: Any,
) -> tuple[Any, _CountingVerifier]:
    verifier = _CountingVerifier(path, builds)
    graph = _subgraph(
        model,
        tools=(_writing_tool(path), echo),
        extra_middleware=[VerifyOnWriteMiddleware(verifier)],
        **agent_options,
    )
    return graph, verifier


def test_a_turn_that_rewrote_the_program_is_told_what_it_built(
    tmp_path: Path,
) -> None:
    """The agent never asks: writing the file is the whole request."""
    path = tmp_path / "model.py"
    graph, verifier = _verifying_agent(
        ScriptedChatModel(
            responses=(
                tool_call("write", {"text": "result = 1"}, "call-1"),
                AIMessage(content="done"),
            )
        ),
        path,
        announce_turns=False,
    )

    result = graph.invoke({"messages": [HumanMessage(content="go")]})

    assert verifier.seen == ["result = 1"]
    assert any("[verified] build 1" in message.text for message in result["messages"])


def test_a_write_after_middleware_construction_is_reported(
    tmp_path: Path,
) -> None:
    """No framework writer remains, so every content change is meaningful."""
    path = tmp_path / "model.py"
    graph, verifier = _verifying_agent(
        ScriptedChatModel(responses=(AIMessage(content="done"),)),
        path,
        announce_turns=False,
    )
    path.write_text("result = 1", encoding="utf-8")

    graph.invoke({"messages": [HumanMessage(content="go")]})

    assert verifier.seen == ["result = 1"]


def test_a_turn_that_rewrites_an_existing_program_is_reported(
    tmp_path: Path,
) -> None:
    """The digest starts from existing source and reports only its later edit."""
    path = tmp_path / "model.py"
    path.write_text("result = 0", encoding="utf-8")
    graph, verifier = _verifying_agent(
        ScriptedChatModel(
            responses=(
                tool_call("write", {"text": "result = 1"}, "call-1"),
                AIMessage(content="done"),
            )
        ),
        path,
        announce_turns=False,
    )

    graph.invoke({"messages": [HumanMessage(content="go")]})

    assert verifier.seen == ["result = 1"]


def test_a_turn_that_only_read_the_program_builds_nothing(tmp_path: Path) -> None:
    """The agent reads the file far more often than it writes it, and a build
    per read would cost more than the verification is worth."""
    path = tmp_path / "model.py"
    path.write_text("result = 1", encoding="utf-8")
    graph, verifier = _verifying_agent(
        ScriptedChatModel(
            responses=(
                tool_call("echo", {"text": "cat model.py"}, "call-1"),
                AIMessage(content="done"),
            )
        ),
        path,
        announce_turns=False,
    )

    graph.invoke({"messages": [HumanMessage(content="go")]})

    assert verifier.seen == []


def test_one_turn_of_edits_costs_one_build(tmp_path: Path) -> None:
    """A turn rewrites the program several times over; only where it ends is an
    answer, and the states in between are nobody's."""
    path = tmp_path / "model.py"
    graph, verifier = _verifying_agent(
        ScriptedChatModel(
            responses=(
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "write", "args": {"text": "first"}, "id": "call-1"},
                        {"name": "write", "args": {"text": "second"}, "id": "call-2"},
                        {"name": "write", "args": {"text": "third"}, "id": "call-3"},
                    ],
                ),
                AIMessage(content="done"),
            )
        ),
        path,
        announce_turns=False,
    )

    graph.invoke({"messages": [HumanMessage(content="go")]})

    # One build, of the state the turn ended in. Which of the three parallel
    # calls landed last is the tool node's business, not this middleware's.
    assert verifier.seen == [path.read_text(encoding="utf-8")]


def test_a_build_is_not_a_turn(tmp_path: Path) -> None:
    """The whole point: the agent spends nothing to learn what it built."""
    path = tmp_path / "model.py"
    responses = tuple(
        tool_call("write", {"text": f"result = {turn}"}, f"call-{turn}")
        for turn in range(3)
    ) + (AIMessage(content="done"),)
    graph, verifier = _verifying_agent(
        ScriptedChatModel(responses=responses), path, announce_turns=False
    )

    result = graph.invoke({"messages": [HumanMessage(content="go")]})

    assert len(verifier.seen) == 3
    assert result["current_turn"] == 4


def test_a_budget_already_spent_builds_nothing(tmp_path: Path) -> None:
    """A build nobody will read is ten seconds spent on nothing.

    The last turn's write goes unverified here, which is the workflow's own
    final verification to catch; what this fixes is the eight turns before it.
    """
    path = tmp_path / "model.py"
    graph, verifier = _verifying_agent(
        ScriptedChatModel(
            responses=tuple(
                tool_call("write", {"text": f"result = {turn}"}, f"call-{turn}")
                for turn in range(4)
            )
        ),
        path,
        max_turns=2,
        announce_turns=False,
    )

    result = graph.invoke({"messages": [HumanMessage(content="go")]})

    assert result["stop_reason"] == StopReason.BUDGET_EXHAUSTED
    # Two turns wrote, but TurnBudget ends the agent ahead of this middleware,
    # so the second write is never built.
    assert verifier.seen == ["result = 0"]


class _Answer(BaseModel):
    """The stage's answer."""

    done: bool


def _answer_call(call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "_Answer",
                "args": {"done": True},
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


def test_an_answer_is_refused_while_the_program_does_not_build(
    tmp_path: Path,
) -> None:
    """The answer schema is bound as a tool, so a model can call it part-way
    through the work; only the build stops a broken program ending the stage."""
    path = tmp_path / "model.py"
    path.write_text("result = broken", encoding="utf-8")
    graph, verifier = _verifying_agent(
        ScriptedChatModel(
            responses=(
                _answer_call("call-1"),
                tool_call("write", {"text": "result = 1"}, "call-2"),
                _answer_call("call-3"),
            )
        ),
        path,
        builds=[False, True],
        announce_turns=False,
        output_schema=_Answer,
        response_format_strategy="tool",
    )

    result = graph.invoke({"messages": [HumanMessage(content="go")]})

    assert result["structured_response"] == _Answer(done=True)
    refusal = next(
        message.text
        for message in result["messages"]
        if "does not build" in message.text
    )
    assert "not ready to submit" in refusal
    assert verifier.seen == ["result = broken", "result = 1"]


def test_an_answer_stands_when_the_program_builds(tmp_path: Path) -> None:
    path = tmp_path / "model.py"
    path.write_text("result = 1", encoding="utf-8")
    graph, _ = _verifying_agent(
        ScriptedChatModel(responses=(_answer_call("call-1"),)),
        path,
        announce_turns=False,
        output_schema=_Answer,
        response_format_strategy="tool",
    )

    result = graph.invoke({"messages": [HumanMessage(content="go")]})

    assert result["structured_response"] == _Answer(done=True)
    assert not any("does not build" in m.text for m in result["messages"])


def test_a_refusal_repeats_what_the_build_reported(tmp_path: Path) -> None:
    """The failure has to travel with the refusal; a turn behind it is a turn
    the model has to go looking for."""
    path = tmp_path / "model.py"
    path.write_text("result = broken", encoding="utf-8")
    graph, _ = _verifying_agent(
        ScriptedChatModel(
            responses=(
                tool_call("write", {"text": "still broken"}, "call-1"),
                _answer_call("call-2"),
                tool_call("write", {"text": "result = 1"}, "call-3"),
                _answer_call("call-4"),
            )
        ),
        path,
        builds=[False, True],
        announce_turns=False,
        output_schema=_Answer,
        response_format_strategy="tool",
    )

    result = graph.invoke({"messages": [HumanMessage(content="go")]})

    refusal = next(
        message.text
        for message in result["messages"]
        if "does not build" in message.text
    )
    # The answer came in a turn that wrote nothing, so no build ran for it;
    # the refusal still carries the report from the build that did.
    assert "[verified] build 1" in refusal


def test_agent_offers_only_the_answer_on_its_final_turn() -> None:
    model = ScriptedChatModel(
        responses=(
            tool_call("echo", {"value": "looking"}, "call-1"),
            AIMessage(content=_ANSWER),
        )
    )

    result = _subgraph(
        model,
        max_turns=2,
        output_schema=Proposal,
    ).invoke({"messages": [HumanMessage(content="go")]})

    working_turn, final_turn = model.bound_tool_name_history
    assert "echo" in working_turn
    assert "echo" not in final_turn
    assert result["structured_response"] == Proposal(
        proposal=["a boss", "a through hole"], rationale="both are turned"
    )


def _empty(finish_reason: str) -> AIMessage:
    """An answer carrying nothing, with the backend's reason for it."""
    return AIMessage(content="", response_metadata={"finish_reason": finish_reason})


def test_a_length_truncated_answer_earns_the_length_correction() -> None:
    """A backend that reports truncation in `finish_reason` rather than by
    raising still gets the correction written for running out of room."""
    model = ScriptedChatModel(
        responses=(_empty("length"), AIMessage(content="done")),
    )

    # No tools, so the turn had nothing to do but answer.
    _subgraph(model, tools=(), announce_turns=False, model_retries=1).invoke(
        {"messages": [HumanMessage(content="go")]}
    )

    assert "output-token limit" in model.received_messages[-1][-1].text


def test_a_working_turn_that_ran_long_is_not_told_to_stop_using_tools() -> None:
    """`Do not call tools` reads as give up while there is still work to do --
    one coding stage answered "not yet implemented" on its first turn."""
    model = ScriptedChatModel(
        responses=(_empty("length"), AIMessage(content="done")),
    )

    _subgraph(model, announce_turns=False, model_retries=1).invoke(
        {"messages": [HumanMessage(content="go")]}
    )

    correction = model.received_messages[-1][-1].text
    assert "Do not call tools" not in correction
    assert "whole output budget on thinking" in correction


def test_an_empty_answer_that_did_not_run_out_earns_the_plain_nudge() -> None:
    model = ScriptedChatModel(
        responses=(_empty("stop"), AIMessage(content="done")),
    )

    _subgraph(model, announce_turns=False, model_retries=1).invoke(
        {"messages": [HumanMessage(content="go")]}
    )

    assert "whole output budget on thinking" in model.received_messages[-1][-1].text
