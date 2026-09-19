"""Correcting a rejected structured answer without leaving the stage.

Both strategies reach the same correction path: the parse failure raises, and
`ModelCallRetryMiddleware` re-issues the call carrying the validation error.
A `structured_response` written into agent state short-circuits it, because
langchain reads that key as an answer already given.
"""

import json

import httpx
import pytest
from langchain_core.messages import AIMessage
from openrouter.errors.badrequestresponse_error import (
    BadRequestResponseError,
    BadRequestResponseErrorData,
)
from pydantic import BaseModel, field_validator

from tests.zeroshot.chat_models import ScriptedChatModel
from tests.zeroshot.workflow.test_agent import _FlakyChatModel, _length_failure
from zeroshot.pipeline.workflow import create_agent
from zeroshot.pipeline.workflow.middleware.agent_retry import (
    _rejected_arguments,
)
from zeroshot.pipeline.workflow.middleware.connection_retry import (
    ModelConnectionRetry,
    _provider_error_detail,
    is_retryable_model_error,
)


class Answer(BaseModel):
    """The stage's answer."""

    ticket_id: str

    @field_validator("ticket_id")
    @classmethod
    def require_a_ticket_identifier(cls, value: str) -> str:
        if not value.startswith("ticket_"):
            raise ValueError("ticket_id must be a ticket_... identifier")
        return value


def _answering(ticket_id: str, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "Answer",
                "args": {"ticket_id": ticket_id},
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


def _answering_twice(call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "Answer",
                "args": {"ticket_id": "ticket_initial"},
                "id": f"{call_id}_{index}",
                "type": "tool_call",
            }
            for index in range(2)
        ],
    )


def _agent(model: ScriptedChatModel, *, model_retries: int = 5):
    return create_agent(
        role="coder",
        model=model,
        tools=[],
        output_schema=Answer,
        response_format_strategy="tool",
        max_turns=5,
        announce_turns=False,
        model_retries=model_retries,
        checkpointer=False,
    )


def test_a_rejected_answer_is_returned_to_the_model_to_correct() -> None:
    model = ScriptedChatModel(
        responses=(_answering("T1", "call_1"), _answering("ticket_initial", "call_2"))
    )

    result = _agent(model).invoke({"messages": []})

    assert result["structured_response"] == Answer(ticket_id="ticket_initial")
    assert len(model.received_messages) == 2
    correction = model.received_messages[1][-1]
    assert "ticket_id must be a ticket_... identifier" in str(correction.content)


def test_an_answer_that_stays_malformed_ends_the_stage_and_says_why() -> None:
    """Exhausting the budget ends the turn with no answer. The graph re-asks a
    stage that produced none, and raising here reached none of that."""
    model = ScriptedChatModel(
        responses=tuple(_answering("T1", "call") for _ in range(3))
    )

    result = _agent(model, model_retries=2).invoke({"messages": []})

    assert len(model.received_messages) == 3
    assert result.get("structured_response") is None
    gave_up = result["messages"][-1].text
    assert "still could not be read" in gave_up
    assert "ticket_id must be a ticket_" in gave_up


def test_a_stage_re_entry_does_not_carry_an_answer_into_the_next_invocation() -> None:
    """`create_reconstruction_graph` hands an agent back its own prior state."""
    model = ScriptedChatModel(
        responses=(_answering("T2", "call_1"), _answering("ticket_new", "call_2"))
    )

    result = _agent(model).invoke(
        {
            "messages": [],
            "structured_response": Answer(ticket_id="ticket_previous"),
            "current_turn": 2,
        }
    )

    assert result["structured_response"] == Answer(ticket_id="ticket_new")


def test_two_structured_answers_in_one_turn_are_returned_to_the_model() -> None:
    """A model that calls the answer tool twice is asked for one, not abandoned."""
    model = ScriptedChatModel(
        responses=(_answering_twice("call_1"), _answering("ticket_initial", "call_2"))
    )

    result = _agent(model).invoke({"messages": []})

    assert result["structured_response"] == Answer(ticket_id="ticket_initial")
    assert len(model.received_messages) == 2
    correction = model.received_messages[1][-1]
    assert "2 structured responses" in str(correction.content)
    assert "Answer, Answer" in str(correction.content)


def test_a_rejected_answer_is_not_replayed_with_its_unanswered_tool_calls() -> None:
    """Replaying a tool-calling AIMessage without results invalidates the retry."""
    model = ScriptedChatModel(
        responses=(_answering_twice("call_1"), _answering("ticket_initial", "call_2"))
    )

    _agent(model).invoke({"messages": []})

    assert not any(
        getattr(message, "tool_calls", None) for message in model.received_messages[1]
    )


def test_a_turn_that_only_thought_is_asked_again_rather_than_ending_the_stage() -> None:
    """A model that spends its output budget reasoning returns an empty message,
    which the agent loop would otherwise read as a finished answer."""
    model = ScriptedChatModel(
        responses=(AIMessage(content=""), _answering("ticket_initial", "call_1"))
    )

    result = _agent(model).invoke({"messages": []})

    assert result["structured_response"] == Answer(ticket_id="ticket_initial")
    assert len(model.received_messages) == 2
    nudge = str(model.received_messages[1][-1].content)
    assert "You have been thinking a long time" in nudge
    assert "build on it rather than starting over" in nudge


def test_text_without_a_tool_call_is_asked_again_rather_than_ending_the_stage() -> None:
    """Under ToolStrategy the agent loop ends on text, with no answer."""
    said = AIMessage(content="The program is done.", response_metadata={"id": "gen-1"})
    model = ScriptedChatModel(responses=(said, _answering("ticket_initial", "call_1")))

    reports = [
        chunk["model_retry"]
        for chunk in _agent(model).stream({"messages": []}, stream_mode="custom")
        if "model_retry" in chunk
    ]

    (report,) = reports
    assert report["error_type"] == "TextWithoutToolCall"
    assert report["generation_id"] == "gen-1"
    replayed, correction = model.received_messages[1][-2:]
    assert replayed.text == "The program is done."
    assert "call Answer to submit your answer" in correction.text


def test_a_rejected_tool_answer_is_asked_for_again_as_a_call() -> None:
    """Asked for raw JSON, a model free to skip tools wrote its answer as text."""
    model = ScriptedChatModel(
        responses=(_answering("T1", "call_1"), _answering("ticket_initial", "call_2"))
    )

    _agent(model).invoke({"messages": []})

    correction = str(model.received_messages[1][-1].content)
    assert "call Answer again with corrected arguments" in correction
    assert "raw JSON" not in correction


def test_a_cut_off_tool_answer_is_asked_for_again_as_a_shorter_call() -> None:
    model = _FlakyChatModel(
        responses=(_answering("ticket_initial", "call_1"),),
        errors=(_length_failure(),),
    )

    _agent(model).invoke({"messages": []})

    correction = model.received_messages[0][-1].text
    assert "Call Answer again with concise arguments" in correction
    assert "Do not call tools" not in correction


def test_a_call_whose_arguments_did_not_parse_is_asked_for_again_whole() -> None:
    """Together finished the call and LangChain dropped it; an empty turn it is not."""
    unparsed = AIMessage(
        content="",
        invalid_tool_calls=[
            {
                "name": "Answer",
                "args": '{"ticket_id": "ticket_initial"',
                "id": "call_1",
                "error": "Expecting ',' delimiter",
                "type": "invalid_tool_call",
            }
        ],
    )
    model = ScriptedChatModel(
        responses=(unparsed, _answering("ticket_initial", "call_2"))
    )

    result = _agent(model).invoke({"messages": []})

    correction = model.received_messages[1][-1].text
    assert "arguments were not valid JSON" in correction
    assert "Expecting ',' delimiter" in correction
    assert "concise arguments" not in correction
    assert result["structured_response"] == Answer(ticket_id="ticket_initial")


def test_a_model_that_never_answers_ends_the_stage_without_failing_the_run() -> None:
    """The graph re-asks a stage that submitted nothing, so raising here would
    end a run the pipeline can still recover."""
    model = ScriptedChatModel(responses=tuple(AIMessage(content="") for _ in range(3)))

    result = _agent(model, model_retries=2).invoke({"messages": []})

    assert result.get("structured_response") is None
    assert len(model.received_messages) == 3
    gave_up = result["messages"][-1].text
    assert "still could not be read" in gave_up
    assert "no tool call, no text" in gave_up
    assert "call Answer with the corrected answer" in gave_up


_STREAM_ERROR = "OpenRouter API returned an error during streaming: "


@pytest.mark.parametrize(
    ("message", "retryable"),
    [
        (_STREAM_ERROR + "Network connection lost. (code: 502)", True),
        (_STREAM_ERROR + "Rate limited. (code: 429)", True),
        (_STREAM_ERROR + "Bad request. (code: 400)", False),
        # A refusal of the request: sending it again would be refused again.
        (_STREAM_ERROR + "Input should be a valid string (code: 422)", False),
        # The model wrapper names the fields it could have been about, so the
        # status is no longer the last thing in the message.
        (_STREAM_ERROR + "Rate limited. (code: 429) Suspect fields: a.b.", True),
        ("some unrelated ValueError", False),
    ],
)
def test_a_dropped_openrouter_stream_is_retryable_by_its_status(
    message: str, retryable: bool
) -> None:
    """The library raises a bare ValueError with the status only in the text."""
    assert is_retryable_model_error(ValueError(message)) is retryable


def test_a_non_valueerror_carrying_the_same_text_is_not_matched() -> None:
    """Matching is on the library's own exception type, not on any message."""
    assert not is_retryable_model_error(
        RuntimeError("OpenRouter API returned an error during streaming: x (code: 502)")
    )


def test_openrouter_400_records_bounded_provider_details_without_retrying():
    data = BadRequestResponseErrorData.model_validate(
        {
            "error": {
                "code": 400,
                "message": "Provider returned error",
                "metadata": {
                    "provider_name": "ExampleProvider",
                    "raw": json.dumps(
                        {
                            "authorization": "provider-secret",
                            "source": "private-source",
                            "image": "data:image/png;base64,private-image",
                            "message": "Unmatched tool call. " + "x" * 5000,
                        }
                    ),
                    "request": "private-request",
                },
            },
        }
    )
    response = httpx.Response(400, headers={"authorization": "header-secret"})
    error = BadRequestResponseError(data, response, body="private-body")
    events = []
    calls = 0

    def fail():
        nonlocal calls
        calls += 1
        raise error

    with pytest.raises(BadRequestResponseError):
        ModelConnectionRetry(5, "interpreter", events.append).invoke(fail)

    assert calls == 1
    event = events[0]["model_retry"]
    assert event["status_code"] == 400 and not event["retrying"]
    assert event["error"] == "Provider returned error"
    assert event["provider_error"]["provider_name"] == "ExampleProvider"
    raw = event["provider_error"]["raw"]
    assert "Unmatched tool call" in raw and len(raw) == 4000
    assert not any(
        value in json.dumps(event)
        for value in (
            "provider-secret",
            "private-source",
            "private-image",
            "private-request",
            "header-secret",
            "private-body",
        )
    )


def test_provider_diagnostics_drop_request_echoes_and_redact_embedded_credentials():
    raw = {
        "error": {
            "message": "Invalid input: data:image/png;base64,cHJpdmF0ZQ==",
            "type": "BadRequestError",
            "code": 400,
            "param": "messages[27].content",
        },
        "detail": [{"loc": ["body", "messages", 27], "input": "private prompt"}],
        "request": {"messages": [{"role": "user", "content": "private prompt"}]},
        "headers": {"X-API-Key": "diagnostic-secret"},
    }
    assert _provider_error_detail(raw) == {
        "error": {
            "message": "Invalid input: <redacted image>",
            "type": "BadRequestError",
            "code": 400,
            "param": "messages[27].content",
        },
        "detail": [{"loc": ["body", "messages", 27]}],
    }
    for text in (
        "Invalid credential: Authorization: Bearer diagnostic-secret",
        "Authorization: Basic diagnostic-secret",
        "Bearer diagnostic-secret",
        "'api_key': 'diagnostic-secret'",
        "X-API-Key=diagnostic-secret",
    ):
        assert "diagnostic-secret" not in _provider_error_detail(text)
    assert _provider_error_detail("temporarily rate-limited upstream") == (
        "temporarily rate-limited upstream"
    )


def test_provider_diagnostics_keep_cloudflare_nested_validation_errors():
    diagnostic = {
        "code": 8007,
        "message": "AiError: Image count 9 exceeds limit 8 per request",
        "detail": [
            {
                "loc": ["body", "messages", 27, "content"],
                "msg": "Input should be a valid string",
                "type": "string_type",
            }
        ],
    }
    raw = {
        "errors": [
            {
                **diagnostic,
                "detail": [
                    {
                        **diagnostic["detail"][0],
                        "input": "data:image/png;base64,private-image",
                        "ctx": {"request": "private prompt"},
                    }
                ],
                "headers": {"Authorization": "Bearer private-key"},
            }
        ],
        "success": False,
        "result": {"request": "private prompt"},
        "messages": ["private prompt"],
    }

    assert _provider_error_detail(raw) == {"errors": [diagnostic]}


def test_a_tool_call_answer_is_shown_back_to_its_author() -> None:
    """A tool-call answer is never replayed as a message -- a call with no
    result invalidates the next request -- so without this the model is told
    only that its answer was wrong, never what it sent."""
    model = ScriptedChatModel(
        responses=(_answering("T1", "call_1"), _answering("ticket_initial", "call_2"))
    )

    _agent(model).invoke({"messages": []})

    correction = str(model.received_messages[1][-1].content)
    assert 'You sent: {"ticket_id": "T1"}' in correction


def test_a_large_rejected_answer_comes_back_as_its_shape() -> None:
    """Which member was wrong is what a rejected answer has to show. A whole
    hypothesis restated is the same information at fifty times the price."""
    long_id = "T" + "x" * 4000
    model = ScriptedChatModel(
        responses=(
            _answering(long_id, "call_1"),
            _answering("ticket_initial", "call_2"),
        )
    )

    _agent(model).invoke({"messages": []})

    correction = str(model.received_messages[1][-1].content)
    assert long_id not in correction
    assert '"ticket_id": "<4001 characters>"' in correction


def test_a_plain_text_answer_is_not_shown_back_twice() -> None:
    """It is already replayed as the message it was."""
    assert _rejected_arguments(AIMessage(content="not JSON"), "Answer") == ""


class _Report(BaseModel):
    findings: list[Answer]


def test_a_large_rejected_answer_shows_the_entry_the_error_names() -> None:
    """Shown only a shape, an auditor rewrote its whole report and broke new parts."""
    model = ScriptedChatModel(
        responses=(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "_Report",
                        "args": {
                            "findings": [
                                *(
                                    {"ticket_id": "ticket_" + "x" * 400}
                                    for _ in range(4)
                                ),
                                {"ticket_id": "T9"},
                            ]
                        },
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "_Report",
                        "args": {"findings": []},
                        "id": "call_2",
                        "type": "tool_call",
                    }
                ],
            ),
        )
    )

    create_agent(
        role="auditor",
        model=model,
        tools=[],
        output_schema=_Report,
        response_format_strategy="tool",
        max_turns=5,
        announce_turns=False,
        checkpointer=False,
    ).invoke({"messages": []})

    correction = str(model.received_messages[1][-1].content)
    assert '{"ticket_id": "T9"}' in correction
    assert (
        "Validation error: $.findings[4].ticket_id: "
        "ticket_id must be a ticket_... identifier." in correction
    )
    assert '"<1 keys>"' in correction
