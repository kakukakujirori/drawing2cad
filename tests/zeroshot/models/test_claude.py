"""ChatClaude under the staged pipeline's structured-output strategies, offline."""

import json
from contextlib import contextmanager

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.content import create_image_block
from langchain_core.tools import tool

from zeroshot.pipeline.models import claude
from zeroshot.pipeline.models.claude import ChatClaude, ClaudeAPIError
from zeroshot.pipeline.models.image_history import INPUT_IMAGE_ID
from zeroshot.pipeline.stages.tickets.contracts import TicketAnswers
from zeroshot.pipeline.workflow import create_agent
from zeroshot.pipeline.workflow.middleware.connection_retry import (
    is_retryable_model_error,
)

_ANSWER = {"responses": {"ticket_0": "built a box"}, "stage_report": {"concerns": {}}}
_ANSWER_CALL = {
    "type": "tool_use",
    "id": "tu_1",
    "name": "TicketAnswers",
    "input": _ANSWER,
}


@tool
def run_shell(command: str) -> str:
    """Run a shell command."""
    return command


def _model() -> ChatClaude:
    return ChatClaude(
        model="claude-opus-5-5",
        token="test-token",
        thinking={"type": "adaptive", "display": "summarized", "effort": "xhigh"},
    )


def _serve(monkeypatch, replies):
    """Answer each Messages API request with the next reply, keeping the request."""
    requests = []

    class Client:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, url, headers=None, json=None):
            del headers
            requests.append(json)
            content = replies[len(requests) - 1]
            return httpx.Response(
                200,
                json={
                    "content": content,
                    "stop_reason": "tool_use"
                    if any(block["type"] == "tool_use" for block in content)
                    else "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr(claude.httpx, "Client", Client)
    return requests


@pytest.mark.parametrize(
    ("tool_choice", "sent"),
    [
        ("any", {"type": "auto"}),
        ("run_shell", {"type": "auto"}),
        ({"type": "tool", "name": "run_shell"}, {"type": "auto"}),
        ("auto", {"type": "auto"}),
        ("none", {"type": "none"}),
        (
            {"type": "any", "disable_parallel_tool_use": True},
            {"type": "auto", "disable_parallel_tool_use": True},
        ),
    ],
)
def test_forced_tool_choice_is_sent_as_auto(tool_choice, sent):
    payload = _model()._build_payload(
        [HumanMessage("go")], tools=[run_shell], tool_choice=tool_choice
    )
    assert payload["tool_choice"] == sent


def test_tool_strategy_answers_through_the_answer_tool(monkeypatch):
    requests = _serve(
        monkeypatch,
        [[_ANSWER_CALL]],
    )
    agent = create_agent(
        role="coder",
        model=_model(),
        tools=[run_shell],
        output_schema=TicketAnswers,
        response_format_strategy="tool",
        max_turns=3,
        model_call_retries=1,
    )
    result = agent.invoke({"messages": [HumanMessage("go")]})

    assert result["structured_response"] == TicketAnswers.model_validate(_ANSWER)
    assert [t["name"] for t in requests[0]["tools"]] == ["run_shell", "TicketAnswers"]
    assert requests[0]["tool_choice"] == {"type": "auto"}


def test_a_prose_answer_is_asked_again_for_the_answer_tool(monkeypatch):
    requests = _serve(
        monkeypatch,
        [
            [{"type": "text", "text": "I built the box."}],
            [_ANSWER_CALL],
        ],
    )
    agent = create_agent(
        role="coder",
        model=_model(),
        tools=[run_shell],
        output_schema=TicketAnswers,
        response_format_strategy="tool",
        max_turns=3,
        model_call_retries=1,
    )
    result = agent.invoke({"messages": [HumanMessage("go")]})

    assert result["structured_response"] == TicketAnswers.model_validate(_ANSWER)
    assert "call TicketAnswers" in str(requests[1]["messages"][-1]["content"])


def test_provider_strategy_is_refused_before_the_api(monkeypatch):
    """Why the Claude configs use `tool`: ChatClaude sends no response format, and
    TicketAnswers' free-form maps are outside what Anthropic structured outputs
    accept (every object must set additionalProperties to false)."""
    requests = _serve(monkeypatch, [])
    agent = create_agent(
        role="coder",
        model=_model(),
        tools=[run_shell],
        output_schema=TicketAnswers,
        response_format_strategy="provider",
        max_turns=3,
        model_call_retries=0,
    )
    with pytest.raises(ValueError, match="response_format_strategy: tool"):
        agent.invoke({"messages": [HumanMessage("go")]})
    assert requests == []


@pytest.mark.parametrize(
    "block",
    [
        # How the staged prompt attaches an input drawing.
        create_image_block(base64="ZHJhd2luZw==", mime_type="image/png", id="x"),
        # How the native prompt does.
        create_image_block(url="data:image/png;base64,ZHJhd2luZw==", id="x"),
    ],
)
def test_langchain_image_blocks_are_sent_in_anthropic_form(block):
    sent = _model()._build_payload([HumanMessage(content=[block])])["messages"]
    assert sent[0]["content"] == [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "ZHJhd2luZw==",
            },
        }
    ]


def test_image_history_keeps_the_drawing_and_the_newest_tool_images():
    """load_image answers in a tool message, as every native inspection does."""
    drawing = "data:image/png;base64,ZHJhd2luZw=="
    messages = [
        HumanMessage(content=[create_image_block(url=drawing, id=INPUT_IMAGE_ID)])
    ]
    for i in range(9):
        messages += [
            AIMessage(
                content="",
                tool_calls=[{"name": "load_image", "args": {}, "id": f"c{i}"}],
            ),
            ToolMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{i}AAA"},
                    }
                ],
                tool_call_id=f"c{i}",
            ),
        ]
    model = ChatClaude(model="claude-opus-5-5", token="t", image_history_limit=8)
    sent = model._build_payload(messages)["messages"]

    images = [
        part
        for message in sent
        for block in message["content"]
        for part in (
            block["content"] if block.get("type") == "tool_result" else [block]
        )
        if part.get("type") == "image"
    ]
    assert images[0]["source"]["data"] == "ZHJhd2luZw=="
    assert [image["source"]["data"] for image in images[1:]] == [
        f"{i}AAA" for i in range(2, 9)
    ]
    # The transcript the agent keeps is not pruned.
    assert all(isinstance(m.content, list) for m in messages if m.type == "tool")


def test_thinking_goes_back_signed_and_in_the_order_claude_wrote_it():
    """Claude reads its earlier reasoning only from the blocks sent back."""
    call = {"name": "run_shell", "args": {"command": "ls"}, "id": "tu_1"}
    answer = AIMessage(
        content=[
            {"type": "reasoning", "reasoning": "plan", "signature": "sig", "index": 0},
            {"type": "reasoning", "redacted": "opaque", "index": 1},
            # Not this model's, so there is nothing the API could verify.
            {"type": "reasoning", "reasoning": "unsigned", "index": 2},
            {"type": "text", "text": "Listing.", "index": 3},
            {"type": "tool_call", **call},
        ],
        tool_calls=[call],
    )
    sent = _model()._build_payload(
        [HumanMessage("go"), answer, ToolMessage("a b", tool_call_id="tu_1")]
    )["messages"]

    assert sent[1] == {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "plan", "signature": "sig"},
            {"type": "redacted_thinking", "data": "opaque"},
            {"type": "text", "text": "Listing."},
            {
                "type": "tool_use",
                "id": "tu_1",
                "name": "run_shell",
                "input": call["args"],
            },
        ],
    }


def test_every_request_of_a_run_shares_its_system_prefix_and_is_cached():
    model = _model()
    first, second = (
        model._build_payload(messages)
        for messages in ([HumanMessage("go")], [HumanMessage("go"), AIMessage("ok")])
    )
    assert first["system"] == second["system"]
    assert first["cache_control"] == {"type": "ephemeral", "ttl": "5m"}
    off = ChatClaude(model="claude-haiku-5-5", token="t", prompt_cache_ttl=None)
    assert "cache_control" not in off._build_payload([HumanMessage("go")])


def _sse(*events):
    return [f"data: {json.dumps(event)}" for event in events]


def _serve_stream(monkeypatch, lines):
    class Response:
        is_error = False

        def iter_lines(self):
            yield from lines

    class Client:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        @contextmanager
        def stream(self, method, url, headers=None, json=None):
            del method, url, headers, json
            yield Response()

    monkeypatch.setattr(claude.httpx, "Client", Client)


_START = {
    "type": "message_start",
    "message": {
        "id": "msg_1",
        "model": "claude-haiku-5-5",
        "usage": {"input_tokens": 10},
    },
}


def test_a_streamed_answer_reports_usage_in_langchain_terms(monkeypatch):
    usage = {
        "input_tokens": 10,
        "cache_read_input_tokens": 90,
        "cache_creation_input_tokens": 5,
        "output_tokens": 7,
        "output_tokens_details": {"thinking_tokens": 4},
    }
    _serve_stream(
        monkeypatch,
        _sse(
            _START,
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "plan"},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": "sig"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "run_shell",
                    "input": {},
                },
            },
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '{"command": "ls"}',
                },
            },
            {"type": "content_block_stop", "index": 1},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": usage,
            },
            {"type": "message_stop"},
        ),
    )
    chunks = list(_model().stream([HumanMessage("go")]))
    message = sum(chunks[1:], chunks[0])

    assert message.usage_metadata == {
        "input_tokens": 105,
        "output_tokens": 7,
        "total_tokens": 112,
        "input_token_details": {"cache_read": 90, "cache_creation": 5},
        "output_token_details": {"reasoning": 4},
    }
    assert message.response_metadata["usage"] == usage
    (reasoning,) = [b for b in message.content if b["type"] == "reasoning"]
    assert (reasoning["reasoning"], reasoning["signature"]) == ("plan", "sig")
    assert message.tool_calls[0]["args"] == {"command": "ls"}


@pytest.mark.parametrize(
    ("lines", "status"),
    [
        # Overloaded after the stream opened with HTTP 200.
        (
            _sse(
                _START,
                {
                    "type": "error",
                    "error": {"type": "overloaded_error", "message": "Overloaded"},
                },
            ),
            529,
        ),
        # Cut short: no message_stop, so no whole answer.
        (
            _sse(
                _START,
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            None,
        ),
    ],
)
def test_a_failed_stream_raises_for_transport_retry(monkeypatch, lines, status):
    _serve_stream(monkeypatch, lines)
    with pytest.raises(ClaudeAPIError) as raised:
        list(_model().stream([HumanMessage("go")]))
    assert raised.value.status_code == status
    assert is_retryable_model_error(raised.value)


@pytest.mark.parametrize(
    ("status", "error_type", "retryable"),
    [
        (429, "rate_limit_error", True),
        (500, "api_error", True),
        (529, "overloaded_error", True),
        (400, "invalid_request_error", False),
        (401, "authentication_error", False),
    ],
)
def test_http_errors_are_retried_only_when_they_can_clear(
    status, error_type, retryable
):
    response = httpx.Response(
        status,
        json={"type": "error", "error": {"type": error_type, "message": "m"}},
        request=httpx.Request("POST", claude.ANTHROPIC_MESSAGES_URL),
    )
    with pytest.raises(ClaudeAPIError) as raised:
        _model()._handle_error(response)
    assert (raised.value.status_code, raised.value.error_type) == (status, error_type)
    assert is_retryable_model_error(raised.value) is retryable


def _thinking_turn(i: int, tokens: int) -> AIMessage:
    call = {"name": "run_shell", "args": {"command": "ls"}, "id": f"tu_{i}"}
    return AIMessage(
        content=[
            {"type": "reasoning", "reasoning": f"t{i}", "signature": f"s{i}"},
            {"type": "tool_call", **call},
        ],
        tool_calls=[call],
        usage_metadata={
            "input_tokens": 1,
            "output_tokens": tokens,
            "total_tokens": tokens + 1,
            "output_token_details": {"reasoning": tokens},
        },
    )


def _conversation(turns: int, tokens: int) -> list:
    messages = [HumanMessage("go")]
    for i in range(turns):
        messages += [
            _thinking_turn(i, tokens),
            ToolMessage("ok", tool_call_id=f"tu_{i}"),
        ]
    return messages


def _replayed(model: ChatClaude, messages: list) -> list[str]:
    return [
        block["signature"]
        for message in model._build_payload(messages)["messages"]
        for block in message["content"]
        if block.get("type") == "thinking"
    ]


def test_thinking_past_its_budget_goes_oldest_first_half_a_budget_at_a_time():
    """Each replayed block costs its full thinking count, so a long run must
    shed some; only a leading run can go without invalidating the rest."""
    model = ChatClaude(model="claude-haiku-5-5", token="t", thinking_replay_budget=100)

    assert _replayed(model, _conversation(5, 20)) == [f"s{i}" for i in range(5)]
    # 120 tokens: one 50-token step withholds the oldest three turns (60).
    assert _replayed(model, _conversation(6, 20)) == ["s3", "s4", "s5"]
    # 140 tokens is still one step, so the cut, and the cached prefix, hold.
    assert _replayed(model, _conversation(7, 20)) == ["s3", "s4", "s5", "s6"]
    # 160 tokens crosses a second step.
    assert _replayed(model, _conversation(8, 20)) == ["s5", "s6", "s7"]


def test_all_thinking_is_replayed_without_a_budget():
    model = ChatClaude(model="claude-haiku-5-5", token="t")
    assert _replayed(model, _conversation(8, 10**6)) == [f"s{i}" for i in range(8)]
