import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.content import create_image_block
from langchain_openai.chat_models.codex import _ChatOpenAICodex

from zeroshot.pipeline.models.codex import ChatCodex
from zeroshot.pipeline.models.image_history import INPUT_IMAGE_ID


def test_image_budget_keeps_input_drawings_and_the_newest_images(monkeypatch):
    def image(name):
        return {
            "type": "image_url",
            "image_url": {"url": f"https://example.test/{name}.png"},
        }

    def report(name):
        return HumanMessage(
            content=[
                {"type": "text", "text": "[Execution result]"},
                image(name),
                image(name + "2"),
                image(name + "3"),
            ],
        )

    # Use the real serializer, stubbing only OAuth so no token/network is needed.
    monkeypatch.setattr(_ChatOpenAICodex, "_codex_headers_sync", lambda self: {})
    model = ChatCodex(model="gpt-6-luna", image_history_limit=6)
    messages = [
        HumanMessage(
            content=[
                create_image_block(
                    url="https://example.test/drawing.png", id=INPUT_IMAGE_ID
                )
            ]
        ),
        report("old"),
        report("latest"),
        *[
            item
            for number in range(6)
            for item in (
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "load_image",
                            "id": str(number),
                            "args": {"image_path": f"/work/{number}.png"},
                        }
                    ],
                ),
                ToolMessage(content=[image(str(number))], tool_call_id=str(number)),
            )
        ],
    ]
    before = [message.model_dump() for message in messages]
    payload = model._get_request_payload(messages)

    def retained_images(payload):
        return [
            part["image_url"]
            for item in payload["input"]
            for part in item.get("content", item.get("output", []))
            if isinstance(part, dict) and part["type"] == "input_image"
        ]

    names = ["drawing", "1", "2", "3", "4", "5"]
    assert retained_images(payload) == [
        image(name)["image_url"]["url"] for name in names
    ]
    omitted_tool = next(
        item
        for item in payload["input"]
        if item.get("type") == "function_call_output" and item["call_id"] == "0"
    )
    assert omitted_tool["output"] == [
        {"type": "input_text", "text": "Image omitted from this request."}
    ]
    original_call = next(
        item
        for item in payload["input"]
        if item.get("type") == "function_call" and item["call_id"] == "0"
    )
    assert json.loads(original_call["arguments"]) == {"image_path": "/work/0.png"}
    assert [message.model_dump() for message in messages] == before
    assert "image_history_limit" not in payload
    assert "image_history_limit" not in model.model_kwargs

    # No budget reproduces the parent payload; a tiny budget keeps the drawing.
    model.image_history_limit = None
    assert model._get_request_payload(
        messages
    ) == _ChatOpenAICodex._get_request_payload(model, messages)
    model.image_history_limit = 1
    payload = model._get_request_payload(messages)
    assert retained_images(payload) == [image("drawing")["image_url"]["url"]]

    # Input drawings take priority over the history budget and do not abort a run.
    drawings = [
        messages[0],
        HumanMessage(
            content=[
                create_image_block(
                    url="https://example.test/second.png", id=INPUT_IMAGE_ID
                )
            ]
        ),
    ]
    assert retained_images(model._get_request_payload(drawings)) == [
        image("drawing")["image_url"]["url"],
        image("second")["image_url"]["url"],
    ]


class _TestTokenProvider:
    def get_token(self):
        return SimpleNamespace(access_token="test-token", account_id=None)

    async def aget_token(self):
        return self.get_token()

    def get_access_token(self):
        return self.get_token().access_token

    async def aget_access_token(self):
        return self.get_access_token()


def _tool_response_events(delivery, interleaved):
    calls = [
        {
            "type": "function_call",
            "id": "fc_shell",
            "call_id": "call_shell",
            "name": "run_shell",
            "arguments": '{"command":"pwd"}',
            "status": "completed",
        },
        {
            "type": "function_call",
            "id": "fc_image",
            "call_id": "call_image",
            "name": "load_image",
            "arguments": '{"image_path":"/work/front.png"}',
            "status": "completed",
        },
    ]
    if delivery == "single_delta":
        calls = calls[:1]
    response = {
        "id": "resp_test",
        "object": "response",
        "created_at": 0,
        "status": "in_progress",
        "model": "gpt-6-luna",
        "output": [],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": None,
    }
    events = [{"type": "response.created", "response": response}]
    reasoning = {"type": "reasoning", "id": "rs_test", "summary": []}
    events.extend(
        [
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": reasoning,
            },
            {
                "type": "response.reasoning_summary_text.delta",
                "output_index": 0,
                "summary_index": 0,
                "item_id": "rs_test",
                "delta": "Inspecting views.",
            },
        ]
    )

    def added(index, call):
        arguments = call["arguments"] if delivery == "added_full" else ""
        return {
            "type": "response.output_item.added",
            "output_index": index,
            "item": {**call, "arguments": arguments, "status": "in_progress"},
        }

    def deltas(index, call):
        if delivery in ("none", "added_full"):
            return []
        arguments = call["arguments"]
        parts = [arguments[:5], arguments[5:]]
        if delivery == "partial":
            parts = parts[:1]
        return [
            {
                "type": "response.function_call_arguments.delta",
                "output_index": index,
                "item_id": call["id"],
                "delta": part,
            }
            for part in parts
        ]

    def done(index, call):
        return [
            {
                "type": "response.function_call_arguments.done",
                "output_index": index,
                "item_id": call["id"],
                "arguments": call["arguments"],
            },
            {"type": "response.output_item.done", "output_index": index, "item": call},
        ]

    if interleaved:
        events.extend(added(index, call) for index, call in enumerate(calls, 1))
        streams = [deltas(index, call) for index, call in enumerate(calls, 1)]
        for parts in zip(*streams):
            events.extend(parts)
        for index, call in reversed(list(enumerate(calls, 1))):
            events.extend(done(index, call))
    else:
        for index, call in enumerate(calls, 1):
            events.append(added(index, call))
            events.extend(deltas(index, call))
            events.extend(done(index, call))
    events.append(
        {
            "type": "response.completed",
            "response": {
                **response,
                "status": "completed",
                "output": [reasoning, *calls],
                "usage": {
                    "input_tokens": 2,
                    "output_tokens": 10,
                    "total_tokens": 12,
                    "input_tokens_details": {"cached_tokens": 1},
                    "output_tokens_details": {"reasoning_tokens": 3},
                },
            },
        }
    )
    for number, event in enumerate(events):
        event["sequence_number"] = number
    return calls, events


@pytest.mark.parametrize("asynchronous", [False, True], ids=["invoke", "ainvoke"])
@pytest.mark.parametrize(
    ("delivery", "interleaved"),
    [
        ("single_delta", False),
        ("none", False),
        ("full", False),
        ("partial", False),
        ("added_full", False),
        ("none", True),
        ("full", True),
    ],
)
def test_completed_tool_arguments_survive_codex_stream(
    delivery, interleaved, asynchronous
):
    calls, events = _tool_response_events(delivery, interleaved)
    sse = "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode()

    def respond(request):
        payload = json.loads(request.content)
        assert payload["stream"] is True
        return httpx.Response(
            200,
            content=sse,
            headers={
                "content-type": "text/event-stream",
                "x-request-id": "request_test",
            },
        )

    class Tokens(BaseCallbackHandler):
        def __init__(self):
            self.chunks = []

        def on_llm_new_token(self, token, *, chunk, **kwargs):
            self.chunks.append(chunk.message)

    model = ChatCodex(
        model="gpt-6-luna",
        token_provider=_TestTokenProvider(),
        max_retries=0,
        include_response_headers=True,
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
        http_async_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )
    tokens = Tokens()
    try:
        if asynchronous:
            message = asyncio.run(
                model.ainvoke("Inspect the views.", config={"callbacks": [tokens]})
            )
        else:
            message = model.invoke("Inspect the views.", config={"callbacks": [tokens]})
    finally:
        model.root_client.close()
        asyncio.run(model.root_async_client.close())

    expected = {
        call["call_id"]: {
            "type": "tool_call",
            "id": call["call_id"],
            "name": call["name"],
            "args": json.loads(call["arguments"]),
        }
        for call in calls
    }
    assert {call["id"]: call for call in message.tool_calls} == expected
    assert message.invalid_tool_calls == []
    assert message.usage_metadata["total_tokens"] == 12
    assert message.response_metadata["headers"]["x-request-id"] == "request_test"
    assert any(
        summary.get("text") == "Inspecting views."
        for chunk in tokens.chunks
        for part in chunk.content
        if isinstance(part, dict)
        for summary in part.get("summary", [])
    )
    content_calls = {
        part["call_id"]: part
        for part in message.content
        if isinstance(part, dict) and part.get("type") == "function_call"
    }
    assert {
        key: json.loads(part["arguments"]) for key, part in content_calls.items()
    } == {key: call["args"] for key, call in expected.items()}
