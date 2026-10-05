import json

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
