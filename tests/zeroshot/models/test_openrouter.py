from typing import Any

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

from zeroshot.models import ChatOpenRouterSingleReasoning


def _delta(text: str) -> ChatGenerationChunk:
    """One streamed reasoning delta as langchain-openrouter builds it."""
    return ChatGenerationChunk(
        message=AIMessageChunk(
            content="",
            additional_kwargs={
                "reasoning_content": text,
                "reasoning_details": [
                    {
                        "type": "reasoning.text",
                        "format": "unknown",
                        "index": 0,
                        "text": text,
                    }
                ],
            },
        )
    )


def _model(monkeypatch: pytest.MonkeyPatch, deltas: list[str]) -> Any:
    model = ChatOpenRouterSingleReasoning(model="test", api_key="EMPTY")
    monkeypatch.setattr(
        type(model).__mro__[1],
        "_stream",
        lambda *_args, **_kwargs: iter([_delta(text) for text in deltas]),
    )
    return model


def test_merged_stream_keeps_one_format(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model(monkeypatch, ["a ", "b ", "c"])

    merged = None
    for chunk in model._stream([], None, None):
        merged = chunk if merged is None else merged + chunk

    assert merged is not None
    (detail,) = merged.message.additional_kwargs["reasoning_details"]
    assert detail["format"] == "unknown"
    assert detail["text"] == "a b c"
    assert merged.message.additional_kwargs["reasoning_content"] == "a b c"


def test_payload_sends_reasoning_once() -> None:
    model = ChatOpenRouterSingleReasoning(model="test", api_key="EMPTY")
    message = AIMessage(
        content="",
        additional_kwargs={
            "reasoning_content": "a b c",
            "reasoning_details": [
                {"type": "reasoning.text", "format": "unknown", "text": "a b c"}
            ],
        },
    )

    (message_dict,), _ = model._create_message_dicts([message], None)

    assert "reasoning" not in message_dict
    assert message_dict["reasoning_details"][0]["text"] == "a b c"


def test_payload_keeps_reasoning_without_detail_text() -> None:
    model = ChatOpenRouterSingleReasoning(model="test", api_key="EMPTY")
    message = AIMessage(
        content="",
        additional_kwargs={
            "reasoning_content": "a b c",
            "reasoning_details": [{"type": "reasoning.encrypted", "data": "xx"}],
        },
    )

    (message_dict,), _ = model._create_message_dicts([message], None)

    assert message_dict["reasoning"] == "a b c"
