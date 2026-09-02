from typing import Any

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

from zeroshot.models import ChatOpenRouterSingleReasoning


def _delta(text: str, fmt: str | None = "unknown") -> ChatGenerationChunk:
    """One streamed reasoning delta as langchain-openrouter builds it."""
    return ChatGenerationChunk(
        message=AIMessageChunk(
            content="",
            additional_kwargs={
                "reasoning_content": text,
                "reasoning_details": [
                    {
                        "type": "reasoning.text",
                        "format": fmt,
                        "index": 0,
                        "text": text,
                    }
                ],
            },
        )
    )


def _model(
    monkeypatch: pytest.MonkeyPatch,
    deltas: list[tuple[str, str | None]],
) -> Any:
    model = ChatOpenRouterSingleReasoning(model="test", api_key="EMPTY")
    monkeypatch.setattr(
        type(model).__mro__[1],
        "_stream",
        lambda *_args, **_kwargs: iter([_delta(t, f) for t, f in deltas]),
    )
    return model


def _merged(model: Any) -> Any:
    merged = None
    for chunk in model._stream([], None, None):
        merged = chunk if merged is None else merged + chunk
    assert merged is not None
    return merged


def test_merged_stream_keeps_one_format(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model(monkeypatch, [("a ", "unknown"), ("b ", "unknown"), ("c", "unknown")])

    merged = _merged(model)

    (detail,) = merged.message.additional_kwargs["reasoning_details"]
    assert detail["format"] == "unknown"
    assert detail["text"] == "a b c"
    assert merged.message.additional_kwargs["reasoning_content"] == "a b c"


def test_a_null_first_format_is_still_filled_in_by_a_later_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dropping the repeats must not drop the value that replaces a null.

    `merge_dicts` lets a later value stand in for a `None` one, and the API
    refuses a null `format` as a string that is not a string.
    """
    model = _model(monkeypatch, [("a ", None), ("b ", "unknown"), ("c", "unknown")])

    merged = _merged(model)

    (detail,) = merged.message.additional_kwargs["reasoning_details"]
    assert detail["format"] == "unknown"
    assert detail["text"] == "a b c"


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


def test_payload_leaves_out_a_null_detail_field() -> None:
    """A null in a string-typed field is refused as 422 when it goes back."""
    model = ChatOpenRouterSingleReasoning(model="test", api_key="EMPTY")
    message = AIMessage(
        content="",
        additional_kwargs={
            "reasoning_details": [
                {"type": "reasoning.text", "format": None, "text": "a b c"}
            ],
        },
    )

    (message_dict,), _ = model._create_message_dicts([message], None)

    (detail,) = message_dict["reasoning_details"]
    assert "format" not in detail
    assert detail["text"] == "a b c"
