from typing import Any

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    ToolMessage,
)
from langchain_core.messages.content import (
    create_image_block,
    create_text_block,
)
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


def _failing_model(monkeypatch: pytest.MonkeyPatch) -> Any:
    model = ChatOpenRouterSingleReasoning(model="test", api_key="EMPTY")

    def _raise(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("... Input should be a valid string (code: 422)")
        yield  # pragma: no cover - a generator that only ever raises

    monkeypatch.setattr(type(model).__mro__[1], "_stream", _raise)
    return model


def _error_of(model: Any, message: AIMessage) -> str:
    with pytest.raises(ValueError) as raised:
        list(model._stream([message], None, None))
    return str(raised.value)


def test_a_rejected_request_names_a_null_the_repairs_left_behind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The nulls that are stripped sit at the top of a detail, so one nested
    inside it survives to the wire and is what the rejection is about."""
    model = _failing_model(monkeypatch)
    message = AIMessage(
        content="",
        additional_kwargs={
            "reasoning_details": [
                {"type": "reasoning.text", "text": "a", "extras": {"format": None}}
            ],
        },
    )

    assert "messages[0].reasoning_details[0].extras.format" in _error_of(
        model, message
    )


def test_a_rejected_request_names_content_that_is_not_a_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _failing_model(monkeypatch)
    message = AIMessage(content=[{"type": "text", "text": "a"}])

    assert "messages[0].content is list" in _error_of(model, message)


def test_a_rejection_with_nothing_to_blame_is_left_as_it_was(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _failing_model(monkeypatch)

    error = _error_of(model, AIMessage(content="a"))

    assert error == "... Input should be a valid string (code: 422)"


_IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}}


def test_payload_leaves_out_the_block_id_langchain_stamps_on_a_part() -> None:
    """Fireworks types a content part by its fields and refuses the extra one."""
    model = ChatOpenRouterSingleReasoning(model="test", api_key="EMPTY")
    message = HumanMessage(
        content=[
            create_text_block("a sheet"),
            create_image_block(base64="QUJD", mime_type="image/png"),
        ]
    )

    (message_dict,), _ = model._create_message_dicts([message], None)

    assert all("id" not in part for part in message_dict["content"])
    assert message_dict["content"][0] == {"type": "text", "text": "a sheet"}


def test_payload_carries_a_tool_image_in_a_user_message_after_it() -> None:
    """A tool result is typed as a string, so DeepInfra refuses an image in one."""
    model = ChatOpenRouterSingleReasoning(model="test", api_key="EMPTY")
    messages = [
        ToolMessage(content=[_IMAGE], tool_call_id="1"),
        HumanMessage(content="what is it"),
    ]

    dicts, _ = model._create_message_dicts(messages, None)

    result, carrier, question = dicts
    assert isinstance(result["content"], str)
    assert "Loaded 1 image" in result["content"]
    assert carrier["role"] == "user"
    assert carrier["content"][1] == _IMAGE
    assert question["content"] == "what is it"


def test_the_images_of_parallel_tool_calls_arrive_after_the_last_result() -> None:
    """A result has to reach the turn that called for it before any other role,
    so two results in a row are not split by the message carrying their images."""
    model = ChatOpenRouterSingleReasoning(model="test", api_key="EMPTY")
    messages = [
        ToolMessage(content=[_IMAGE], tool_call_id="1"),
        ToolMessage(content=[_IMAGE], tool_call_id="2"),
    ]

    dicts, _ = model._create_message_dicts(messages, None)

    assert [message_dict["role"] for message_dict in dicts] == [
        "tool",
        "tool",
        "user",
    ]
    assert len(dicts[2]["content"]) == 3


def test_a_tool_result_that_is_only_text_is_left_alone() -> None:
    model = ChatOpenRouterSingleReasoning(model="test", api_key="EMPTY")

    dicts, _ = model._create_message_dicts(
        [ToolMessage(content="done", tool_call_id="1")], None
    )

    assert [message_dict["role"] for message_dict in dicts] == ["tool"]
    assert dicts[0]["content"] == "done"
