from typing import Any

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.messages.content import (
    create_image_block,
    create_text_block,
)
from langchain_core.outputs import ChatGenerationChunk
from pydantic import BaseModel

from zeroshot.pipeline.models.openrouter import ChatOpenRouterSingleReasoning


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
    model = _model(
        monkeypatch, [("a ", "unknown"), ("b ", "unknown"), ("c", "unknown")]
    )

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


class _Echo(BaseModel):
    value: str


class _Answer(BaseModel):
    done: bool


def test_a_call_is_forced_only_when_the_answer_is_the_one_tool_left() -> None:
    """Forced to pick among several tools, GLM submitted near-empty answers."""
    model = ChatOpenRouterSingleReasoning(model="test", api_key="EMPTY")

    working: Any = model.bind_tools([_Echo, _Answer], tool_choice="any")
    final: Any = model.bind_tools([_Answer], tool_choice="any")

    assert working.kwargs["tool_choice"] == "auto"
    assert final.kwargs["tool_choice"] == "required"


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

    assert "messages[0].reasoning_details[0].extras.format" in _error_of(model, message)


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


def _image(name: str) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {"url": f"https://example.test/{name}.png"},
    }


@pytest.mark.parametrize("limit", [None, 8, 1])
def test_image_cap_keeps_sources_newest_images_and_complete_parallel_groups(limit):
    model = ChatOpenRouterSingleReasoning(
        model="test",
        api_key="EMPTY",
        **({} if limit is None else {"max_images_per_request": limit}),
    )
    assert model.max_images_per_request == limit
    messages: list[BaseMessage] = [HumanMessage(content=[_image("source")])]
    for start in (0, 4):
        messages.append(
            AIMessage(
                content=f"Inspect group {start}.",
                tool_calls=[
                    {
                        "name": "load_image",
                        "args": {"image_path": f"{i}.png"},
                        "id": str(i),
                    }
                    for i in range(start, start + 4)
                ],
            )
        )
        messages.extend(
            ToolMessage(
                content=[{"type": "text", "text": f"Read {i}."}, _image(str(i))],
                tool_call_id=str(i),
            )
            for i in range(start, start + 4)
        )
    before = [message.model_dump() for message in messages]

    dicts, params = model._create_message_dicts(messages, None)

    kept = range(8) if limit is None else range(9 - limit, 8)
    urls = [
        part["image_url"]["url"]
        for message in dicts
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if part.get("type") == "image_url"
    ]
    assert urls == [
        _image(name)["image_url"]["url"] for name in ["source", *map(str, kept)]
    ]
    assert dicts[0]["content"] == [_image("source")]
    assert [message.model_dump() for message in messages] == before
    assert "max_images_per_request" not in params
    assert "max_images_per_request" not in model.model_kwargs
    for index, message in enumerate(dicts):
        if message["role"] == "assistant":
            calls = [call["id"] for call in message["tool_calls"]]
            results = dicts[index + 1 : index + 1 + len(calls)]
            assert [result["role"] for result in results] == ["tool"] * len(calls)
            assert [result["tool_call_id"] for result in results] == calls
    results = [message for message in dicts if message["role"] == "tool"]
    assert len(results) == 8
    for i, result in enumerate(results):
        assert isinstance(result["content"], str)
        assert f"Read {i}." in result["content"]
        assert ("Previously loaded image omitted" in result["content"]) == (
            i not in kept
        )
        assert "Loaded 0 images" not in result["content"]


def test_image_cap_retains_newest_parts_within_one_tool_result():
    model = ChatOpenRouterSingleReasoning(
        model="test", api_key="EMPTY", max_images_per_request=2
    )
    messages = [
        ToolMessage(
            content=[
                _image("old"),
                {"type": "text", "text": "Read all three."},
                _image("new"),
                _image("newest"),
            ],
            tool_call_id="images",
        )
    ]

    (result, carrier), _ = model._create_message_dicts(messages, None)

    assert result["tool_call_id"] == "images"
    assert "Read all three." in result["content"]
    assert "Previously loaded image omitted" in result["content"]
    assert "Loaded 2 images, attached below." in result["content"]
    assert carrier["content"][1:] == [_image("new"), _image("newest")]
    assert len(messages[0].content) == 4


def test_image_cap_refuses_to_drop_original_attachments():
    model = ChatOpenRouterSingleReasoning(
        model="test", api_key="EMPTY", max_images_per_request=1
    )
    message = HumanMessage(content=[_image("source_1"), _image("source_2")])
    before = message.model_dump()

    with pytest.raises(
        ValueError,
        match=r"Original image attachments \(2\).*max_images_per_request \(1\)",
    ):
        model._create_message_dicts([message], None)

    assert message.model_dump() == before


@pytest.mark.parametrize("limit", [0, -1, 1.5, True])
def test_image_cap_requires_a_positive_integer(limit):
    with pytest.raises(ValueError, match="max_images_per_request"):
        ChatOpenRouterSingleReasoning(
            model="test", api_key="EMPTY", max_images_per_request=limit
        )
