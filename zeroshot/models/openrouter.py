"""Undo the reasoning duplication `langchain-openrouter` puts on the wire.

OpenRouter returns a turn's chain of thought twice: as the `reasoning` string
and as `reasoning_details[].text`.  `_convert_message_to_dict` sends both back
on every subsequent request, so a tool-calling loop replays each turn's
reasoning twice for the rest of the run.

Streaming corrupts it further.  Every delta carries the same constant fields --
`format` above all -- and `AIMessageChunk.__add__` merges `reasoning_details`
entries through `merge_dicts`, which concatenates string values.  `merge_lists`
exempts `type`; nothing exempts `format`, so a block streamed in N deltas ends
up with `format` repeated N times.  One measured GLM run reached 240,926 bytes
of `"unknown"` in a single transcript, replayed on every later request.

`_merge_reasoning_details` in the package repairs the neighbouring failure --
`reasoning_details` left as many un-merged fragments -- and returns early on the
single merged entry this produces, so it does not reach either problem here.
"""

from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any, override

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_openrouter.chat_models import ChatOpenRouter

# Constant across a block's deltas, and concatenated by the chunk merge because
# it is a string the merge has no exemption for. `type` is already exempt in
# `merge_lists`; `index` and `id` are exempt in `merge_dicts`.
_CONSTANT_DETAIL_FIELDS = ("format", "signature", "provider", "model")

# Enough to name the field; a request holding an image must not be printed.
_SUSPECT_LIMIT = 20


def _thin_reasoning_details(details: Any, kept: set[Any]) -> None:
    """Keep each constant field once per block, and drop the repeats.

    Merging then reproduces the shape a non-streaming response would have had:
    the field appears once, carrying the value the provider sent.

    A field counts as kept only once a delta carries something for it, because
    `merge_dicts` lets a later value stand in for a `None` one and dropping the
    repeats must not take that repair away -- a block whose first delta has
    `format: null` would otherwise go back to the API with the null, and be
    refused as a string that is not a string.
    """
    if not isinstance(details, list):
        return
    for entry in details:
        if not isinstance(entry, dict):
            continue
        block = (entry.get("type"), entry.get("index"), entry.get("id"))
        for field in _CONSTANT_DETAIL_FIELDS:
            if field not in entry:
                continue
            if (block, field) in kept:
                entry.pop(field)
            elif entry[field] is not None:
                kept.add((block, field))


class ChatOpenRouterSingleReasoning(ChatOpenRouter):
    """Send each turn's reasoning once, and its constant fields once."""

    @override
    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        kept: set[Any] = set()
        try:
            for chunk in super()._stream(messages, stop, run_manager, **kwargs):
                _thin_chunk(chunk, kept)
                yield chunk
        except ValueError as error:
            raise self._named(error, messages, stop) from error

    @override
    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        kept: set[Any] = set()
        try:
            async for chunk in super()._astream(messages, stop, run_manager, **kwargs):
                _thin_chunk(chunk, kept)
                yield chunk
        except ValueError as error:
            raise self._named(error, messages, stop) from error

    def _named(
        self,
        error: ValueError,
        messages: list[BaseMessage],
        stop: list[str] | None,
    ) -> ValueError:
        """Name the fields a rejection of the request could have been about.

        The API says what a field should have been and not which field it was,
        so the request is rebuilt and scanned for what could match.  Values are
        left out: one of these fields is an image.
        """
        suspects = _suspect_paths(self._create_message_dicts(messages, stop)[0])
        if not suspects:
            return error
        return ValueError(f"{error} Suspect fields: {', '.join(suspects)}.")

    @override
    def _create_message_dicts(
        self, messages: list[BaseMessage], stop: list[str] | None
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        message_dicts, params = super()._create_message_dicts(messages, stop)
        for message_dict in message_dicts:
            _drop_block_ids(message_dict)
            _drop_null_detail_fields(message_dict)
            _drop_redundant_reasoning(message_dict)
        return _lift_tool_images(message_dicts), params


def _thin_chunk(chunk: ChatGenerationChunk, kept: set[Any]) -> None:
    message = chunk.message
    if isinstance(message, AIMessageChunk):
        _thin_reasoning_details(
            message.additional_kwargs.get("reasoning_details"), kept
        )


def _drop_block_ids(message_dict: dict[str, Any]) -> None:
    """Leave LangChain's own block id out of a content part.

    `create_text_block` stamps an `lc_` id on the block, and it survives the
    conversion to the wire.  Fireworks types a content part by its fields and
    refuses the extra one -- 400, `Input should be a valid string`, naming
    `messages[n].content.str`, which is the first branch of its `str | list`
    union rather than the part that actually failed.
    """
    content = message_dict.get("content")
    if not isinstance(content, list):
        return
    message_dict["content"] = [
        {key: value for key, value in part.items() if key != "id"}
        if isinstance(part, dict)
        else part
        for part in content
    ]


def _image_parts(content: Any) -> list[dict[str, Any]]:
    if not isinstance(content, list):
        return []
    return [
        part
        for part in content
        if isinstance(part, dict) and part.get("type") == "image_url"
    ]


def _lift_tool_images(
    message_dicts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Carry an image out of a tool result into a user message after it.

    A tool result is typed as a string, so an image inside one is refused --
    DeepInfra answers 422 `Input should be a valid string` for
    `messages[n].tool.content.str`.  A user message is where every provider
    takes an image, and it is the only place to put one that the tool result
    can be followed by.

    The images of one run of tool results are lifted together: a result has to
    reach the turn that called for it before any other role does.
    """
    lifted: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for message_dict in message_dicts:
        if message_dict.get("role") == "tool":
            images = _image_parts(message_dict.get("content"))
            if images:
                message_dict["content"] = _tool_result_text(
                    message_dict["content"], len(images)
                )
                pending.extend(images)
            lifted.append(message_dict)
            continue
        if pending:
            lifted.append(_image_message(pending))
            pending = []
        lifted.append(message_dict)
    if pending:
        lifted.append(_image_message(pending))
    return lifted


def _tool_result_text(content: list[Any], images: int) -> str:
    """What the tool result says once its images have been lifted out of it."""
    said = " ".join(
        part["text"]
        for part in content
        if isinstance(part, dict) and part.get("type") == "text" and part.get("text")
    )
    loaded = f"Loaded {images} image{'s' if images > 1 else ''}, attached below."
    return f"{said} {loaded}".strip()


def _image_message(images: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    f"{'Images' if len(images) > 1 else 'The image'} the tool loaded:"
                ),
            },
            *images,
        ],
    }


def _drop_null_detail_fields(message_dict: dict[str, Any]) -> None:
    """Say nothing rather than null in a reasoning detail.

    These fields are typed as strings, so a null the provider sent is refused
    when it goes back -- 422, `Input should be a valid string` -- while leaving
    the field out is what a response without it looks like anyway.
    """
    details = message_dict.get("reasoning_details")
    if not isinstance(details, list):
        return
    message_dict["reasoning_details"] = [
        {key: value for key, value in entry.items() if value is not None}
        if isinstance(entry, dict)
        else entry
        for entry in details
    ]


def _drop_redundant_reasoning(message_dict: dict[str, Any]) -> None:
    """Drop `reasoning` where `reasoning_details` already carries the same text.

    `reasoning_details` is what a provider resumes a thinking block from, so it
    is the copy to keep; `reasoning` is the flat rendering of it. Where the
    details hold no text -- a redacted or signature-only block -- `reasoning` is
    the only copy and stays.
    """
    reasoning = message_dict.get("reasoning")
    details = message_dict.get("reasoning_details")
    if not reasoning or not isinstance(details, Sequence) or isinstance(details, str):
        return
    if any(isinstance(entry, dict) and entry.get("text") for entry in details):
        message_dict.pop("reasoning", None)


def _suspect_paths(message_dicts: list[dict[str, Any]]) -> list[str]:
    """Where the request holds a null, or content that is not a string.

    These are the two shapes a provider that types a field as a string refuses,
    and neither is visible in what it sends back.
    """
    paths: list[str] = []
    for index, message_dict in enumerate(message_dicts):
        where = f"messages[{index}]"
        content = message_dict.get("content")
        if content is not None and not isinstance(content, str):
            paths.append(f"{where}.content is {type(content).__name__}")
        _walk(message_dict, where, paths)
    return paths[:_SUSPECT_LIMIT]


def _walk(value: Any, path: str, paths: list[str]) -> None:
    if value is None:
        paths.append(path)
    elif isinstance(value, dict):
        for key, item in value.items():
            _walk(item, f"{path}.{key}", paths)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _walk(item, f"{path}[{index}]", paths)
