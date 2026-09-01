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


def _thin_reasoning_details(details: Any, seen: set[Any]) -> None:
    """Keep each constant field on the first delta of a block, drop the rest.

    Merging then reproduces the shape a non-streaming response would have had:
    the field appears once, with the value the provider sent.
    """
    if not isinstance(details, list):
        return
    for entry in details:
        if not isinstance(entry, dict):
            continue
        key = (entry.get("type"), entry.get("index"), entry.get("id"))
        first = key not in seen
        seen.add(key)
        if first:
            continue
        for field in _CONSTANT_DETAIL_FIELDS:
            entry.pop(field, None)


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
        seen: set[Any] = set()
        for chunk in super()._stream(messages, stop, run_manager, **kwargs):
            _thin_chunk(chunk, seen)
            yield chunk

    @override
    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        seen: set[Any] = set()
        async for chunk in super()._astream(messages, stop, run_manager, **kwargs):
            _thin_chunk(chunk, seen)
            yield chunk

    @override
    def _create_message_dicts(
        self, messages: list[BaseMessage], stop: list[str] | None
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        message_dicts, params = super()._create_message_dicts(messages, stop)
        for message_dict in message_dicts:
            _drop_redundant_reasoning(message_dict)
        return message_dicts, params


def _thin_chunk(chunk: ChatGenerationChunk, seen: set[Any]) -> None:
    message = chunk.message
    if isinstance(message, AIMessageChunk):
        _thin_reasoning_details(
            message.additional_kwargs.get("reasoning_details"), seen
        )


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
    if any(
        isinstance(entry, dict) and entry.get("text") for entry in details
    ):
        message_dict.pop("reasoning", None)
