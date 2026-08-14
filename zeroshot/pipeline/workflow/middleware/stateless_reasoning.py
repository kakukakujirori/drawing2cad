"""Keep a reasoning chain alive across turns on a backend that stores nothing.

The Codex backend pins `store=false`, so a reasoning item is replayable only
through its own `encrypted_content`.  That payload is streamed -- but on
`response.output_item.done`, and `langchain_openai` reads reasoning items only
from `response.output_item.added`, where the field is still an empty string.
The finished item matches none of the converter's branches and is dropped, so
the `AIMessage` keeps an `rs_` id pointing at content nobody kept.  The next
turn replays that bare id and the request fails with 404 `Item with id
'rs_...' not found`.

The fix is to build the message correctly in the first place: emit the missing
block while the stream is still being converted, so the payload lands in the
`AIMessage` itself and travels with it into the state, the event log and the
checkpoint.  `StatelessReasoningMiddleware` is then only a guard for reasoning
that arrived before this module did -- from an older checkpoint, say -- because
a bare id is exactly what the backend rejects.

Upstream equivalent: `langchain-ai/langchainjs#10844`.
"""

from collections.abc import Awaitable, Callable, Iterable
from typing import Any, TypeGuard, override

import langchain_openai.chat_models.base as _openai_base
from langchain.agents import AgentState as _AgentState
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, AIMessageChunk, AnyMessage
from langchain_core.outputs import ChatGenerationChunk

_installed = False


def install_reasoning_repair() -> None:
    """Emit the reasoning payload the streaming converter drops.

    Wraps the converter rather than reimplementing it: every chunk is handed to
    the original first, and only a finished reasoning item it declined to
    convert produces anything extra.  The extra block carries the same content
    index, so `AIMessageChunk` merging folds it into the block the `added`
    event opened -- filling in `encrypted_content` and touching nothing else.

    Idempotent, so constructing many agents installs one wrapper.  It also
    stands down on its own: should a future `langchain_openai` convert the
    finished item itself, the original returns a chunk and this adds nothing.
    """
    global _installed
    if _installed:
        return
    original = _openai_base._convert_responses_chunk_to_generation_chunk

    def convert(
        chunk: Any,
        current_index: int,
        current_output_index: int,
        current_sub_index: int,
        **kwargs: Any,
    ) -> tuple[int, int, int, ChatGenerationChunk | None]:
        result = original(
            chunk, current_index, current_output_index, current_sub_index, **kwargs
        )
        # `v0` projects reasoning into `additional_kwargs` by a different route.
        # Only the Responses-shaped content blocks are ours to complete.
        if kwargs.get("output_version") == "v0" or result[3] is not None:
            return result
        item = getattr(chunk, "item", None)
        if (
            getattr(chunk, "type", None) != "response.output_item.done"
            or getattr(item, "type", None) != "reasoning"
        ):
            return result
        payload = getattr(item, "encrypted_content", None)
        item_id = getattr(item, "id", None)
        if not payload or not item_id:
            return result

        # Mirror the converter's own `_advance`: a new output item takes the
        # next content index, the item already open keeps its own.
        index = current_index
        if current_output_index != chunk.output_index:
            index += 1
        block = {
            "type": "reasoning",
            "id": item_id,
            "encrypted_content": payload,
            # Carrying `summary` is what routes the block through the v1
            # translator's reasoning explosion, which re-indexes it to the
            # `lc_rs_*` index the `added` block was given. Without the key the
            # translator passes it through on its raw integer index and it
            # lands as a second, separate reasoning block -- which is how the
            # streaming path differs from plain aggregation, where the raw
            # indexes match and it merges either way.
            "summary": [],
            "index": index,
        }
        # `model_provider` is what selects the provider's block translator.
        # Without it the chunk is never translated, so it keeps the raw integer
        # index while its siblings are re-indexed, and merges with nothing.
        message = AIMessageChunk(
            content=[block],  # type: ignore[arg-type]
            response_metadata={"model_provider": "openai"},
        )
        return (
            index,
            chunk.output_index,
            current_sub_index,
            ChatGenerationChunk(message=message),
        )

    _openai_base._convert_responses_chunk_to_generation_chunk = convert
    _installed = True


def _is_reasoning(block: Any) -> TypeGuard[dict[str, Any]]:
    return isinstance(block, dict) and block.get("type") == "reasoning"


def _payload(block: dict[str, Any]) -> str:
    extras = block.get("extras") or {}
    return extras.get("encrypted_content") or block.get("encrypted_content") or ""


def _drop_bare_reasoning(messages: Iterable[AnyMessage]) -> list[AnyMessage]:
    """Drop reasoning only where the whole item is beyond recovery.

    One reasoning item becomes one v1 block per summary part, and the payload
    rides on the first of them alone.  Judging a block on its own contents
    would therefore throw away every part after the first -- summary the model
    wrote about its own thinking -- so recoverability is decided per item id,
    which is what the backend resolves anyway.
    """
    kept_messages: list[AnyMessage] = []
    for message in messages:
        content = message.content
        if not isinstance(message, AIMessage) or not isinstance(content, list):
            kept_messages.append(message)
            continue
        recoverable = {
            block.get("id")
            for block in content
            if _is_reasoning(block) and _payload(block)
        }
        kept = [
            block
            for block in content
            if not (_is_reasoning(block) and block.get("id") not in recoverable)
        ]
        if len(kept) == len(content):
            kept_messages.append(message)
        else:
            # A copy: the state's message list outlives this one request.
            kept_messages.append(message.model_copy(update={"content": kept}))
    return kept_messages


class StatelessReasoningMiddleware(AgentMiddleware[_AgentState[Any], None, Any]):
    """Withhold reasoning ids a stateless backend could not resolve.

    Installs the converter repair, which is what actually keeps the reasoning
    chain intact; anything still missing a payload by the time it is sent
    predates that repair, and would go out as a bare id.  Sending nothing is
    the only alternative the backend accepts.

    Applies only when the model is pinned to `store=False`. Anywhere else the
    id is a valid server-side reference and dropping the block would cost the
    model its chain of thought for nothing.
    """

    def __init__(self) -> None:
        super().__init__()
        install_reasoning_repair()

    @staticmethod
    def _sanitised(request: ModelRequest[None]) -> ModelRequest[None]:
        if getattr(request.model, "store", None) is not False:
            return request
        return request.override(messages=_drop_bare_reasoning(request.messages))

    @override
    def wrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        return handler(self._sanitised(request))

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        return await handler(self._sanitised(request))
