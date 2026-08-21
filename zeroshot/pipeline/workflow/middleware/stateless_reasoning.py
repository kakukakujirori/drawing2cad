"""Withhold reasoning a stateless backend could not resolve.

The Codex backend pins `store=false`, so a reasoning item is replayable only
through its own `encrypted_content`.  `langchain_openai` once read reasoning
items from `response.output_item.added` alone, where that field is still an
empty string, and dropped the finished item that carried the payload -- leaving
an `rs_` id pointing at content nobody kept, which the next turn replays and
the backend rejects with 404 `Item with id 'rs_...' not found`.  This module
used to fill the payload in itself, from the stream, as the message was built.

`langchain_openai` supplies the payload now, so that repair is gone.  It had
outlived its premise in a way it could not detect: rather than filling a blank
it appended to a payload already there, and the doubled string reached the
backend as 400 `invalid_encrypted_content`.

What remains is the guard.  A message can still arrive holding a reasoning id
with no payload behind it -- from a checkpoint written before any of this, say
-- and a bare id is exactly what the backend rejects.

Upstream equivalent: `langchain-ai/langchainjs#10844`.
"""

from collections.abc import Awaitable, Callable, Iterable
from typing import Any, TypeGuard, override

from langchain.agents import AgentState as _AgentState
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, AnyMessage


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

    A reasoning block whose payload is missing would go out as a bare id, and
    sending nothing is the only alternative the backend accepts.

    Applies only when the model is pinned to `store=False`. Anywhere else the
    id is a valid server-side reference and dropping the block would cost the
    model its chain of thought for nothing.
    """

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
