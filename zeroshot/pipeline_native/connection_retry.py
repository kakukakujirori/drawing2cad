"""Model-call retries for the native baseline."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any, override

from langchain.agents import AgentState as _AgentState
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import HumanMessage

from zeroshot.pipeline.workflow.middleware.connection_retry import (
    _backoff_delay,
    is_retryable_model_error,
)

logger = logging.getLogger(__name__)


class ModelResponseError(RuntimeError):
    """A response that ended in an error or exhausted its output limit."""

    def __init__(self, finish_reason: str, generation_id: str | None) -> None:
        self.finish_reason = finish_reason
        super().__init__(
            f"Model response ended with finish_reason={finish_reason!r} "
            f"(generation_id={generation_id!r})"
        )


class ConnectionRetryMiddleware(AgentMiddleware[_AgentState[Any], None, Any]):
    """Retry transport failures and error responses, optionally output limits.

    Shares the staged pipeline's transport policy. Output-limit retries only
    add a brief request to finish; there is no answer validation or tool forcing.
    All retries share the existing per-call budget.
    """

    def __init__(
        self,
        max_retries: int = 5,
        role: str = "",
        *,
        length_retry_prompt: str | None = None,
    ) -> None:
        super().__init__()
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.max_retries = max_retries
        self.role = role
        self.length_retry_prompt = length_retry_prompt

    def _check_response(self, response: ModelResponse[Any]) -> ModelResponse[Any]:
        last = response.result[-1] if response.result else None
        metadata = getattr(last, "response_metadata", None) or {}
        reason = metadata.get("finish_reason")
        if reason == "error" or (
            reason == "length" and self.length_retry_prompt is not None
        ):
            raise ModelResponseError(reason, metadata.get("id"))
        return response

    def _length_retry_request(self, request: ModelRequest[None]) -> ModelRequest[None]:
        # Reuse the history before the truncated response, with just one nudge.
        assert self.length_retry_prompt is not None
        return request.override(
            messages=[*request.messages, HumanMessage(content=self.length_retry_prompt)]
        )

    def _retry(self, attempt: int, error: Exception) -> bool:
        retrying = (
            isinstance(error, ModelResponseError) or is_retryable_model_error(error)
        ) and attempt < self.max_retries
        request_adjusted = (
            retrying
            and isinstance(error, ModelResponseError)
            and error.finish_reason == "length"
        )
        payload: dict[str, Any] = {
            "role": self.role,
            "attempt": attempt + 1,
            "max_attempts": self.max_retries + 1,
            "error_type": type(error).__qualname__,
            "error": str(error)[:500],
            "retrying": retrying,
            "request_adjusted": request_adjusted,
        }
        if isinstance(error, ModelResponseError):
            payload["finish_reason"] = error.finish_reason
        if request_adjusted:
            payload["retry_prompt"] = self.length_retry_prompt
        logger.warning("model_retry: %s", payload)
        return retrying

    @override
    def wrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        attempt = 0
        current_request = request
        while True:
            try:
                return self._check_response(handler(current_request))
            except Exception as error:
                if not self._retry(attempt, error):
                    raise
                attempt += 1
                if (
                    isinstance(error, ModelResponseError)
                    and error.finish_reason == "length"
                ):
                    current_request = self._length_retry_request(request)
                else:
                    time.sleep(_backoff_delay(attempt))

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        attempt = 0
        current_request = request
        while True:
            try:
                return self._check_response(await handler(current_request))
            except Exception as error:
                if not self._retry(attempt, error):
                    raise
                attempt += 1
                if (
                    isinstance(error, ModelResponseError)
                    and error.finish_reason == "length"
                ):
                    current_request = self._length_retry_request(request)
                else:
                    await asyncio.sleep(_backoff_delay(attempt))
