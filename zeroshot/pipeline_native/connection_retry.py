"""Transport retry middleware for native agent pipelines."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any, override

from langchain.agents import AgentState as _AgentState
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse

from zeroshot.pipeline.workflow.middleware.connection_retry import (
    _backoff_delay,
    is_retryable_model_error,
)

logger = logging.getLogger(__name__)


class ConnectionRetryMiddleware(AgentMiddleware[_AgentState[Any], None, Any]):
    """Retry transport failures (dropped streams, timeouts, HTTP 429/5xx).

    Delegates retryability checks and backoff delays to the shared transport
    policy. Unlike answer-correction middlewares, it does not alter the
    request or inspect the model's output beyond connection failures.
    """

    def __init__(self, max_retries: int = 5, role: str = "") -> None:
        super().__init__()
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.max_retries = max_retries
        self.role = role

    def _retry(self, attempt: int, error: Exception) -> bool:
        retrying = is_retryable_model_error(error) and attempt < self.max_retries
        payload = {
            "role": self.role,
            "attempt": attempt + 1,
            "max_attempts": self.max_retries + 1,
            "error_type": type(error).__qualname__,
            "error": str(error)[:500],
            "retrying": retrying,
        }
        logger.warning("model_retry: %s", payload)
        return retrying

    @override
    def wrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        attempt = 0
        while True:
            try:
                return handler(request)
            except Exception as error:
                if not self._retry(attempt, error):
                    raise
                attempt += 1
                time.sleep(_backoff_delay(attempt))

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        attempt = 0
        while True:
            try:
                return await handler(request)
            except Exception as error:
                if not self._retry(attempt, error):
                    raise
                attempt += 1
                await asyncio.sleep(_backoff_delay(attempt))
