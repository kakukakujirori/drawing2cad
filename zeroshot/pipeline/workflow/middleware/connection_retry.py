"""Retry connection failures and transient API errors, including HTTP 429/5xx."""

import asyncio
import logging
import random
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
from langchain.agents.structured_output import StructuredOutputError
from langgraph.runtime import get_runtime
from openai import APIConnectionError, APIError, APIStatusError, LengthFinishReasonError
from openrouter.errors import OpenRouterError


@dataclass
class ModelConnectionRetry:
    """One transport budget, retained across corrections to a model's answer."""

    max_retries: int
    role: str
    stream_writer: Callable[[Any], None] | None = None
    _retries: int = field(default=0, init=False)

    def _retry(self, error: Exception) -> bool:
        # Answer failures belong to the caller's correction loop and budget.
        if isinstance(error, (LengthFinishReasonError, StructuredOutputError)):
            return False
        retrying = is_retryable_model_error(error) and self._retries < self.max_retries
        report_model_retry(
            error,
            role=self.role,
            attempt=self._retries,
            max_retries=self.max_retries,
            retrying=retrying,
            adjusted=False,
            stream_writer=self.stream_writer,
        )
        if retrying:
            self._retries += 1
        return retrying

    def invoke[T](self, call: Callable[[], T]) -> T:
        while True:
            try:
                return call()
            except Exception as error:
                if not self._retry(error):
                    raise
                time.sleep(_backoff_delay(self._retries))

    async def ainvoke[T](self, call: Callable[[], Awaitable[T]]) -> T:
        while True:
            try:
                return await call()
            except Exception as error:
                if not self._retry(error):
                    raise
                await asyncio.sleep(_backoff_delay(self._retries))


def _backoff_delay(retry_number: int) -> float:
    """Exponential backoff with ±25% jitter, starting with retry number one."""
    delay = min(2.0**retry_number, 60.0)
    jitter = delay * 0.25
    return max(0.0, delay + random.uniform(-jitter, jitter))


def report_model_retry(
    error: Exception,
    *,
    role: str,
    attempt: int,
    max_retries: int,
    retrying: bool,
    adjusted: bool,
    details: dict[str, object] | None = None,
    stream_writer: Callable[[Any], None] | None = None,
) -> None:
    """Report a failed attempt (zero-based), including the final failure."""
    payload: dict[str, object] = {
        "role": role,
        "attempt": attempt + 1,
        "max_attempts": max_retries + 1,
        "error_type": type(error).__qualname__,
        "error": str(error)[:500],
        "retrying": retrying,
        "request_adjusted": adjusted,
    }
    status = (
        error.status_code
        if isinstance(error, (APIStatusError, OpenRouterError))
        else _openrouter_stream_status(error)
    )
    if status is not None:
        payload["status_code"] = status
    payload.update(details or {})
    if stream_writer is None:
        try:
            runtime = get_runtime()
        except RuntimeError:  # Direct calls outside a runnable have no runtime.
            runtime = None
        if runtime is not None:
            stream_writer = runtime.stream_writer
    if stream_writer is not None:
        stream_writer({"model_retry": payload})
    else:
        logging.getLogger(__name__).warning("model_retry: %s", payload)


_OPENROUTER_STREAM_ERROR = re.compile(
    r"^OpenRouter API returned an error during streaming:.*\(code: (\d+)\)",
    re.DOTALL,
)


def _openrouter_stream_status(exception: Exception) -> int | None:
    """Return the HTTP status behind a streaming failure, or None if not one.

    The library puts the status only in the message text, so a reworded message
    silently stops matching — costing a retry, not correctness.
    """
    if type(exception) is not ValueError:
        return None
    match = _OPENROUTER_STREAM_ERROR.match(str(exception))
    return int(match[1]) if match else None


def is_retryable_model_error(exception: Exception) -> bool:
    """Whether to re-issue this call. SDK retries are off, so this is the policy.

    ``APIError`` is matched by exact class, not isinstance: Codex reports an
    overloaded stream as one after HTTP 200. ``TimeoutException`` is a sibling
    of ``NetworkError``, not a subclass, so it has to be named.
    """
    if (status := _openrouter_stream_status(exception)) is not None:
        return status == 429 or status >= 500
    if type(exception) is APIError:
        return True
    if isinstance(exception, APIConnectionError):
        return True
    if isinstance(exception, (APIStatusError, OpenRouterError)):
        return exception.status_code == 429 or exception.status_code >= 500
    return isinstance(
        exception,
        (httpx.NetworkError, httpx.ProtocolError, httpx.TimeoutException),
    )
