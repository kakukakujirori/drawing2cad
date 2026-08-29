"""Re-issue a model call the client could not retry itself, and say so."""

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from typing import Any, override

import httpx
from langchain.agents import AgentState as _AgentState
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.structured_output import StructuredOutputValidationError
from langchain_core.messages import HumanMessage
from openai import APIConnectionError, APIError, APIStatusError, LengthFinishReasonError


class ModelCallRetryMiddleware(AgentMiddleware[_AgentState[Any], None, Any]):
    """Retry transient and incomplete model calls from one shared budget.

    Every attempt it gives up on is reported. A retry is otherwise invisible:
    it leaves no turn, no message and no event, so the only trace is an extra
    HTTP request in a log that also shows requests a single attempt made. What
    an abandoned attempt does leave behind is real -- a model stream nobody
    will ever finish -- so a run has to be able to say when one happened.
    """

    def __init__(self, max_retries: int, role: str = "") -> None:
        super().__init__()
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.max_retries = max_retries
        self.role = role

    def _report(
        self,
        request: ModelRequest[None],
        attempt: int,
        error: Exception,
        *,
        retrying: bool,
        adjusted: bool,
    ) -> None:
        """Say which attempt failed, why, and what happens next."""
        details: dict[str, object] = {
            "role": self.role,
            "attempt": attempt + 1,
            "max_attempts": self.max_retries + 1,
            "error_type": type(error).__qualname__,
            "error": str(error)[:500],
            "retrying": retrying,
            "request_adjusted": adjusted,
        }
        if isinstance(error, StructuredOutputValidationError):
            # The retry request carries this response only in memory. Preserve
            # the rejected raw output so a contract failure is reproducible.
            details["failed_response"] = error.ai_message.text
        request.runtime.stream_writer({"model_retry": details})

    @staticmethod
    def _retry_length_limited_request(
        request: ModelRequest[None],
    ) -> ModelRequest[None]:
        return request.override(
            messages=[
                *request.messages,
                HumanMessage(
                    content=(
                        "Your response reached the output-token limit and could not "
                        "be parsed. Do not call tools. Return only concise raw JSON "
                        "that matches the required schema, with no explanation, "
                        "analysis, or Markdown outside it. Keep every string value "
                        "short enough to complete the entire JSON object."
                    )
                ),
            ]
        )

    @staticmethod
    def _retry_invalid_structured_output_request(
        request: ModelRequest[None],
        error: StructuredOutputValidationError,
    ) -> ModelRequest[None]:
        failed_response = [error.ai_message] if error.ai_message.text.strip() else []
        return request.override(
            messages=[
                *request.messages,
                *failed_response,
                HumanMessage(
                    content=(
                        f"Your previous response could not be parsed as the required "
                        f"{error.tool_name} structured output. Validation error: "
                        f"{error.source} Do not call tools. Return only corrected "
                        "raw JSON that matches the required schema, with no "
                        "explanation or Markdown outside it."
                    )
                ),
            ]
        )

    @staticmethod
    def _backoff_delay(retry_number: int) -> float:
        """Match LangChain's default exponential backoff with ±25% jitter."""
        delay = min(2.0**retry_number, 60.0)
        jitter = delay * 0.25
        return max(0.0, delay + random.uniform(-jitter, jitter))

    @override
    def wrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        current_request = request
        for attempt in range(self.max_retries + 1):
            try:
                return handler(current_request)
            except LengthFinishReasonError as error:
                retrying = attempt < self.max_retries
                self._report(
                    current_request, attempt, error, retrying=retrying, adjusted=True
                )
                if not retrying:
                    raise
                current_request = self._retry_length_limited_request(current_request)
            except StructuredOutputValidationError as error:
                retrying = attempt < self.max_retries
                self._report(
                    current_request, attempt, error, retrying=retrying, adjusted=True
                )
                if not retrying:
                    raise
                current_request = self._retry_invalid_structured_output_request(
                    current_request,
                    error,
                )
            except Exception as error:
                retrying = (
                    _is_retryable_model_error(error) and attempt < self.max_retries
                )
                self._report(
                    current_request, attempt, error, retrying=retrying, adjusted=False
                )
                if not retrying:
                    raise
                time.sleep(self._backoff_delay(attempt))
        raise AssertionError("unreachable")

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        current_request = request
        for attempt in range(self.max_retries + 1):
            try:
                return await handler(current_request)
            except LengthFinishReasonError as error:
                retrying = attempt < self.max_retries
                self._report(
                    current_request, attempt, error, retrying=retrying, adjusted=True
                )
                if not retrying:
                    raise
                current_request = self._retry_length_limited_request(current_request)
            except StructuredOutputValidationError as error:
                retrying = attempt < self.max_retries
                self._report(
                    current_request, attempt, error, retrying=retrying, adjusted=True
                )
                if not retrying:
                    raise
                current_request = self._retry_invalid_structured_output_request(
                    current_request,
                    error,
                )
            except Exception as error:
                retrying = (
                    _is_retryable_model_error(error) and attempt < self.max_retries
                )
                self._report(
                    current_request, attempt, error, retrying=retrying, adjusted=False
                )
                if not retrying:
                    raise
                await asyncio.sleep(self._backoff_delay(attempt))
        raise AssertionError("unreachable")


def _is_retryable_model_error(exception: Exception) -> bool:
    """Retry transient failures in the agent layer only.

    Model clients are configured with SDK retries disabled so this predicate is
    the single retry policy. Codex can report an overloaded stream as a plain
    ``APIError`` after HTTP 200; match that exact class without accidentally
    admitting every ``APIStatusError`` subclass. Status failures are eligible
    only when they are rate limits or server errors.

    A backend that accepts the request and then stalls surfaces as a bare
    ``httpx.TimeoutException``: the SDK wraps a timeout as ``APITimeoutError``
    only while it owns the request, and a stream that dies mid-body is past
    that point. ``TimeoutException`` is a sibling of ``NetworkError``, not a
    subclass, so it has to be named here.
    """
    if type(exception) is APIError:
        return True
    if isinstance(exception, APIConnectionError):
        return True
    if isinstance(exception, APIStatusError):
        return exception.status_code == 429 or exception.status_code >= 500
    return isinstance(
        exception,
        (httpx.NetworkError, httpx.ProtocolError, httpx.TimeoutException),
    )
