"""Re-issue a model call the client could not retry itself, and say so."""

import asyncio
import random
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any, override

import httpx
from langchain.agents import AgentState as _AgentState
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.structured_output import (
    MultipleStructuredOutputsError,
    StructuredOutputError,
    StructuredOutputValidationError,
)
from langchain_core.messages import HumanMessage
from openai import APIConnectionError, APIError, APIStatusError, LengthFinishReasonError


class UnansweredModelCall(Exception):
    """A model call that came back with no tool call, no text and no answer."""


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
        response: ModelResponse[Any] | None = None,
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
        if isinstance(error, StructuredOutputError):
            # The retry request carries this response only in memory. Preserve
            # the rejected raw output so a contract failure is reproducible.
            details["failed_response"] = error.ai_message.text
        if isinstance(error, UnansweredModelCall) and response is not None:
            details.update(_unanswered_diagnosis(response))
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
    def _retry_structured_output_request(
        request: ModelRequest[None],
        error: StructuredOutputError,
    ) -> ModelRequest[None]:
        # Replaying an AI message whose tool calls have no results makes the
        # next request invalid, so carry back only a plain-text answer.
        rejected = error.ai_message
        replay = [rejected] if rejected.text.strip() and not rejected.tool_calls else []
        return request.override(
            messages=[
                *request.messages,
                *replay,
                HumanMessage(content=_correction_text(error)),
            ]
        )

    @staticmethod
    def _retry_unanswered_request(
        request: ModelRequest[None],
    ) -> ModelRequest[None]:
        return request.override(
            messages=[
                *request.messages,
                HumanMessage(
                    content=(
                        "Your last turn spent its whole output budget on thinking "
                        "and came back empty. You have been thinking a long time, "
                        "so answer now. The analysis you have already done is "
                        "above; build on it rather than starting over."
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
                response = handler(current_request)
            except LengthFinishReasonError as error:
                retrying = attempt < self.max_retries
                self._report(
                    current_request, attempt, error, retrying=retrying, adjusted=True
                )
                if not retrying:
                    raise
                current_request = self._retry_length_limited_request(current_request)
            except StructuredOutputError as error:
                retrying = attempt < self.max_retries
                self._report(
                    current_request, attempt, error, retrying=retrying, adjusted=True
                )
                if not retrying:
                    raise
                current_request = self._retry_structured_output_request(
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
            else:
                if not _is_unanswered(response) or attempt == self.max_retries:
                    return response
                self._report(
                    current_request,
                    attempt,
                    UnansweredModelCall(_UNANSWERED),
                    retrying=True,
                    adjusted=True,
                    response=response,
                )
                current_request = self._retry_unanswered_request(current_request)
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
                response = await handler(current_request)
            except LengthFinishReasonError as error:
                retrying = attempt < self.max_retries
                self._report(
                    current_request, attempt, error, retrying=retrying, adjusted=True
                )
                if not retrying:
                    raise
                current_request = self._retry_length_limited_request(current_request)
            except StructuredOutputError as error:
                retrying = attempt < self.max_retries
                self._report(
                    current_request, attempt, error, retrying=retrying, adjusted=True
                )
                if not retrying:
                    raise
                current_request = self._retry_structured_output_request(
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
            else:
                if not _is_unanswered(response) or attempt == self.max_retries:
                    return response
                self._report(
                    current_request,
                    attempt,
                    UnansweredModelCall(_UNANSWERED),
                    retrying=True,
                    adjusted=True,
                    response=response,
                )
                current_request = self._retry_unanswered_request(current_request)
        raise AssertionError("unreachable")


_UNANSWERED = "the model returned no tool call, no text and no structured output"


def _is_unanswered(response: ModelResponse[Any]) -> bool:
    """Whether the response holds nothing to act on.

    Reasoning alone leaves this true, and the agent loop reads that as a
    finished answer.
    """
    if response.structured_response is not None:
        return False
    return not any(
        getattr(message, "tool_calls", None) or message.text.strip()
        for message in response.result
    )


def _unanswered_diagnosis(response: ModelResponse[Any]) -> dict[str, object]:
    """Say what the backend reported about an answer that carried nothing.

    An empty completion is the same event whether the model ran out of output
    budget mid-thought, was cut off upstream, or simply stopped -- and the
    three want different fixes. The backend distinguishes them and this is the
    only place that sees its report.
    """
    last = response.result[-1] if response.result else None
    if last is None:
        return {}
    metadata = getattr(last, "response_metadata", None) or {}
    usage = getattr(last, "usage_metadata", None) or {}
    return {
        "finish_reason": metadata.get("finish_reason"),
        "provider": metadata.get("provider"),
        "output_tokens": usage.get("output_tokens"),
        "reasoning_tokens": (usage.get("output_token_details") or {}).get("reasoning"),
    }


def _correction_text(error: StructuredOutputError) -> str:
    """Tell the model what was wrong with its answer, in terms it can act on."""
    if isinstance(error, MultipleStructuredOutputsError):
        return (
            f"You returned {len(error.tool_names)} structured responses "
            f"({', '.join(error.tool_names)}) when exactly one is expected. "
            "Return one call that carries the whole answer."
        )
    if isinstance(error, StructuredOutputValidationError):
        return (
            f"Your previous response could not be parsed as the required "
            f"{error.tool_name} structured output. Validation error: "
            f"{error.source}. You may continue using tools if you need more "
            "information. When you are ready to answer, return corrected raw "
            "JSON that matches the required schema, with no explanation or "
            "Markdown outside it."
        )
    return f"Your previous response was rejected: {error}. Answer again."


_OPENROUTER_STREAM_ERROR = re.compile(
    r"^OpenRouter API returned an error during streaming:.*\(code: (\d+)\)$",
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


def _is_retryable_model_error(exception: Exception) -> bool:
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
    if isinstance(exception, APIStatusError):
        return exception.status_code == 429 or exception.status_code >= 500
    return isinstance(
        exception,
        (httpx.NetworkError, httpx.ProtocolError, httpx.TimeoutException),
    )
