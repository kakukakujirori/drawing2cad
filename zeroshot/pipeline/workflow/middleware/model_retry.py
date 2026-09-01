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
    """Retry incomplete model calls, and transport failures beside them.

    The two are counted apart. An answer this middleware rejects -- truncated,
    unparseable, or empty -- is evidence about the model, and the retry that
    follows carries feedback the next attempt is meant to act on. A dropped
    stream or a gateway timeout is evidence about nothing: the model never
    answered, the request goes back unchanged, and spending one of the
    corrections on it would take away a chance the model never got to use.

    Every attempt it gives up on is reported. A retry is otherwise invisible:
    it leaves no turn, no message and no event, so the only trace is an extra
    HTTP request in a log that also shows requests a single attempt made. What
    an abandoned attempt does leave behind is real -- a model stream nobody
    will ever finish -- so a run has to be able to say when one happened.
    """

    def __init__(
        self,
        max_retries: int,
        role: str = "",
        max_transport_retries: int | None = None,
    ) -> None:
        super().__init__()
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if max_transport_retries is not None and max_transport_retries < 0:
            raise ValueError("max_transport_retries must be >= 0")
        self.max_retries = max_retries
        self.max_transport_retries = (
            max_retries if max_transport_retries is None else max_transport_retries
        )
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
        max_attempts: int | None = None,
    ) -> None:
        """Say which attempt failed, why, and what happens next."""
        details: dict[str, object] = {
            "role": self.role,
            "attempt": attempt + 1,
            "max_attempts": (self.max_retries if max_attempts is None else max_attempts)
            + 1,
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
        rejected = 0
        dropped = 0
        while True:
            try:
                response = handler(current_request)
            except LengthFinishReasonError as error:
                retrying = rejected < self.max_retries
                self._report(
                    current_request, rejected, error, retrying=retrying, adjusted=True
                )
                if not retrying:
                    raise
                rejected += 1
                current_request = self._retry_length_limited_request(current_request)
            except StructuredOutputError as error:
                retrying = rejected < self.max_retries
                self._report(
                    current_request, rejected, error, retrying=retrying, adjusted=True
                )
                if not retrying:
                    raise
                rejected += 1
                current_request = self._retry_structured_output_request(
                    current_request,
                    error,
                )
            except Exception as error:
                retrying = (
                    _is_retryable_model_error(error)
                    and dropped < self.max_transport_retries
                )
                self._report(
                    current_request,
                    dropped,
                    error,
                    retrying=retrying,
                    adjusted=False,
                    max_attempts=self.max_transport_retries,
                )
                if not retrying:
                    raise
                dropped += 1
                time.sleep(self._backoff_delay(dropped))
            else:
                if not _is_unanswered(response) or rejected >= self.max_retries:
                    return response
                self._report(
                    current_request,
                    rejected,
                    UnansweredModelCall(_UNANSWERED),
                    retrying=True,
                    adjusted=True,
                    response=response,
                )
                rejected += 1
                current_request = (
                    self._retry_length_limited_request(current_request)
                    if _ran_out_of_output(response)
                    else self._retry_unanswered_request(current_request)
                )

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        current_request = request
        rejected = 0
        dropped = 0
        while True:
            try:
                response = await handler(current_request)
            except LengthFinishReasonError as error:
                retrying = rejected < self.max_retries
                self._report(
                    current_request, rejected, error, retrying=retrying, adjusted=True
                )
                if not retrying:
                    raise
                rejected += 1
                current_request = self._retry_length_limited_request(current_request)
            except StructuredOutputError as error:
                retrying = rejected < self.max_retries
                self._report(
                    current_request, rejected, error, retrying=retrying, adjusted=True
                )
                if not retrying:
                    raise
                rejected += 1
                current_request = self._retry_structured_output_request(
                    current_request,
                    error,
                )
            except Exception as error:
                retrying = (
                    _is_retryable_model_error(error)
                    and dropped < self.max_transport_retries
                )
                self._report(
                    current_request,
                    dropped,
                    error,
                    retrying=retrying,
                    adjusted=False,
                    max_attempts=self.max_transport_retries,
                )
                if not retrying:
                    raise
                dropped += 1
                await asyncio.sleep(self._backoff_delay(dropped))
            else:
                if not _is_unanswered(response) or rejected >= self.max_retries:
                    return response
                self._report(
                    current_request,
                    rejected,
                    UnansweredModelCall(_UNANSWERED),
                    retrying=True,
                    adjusted=True,
                    response=response,
                )
                rejected += 1
                current_request = (
                    self._retry_length_limited_request(current_request)
                    if _ran_out_of_output(response)
                    else self._retry_unanswered_request(current_request)
                )


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


def _finish_reason(response: ModelResponse[Any]) -> str | None:
    last = response.result[-1] if response.result else None
    metadata = getattr(last, "response_metadata", None) or {}
    return metadata.get("finish_reason")


def _ran_out_of_output(response: ModelResponse[Any]) -> bool:
    """Whether the backend cut this answer off at the output limit.

    The OpenAI client raises `LengthFinishReasonError` for this, and the
    correction it earns -- drop the tools, keep the JSON short enough to close
    -- is written for it. A backend that reports the same thing in
    `finish_reason` instead would otherwise get the generic nudge, which asks
    the model to answer without telling it why the last one did not fit.
    """
    return _finish_reason(response) == "length"


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
        "finish_reason": _finish_reason(response),
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
