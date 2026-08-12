import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any, NotRequired, override

import httpx
from langchain.agents import AgentState as _AgentState
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    hook_config,
)
from langchain.agents.structured_output import StructuredOutputValidationError
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime
from openai import APIConnectionError, APIError, APIStatusError, LengthFinishReasonError


class StopReason(Enum):
    COMPLETED = "COMPLETED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"


class TurnBudgetState(_AgentState[Any]):
    """State fields owned and required by TurnBudgetMiddleware."""

    current_turn: NotRequired[int]
    total_turns: NotRequired[int]
    stop_reason: NotRequired[StopReason | None]


class TurnBudgetMiddleware(AgentMiddleware[TurnBudgetState, None, Any]):
    """Turn budget middleware.

    This middleware enforces a turn budget for the agent.
    """

    state_schema = TurnBudgetState

    def __init__(
        self,
        max_turns: int,
        announce_turns: bool = True,
        reset_turns_when_reentrant: bool = True,
    ) -> None:
        super().__init__()
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        self.max_turns = max_turns
        self.announce_turns = announce_turns
        self.reset_turns_when_reentrant = reset_turns_when_reentrant

    @override
    def before_agent(
        self, state: TurnBudgetState, runtime: Runtime[None]
    ) -> dict[str, Any] | None:
        """Reset progress that belongs to this invocation."""
        del runtime
        update: dict[str, Any] = {
            "structured_response": None,
            "stop_reason": None,
        }
        # `current_turn > 0` checks that this is 're'entrant
        if self.reset_turns_when_reentrant and state.get("current_turn", 0) > 0:
            update |= {
                "messages": [
                    HumanMessage(content=f"[turn reset to 0/{self.max_turns}]")
                ],
                "current_turn": 0,
            }
        return update

    @hook_config(can_jump_to=["end"])
    @override
    def before_model(
        self, state: TurnBudgetState, runtime: Runtime[None]
    ) -> dict[str, Any] | None:
        """Check turns, and add turn count to messages."""
        del runtime

        turns = state.get("current_turn", 0)

        if turns >= self.max_turns:
            return {
                "jump_to": "end",
                "stop_reason": StopReason.BUDGET_EXHAUSTED,
            }

        if not self.announce_turns:
            return None

        return {"messages": [HumanMessage(content=self._announcement(turns + 1))]}

    def _announcement(self, turn: int) -> str:
        """Announce the turn, and near the end say what running out costs.

        A budget that only ever reports a number leaves the agent to find the
        edge by going over it: the tool call it spends its last turn on is
        dropped along with the work that call was meant to settle, and a stage
        that never reaches an answer ends the run holding nothing. The warning
        arrives one turn early because the turn that hears it is the last one
        that can still act on it.
        """
        countdown = f"[turn {turn}/{self.max_turns}]"
        if turn >= self.max_turns:
            return (
                f"{countdown} Final turn: a tool call made now is dropped, so "
                "nothing you have not already run or written can still take "
                "effect. Answer from what you have, and say in the answer what "
                "remains uncertain."
            )
        if turn == self.max_turns - 1:
            return (
                f"{countdown} One turn remains after this one, and a tool call "
                "made on it is dropped. Run or write now whatever your answer "
                "still depends on."
            )
        return countdown

    @hook_config(can_jump_to=["end"])
    @override
    def after_model(
        self, state: TurnBudgetState, runtime: Runtime[None]
    ) -> dict[str, Any]:
        """Increment turns."""
        del runtime

        last_message = state["messages"][-1]
        tool_calls = (
            last_message.tool_calls if isinstance(last_message, AIMessage) else []
        )

        progress = {
            "current_turn": state.get("current_turn", 0) + 1,
            "total_turns": state.get("total_turns", 0) + 1,
        }
        if not tool_calls:
            return progress | {"stop_reason": StopReason.COMPLETED}
        if progress["current_turn"] >= self.max_turns:
            return progress | {
                "jump_to": "end",
                "stop_reason": StopReason.BUDGET_EXHAUSTED,
            }
        return progress


class PromptLogMiddleware(AgentMiddleware[_AgentState[Any], None, Any]):
    """Publish the prompt an agent is called with, once per invocation.

    Neither half of that prompt reaches the event log on its own. A system
    prompt never enters agent state, and an entry instruction is built by the
    workflow and handed to `invoke`, so no node reports it as something it
    produced. What an agent was asked is what an experiment varies, so it is
    worth a record beside the answers it produced.
    """

    def __init__(self, role: str) -> None:
        super().__init__()
        self.role = role
        self._reported = 0
        self._pending = False

    @override
    def before_agent(
        self, state: _AgentState[Any], runtime: Runtime[None]
    ) -> dict[str, Any] | None:
        """Arm the report for this ask.

        An invocation, not a turn: which turn an ask starts on depends on the
        stage's reset policy, and a role that carries its budget across asks
        would otherwise report only the first one it ever received.
        """
        del state, runtime
        self._pending = True
        return None

    @override
    def after_agent(
        self, state: _AgentState[Any], runtime: Runtime[None]
    ) -> dict[str, Any] | None:
        """Mark where this ask ended, so the next one reports only its own."""
        del runtime
        self._reported = len(state["messages"])
        return None

    def _publish(self, request: ModelRequest[None]) -> None:
        """Report what this ask added, once, on its first model call.

        A re-entered agent keeps the transcript it built, and repeating it would
        bury the instruction that is the reason for the ask.  What is new is
        everything after where the last ask ended; the system prompt does not
        change at all, so it is stated once.  The writer is the runtime's, a
        no-op when nothing is streaming, so a plain `invoke` stays silent.
        """
        if not self._pending:
            return
        self._pending = False
        request.runtime.stream_writer(
            {
                "prompt": {
                    "role": self.role,
                    "system": "" if self._reported else (request.system_prompt or ""),
                    "messages": list(request.messages[self._reported :]),
                }
            }
        )

    @override
    def wrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        self._publish(request)
        return handler(request)

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        self._publish(request)
        return await handler(request)


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
        request.runtime.stream_writer(
            {
                "model_retry": {
                    "role": self.role,
                    "attempt": attempt + 1,
                    "max_attempts": self.max_retries + 1,
                    "error_type": type(error).__qualname__,
                    "error": str(error)[:500],
                    "retrying": retrying,
                    "request_adjusted": adjusted,
                }
            }
        )

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
