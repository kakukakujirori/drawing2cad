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

        return {
            "messages": [
                HumanMessage(
                    content=f"[turn {turns + 1}/{self.max_turns}]",
                )
            ]
        }

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


class ModelCallRetryMiddleware(AgentMiddleware[_AgentState[Any], None, Any]):
    """Retry transient and incomplete model calls from one shared budget."""

    def __init__(self, max_retries: int) -> None:
        super().__init__()
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.max_retries = max_retries

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
            except LengthFinishReasonError:
                if attempt >= self.max_retries:
                    raise
                current_request = self._retry_length_limited_request(current_request)
            except StructuredOutputValidationError as error:
                if attempt >= self.max_retries:
                    raise
                current_request = self._retry_invalid_structured_output_request(
                    current_request,
                    error,
                )
            except Exception as error:
                if not _is_retryable_model_error(error) or attempt >= self.max_retries:
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
            except LengthFinishReasonError:
                if attempt >= self.max_retries:
                    raise
                current_request = self._retry_length_limited_request(current_request)
            except StructuredOutputValidationError as error:
                if attempt >= self.max_retries:
                    raise
                current_request = self._retry_invalid_structured_output_request(
                    current_request,
                    error,
                )
            except Exception as error:
                if not _is_retryable_model_error(error) or attempt >= self.max_retries:
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
    """
    if type(exception) is APIError:
        return True
    if isinstance(exception, APIConnectionError):
        return True
    if isinstance(exception, APIStatusError):
        return exception.status_code == 429 or exception.status_code >= 500
    return isinstance(exception, (httpx.NetworkError, httpx.ProtocolError))
