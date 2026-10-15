"""Turn limit middleware for native generator pipeline."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, override

from langchain.agents import AgentState as _AgentState
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse

logger = logging.getLogger(__name__)


class TurnLimitExceededError(RuntimeError):
    """Raised when the agent exceeds the allowed turns."""


class TurnLimitMiddleware(AgentMiddleware[_AgentState[Any], None, Any]):
    """Enforce a maximum turn limit on the agent generator."""

    def __init__(self, workspace: Path, max_turns: int = 20) -> None:
        super().__init__()
        self.workspace = workspace
        self.max_turns = max_turns
        self.turn_count = 0
        self.last_messages: list[Any] = []

    def _check(self, request: ModelRequest[None]) -> None:
        self.turn_count += 1
        self.last_messages = list(request.messages)
        if self.turn_count > self.max_turns:
            model_file = self.workspace / "model.py"
            has_model = (
                model_file.is_file()
                and not model_file.is_symlink()
                and model_file.stat().st_size > 0
            )
            msg = (
                f"Turn limit ({self.max_turns}) exceeded; "
                f"model.py {'exists' if has_model else 'was not written'}."
            )
            logger.warning("turn_limit_exceeded: %s", msg)
            raise TurnLimitExceededError(msg)

    @override
    def wrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        self._check(request)
        return handler(request)

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        self._check(request)
        return await handler(request)
