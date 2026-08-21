"""Publish what an agent was asked, which no other node reports."""

from collections.abc import Awaitable, Callable
from typing import Any, NotRequired, override

from langchain.agents import AgentState as _AgentState
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langgraph.runtime import Runtime


class PromptLogState(_AgentState[Any]):
    """State fields owned and required by PromptLogMiddleware."""

    reported_message_count: NotRequired[int]


class PromptLogMiddleware(AgentMiddleware[PromptLogState, None, Any]):
    """Publish the prompt an agent is called with, once per invocation.

    Neither half of that prompt reaches the event log on its own. A system
    prompt never enters agent state, and an entry instruction is built by the
    workflow and handed to `invoke`, so no node reports it as something it
    produced. What an agent was asked is what an experiment varies, so it is
    worth a record beside the answers it produced.
    """

    state_schema = PromptLogState

    def __init__(self, role: str) -> None:
        super().__init__()
        self.role = role
        self._pending = False

    @override
    def before_agent(
        self, state: PromptLogState, runtime: Runtime[None]
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
        self, state: PromptLogState, runtime: Runtime[None]
    ) -> dict[str, Any] | None:
        """Mark where this ask ended, so the next one reports only its own.

        The mark rides in agent state rather than on this instance, so that an
        agent handed a transcript someone else built -- the fan-out reducer
        continuing its head proposer, or a stage continuing the thread of the
        stage before it -- reports its own ask instead of replaying an inherited
        one.  The caller states the inheritance by seeding the count.
        """
        del runtime
        return {"reported_message_count": len(state["messages"])}

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
        reported = request.state.get("reported_message_count", 0)
        request.runtime.stream_writer(
            {
                "prompt": {
                    "role": self.role,
                    "system": "" if reported else (request.system_prompt or ""),
                    "messages": list(request.messages[reported:]),
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
