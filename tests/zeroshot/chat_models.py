"""A model that answers from a script, and records what it was asked.

Shared because an agent test and a workflow test need the same thing: a real
`BaseChatModel`, since the agent loop binds tools to it and reads its profile,
and a fixed sequence of replies, since the point is what the graph does with
them.
"""

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import PrivateAttr


class ScriptedChatModel(BaseChatModel):
    responses: tuple[AIMessage, ...]

    _response_index: int = PrivateAttr(default=0)
    _received_messages: list[list[BaseMessage]] = PrivateAttr(default_factory=list)
    _bound_tool_names: tuple[str, ...] = PrivateAttr(default=())
    _bound_tool_name_history: list[tuple[str, ...]] = PrivateAttr(
        default_factory=list
    )

    @property
    def _llm_type(self) -> str:
        return "scripted-test-model"

    @property
    def received_messages(self) -> list[list[BaseMessage]]:
        return self._received_messages

    @property
    def bound_tool_names(self) -> tuple[str, ...]:
        return self._bound_tool_names

    @property
    def bound_tool_name_history(self) -> list[tuple[str, ...]]:
        """What was bound on each call, so a per-turn change is visible."""
        return self._bound_tool_name_history

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        del tool_choice, kwargs

        if not all(isinstance(tool, BaseTool) for tool in tools):
            raise TypeError("This scripted model only accepts BaseTool instances")

        self._bound_tool_names = tuple(
            tool.name for tool in tools if isinstance(tool, BaseTool)
        )
        self._bound_tool_name_history.append(self._bound_tool_names)
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs

        self._received_messages.append(list(messages))
        try:
            response = self.responses[self._response_index]
        except IndexError as error:
            raise AssertionError("Scripted model ran out of responses") from error

        self._response_index += 1
        return ChatResult(
            generations=[ChatGeneration(message=response)],
        )


def tool_call(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    """One turn that asks for one tool."""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def unanswered_tool_calls(messages: Sequence[BaseMessage]) -> list[str | None]:
    """Calls in a transcript that no result answers.

    A scripted model accepts any history; OpenAI rejects one that holds a
    function call without its output.
    """
    answered = {
        message.tool_call_id for message in messages if isinstance(message, ToolMessage)
    }
    return [
        call["id"]
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
        if call["id"] not in answered
    ]
