from enum import Enum
from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from zeroshot.pipeline.tools import VerifyOutputResult


class StopReason(Enum):
    COMPLETED = "COMPLETED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"


class AgentState(TypedDict):
    """What an agent is given, and what it produces.

    `task` and `messages` are separate because the system prompt is the agent's
    own property, not the caller's: the agent opens its transcript with it and
    then with the task.  A caller that pre-filled `messages` would push the
    system prompt behind its own turns, since `add_messages` appends.
    """

    task: list[BaseMessage]
    messages: Annotated[list[BaseMessage], add_messages]
    turns: NotRequired[int]
    stop_reason: NotRequired[StopReason]


class ReconstructionState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    agent_turns: NotRequired[int]
    stop_reason: NotRequired[StopReason]
    last_verification: NotRequired[VerifyOutputResult]
