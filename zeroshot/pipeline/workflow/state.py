from enum import Enum
from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from zeroshot.pipeline.tools import VerifyOutputResult


class StopReason(Enum):
    COMPLETED = "COMPLETED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"


class ReconstructionState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    agent_turns: NotRequired[int]
    stop_reason: NotRequired[StopReason]
    last_verification: NotRequired[VerifyOutputResult]
