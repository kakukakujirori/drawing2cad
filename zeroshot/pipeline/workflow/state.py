from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from zeroshot.pipeline.tools import VerifyOutputResult


class ReconstructionState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    last_verification: NotRequired[VerifyOutputResult]
