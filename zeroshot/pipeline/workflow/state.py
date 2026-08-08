from enum import Enum
from typing import Annotated, Literal, NotRequired, Self, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field, model_validator

from zeroshot.pipeline.tools import VerifyOutputResult


class SemanticHypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    semantics: list[str] = Field(
        ...,
        description="Notable semantic features characterizing the CAD model, such as Boss, Hole, Flange, etc.",
    )


class SemanticHypothesisReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: Literal["accept", "revise"]
    feedback: str

    @model_validator(mode="after")
    def require_feedback_for_revision(self) -> Self:
        if self.decision == "revise" and not self.feedback.strip():
            raise ValueError("feedback is required when decision is revise")
        return self


################################


class StopReason(Enum):
    COMPLETED = "COMPLETED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"


class ReconstructionState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    semantic_hypothesis: NotRequired[SemanticHypothesis]
    agent_turns: NotRequired[int]
    stop_reason: NotRequired[StopReason]
    last_verification: NotRequired[VerifyOutputResult]
