import operator
from collections.abc import Mapping
from enum import Enum
from typing import (
    Annotated,
    Literal,
    NotRequired,
    Self,
    TypedDict,
    get_args,
    get_type_hints,
)

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


class OperationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operations: list[str] = Field(
        ...,
        description="Modelling steps in build order, one per entry, each naming the feature it produces and the dimensions it uses.",
    )


class Review(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: Literal["accept", "revise"] = Field(
        ...,
        description="'accept' if the proposal can be used as it stands, 'revise' if it must be corrected first.",
    )
    feedback: str = Field(
        ...,
        description="Why that decision. For 'revise', name what is wrong and what it should be instead, specific enough to act on without guessing; it must not be empty. For 'accept', summarize what makes the proposal correct.",
    )

    @model_validator(mode="after")
    def require_feedback_for_revision(self) -> Self:
        if self.decision == "revise" and not self.feedback.strip():
            raise ValueError("feedback is required when decision is revise")
        return self


class Audit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: Literal["accept", "redo_code", "redo_operations", "redo_semantics"] = (
        Field(
            ...,
            description="'accept' if the built model matches the drawing. Otherwise the stage to go back to: 'redo_code' when the plan is right but the program does not build it, 'redo_operations' when the features are right but the plan cannot produce them, 'redo_semantics' when the part was misread.",
        )
    )
    feedback: str = Field(
        ...,
        description="Why that decision. For any 'redo_', state the mismatch against the drawing and what the named stage has to change, specific enough to act on without guessing; it must not be empty. For 'accept', summarize what makes the model correct.",
    )

    @model_validator(mode="after")
    def require_feedback_for_redo(self) -> Self:
        if self.decision != "accept" and not self.feedback.strip():
            raise ValueError("feedback is required when a stage is sent back")
        return self


################################


class StopReason(Enum):
    COMPLETED = "COMPLETED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"


def add_turns(
    current: Mapping[str, int], incoming: Mapping[str, int]
) -> dict[str, int]:
    """Turns spent per role, accumulated.

    Added rather than replaced, unlike `stop_reasons`: a role the workflow sent
    back has spent both its rounds, while how it finished is only ever how it
    finished last.
    """
    return {
        role: current.get(role, 0) + incoming.get(role, 0)
        for role in {*current, *incoming}
    }


class ReconstructionState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    semantic_hypothesis: NotRequired[SemanticHypothesis]
    operation_plan: NotRequired[OperationPlan]
    agent_turns: NotRequired[Annotated[dict[str, int], add_turns]]
    stop_reasons: NotRequired[Annotated[dict[str, StopReason], operator.or_]]
    last_verification: NotRequired[VerifyOutputResult]
    audit: NotRequired[Audit]
    audit_reject_count: NotRequired[Annotated[int, operator.add]]


def _custom_state_types() -> tuple[type, ...]:
    """Collect the classes this project defines that ReconstructionState holds.

    A checkpointer has to be told about each of them to restore a field as its
    own type rather than as a plain dict.
    """
    by_name: dict[str, type] = {}

    def walk(annotation: object) -> None:
        for argument in get_args(annotation):
            walk(argument)
        if isinstance(annotation, type) and annotation.__module__.startswith(
            "zeroshot."
        ):
            by_name[f"{annotation.__module__}.{annotation.__qualname__}"] = annotation

    for annotation in get_type_hints(ReconstructionState).values():
        walk(annotation)
    return tuple(by_name.values())


CUSTOM_STATE_TYPES = _custom_state_types()
