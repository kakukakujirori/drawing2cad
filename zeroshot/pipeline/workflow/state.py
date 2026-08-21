import operator
from typing import (
    Annotated,
    Literal,
    NotRequired,
    Self,
    TypedDict,
    get_args,
    get_type_hints,
)

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import is_typeddict

from zeroshot.pipeline.tools import VerifyOutputResult
from zeroshot.pipeline.workflow.agent import AgentState
from zeroshot.pipeline.workflow.fanout_reduce import FanoutReduceState
from zeroshot.pipeline.workflow.fanout_reduce import Proposal as SemanticProposal
from zeroshot.pipeline.workflow.proposer_reviewer import Proposal as OperationProposal
from zeroshot.pipeline.workflow.proposer_reviewer import ProposerReviewerState


class Audit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revise: Literal["semantics", "operations", "coding"] | None = Field(
        ...,
        description=(
            "The earliest stage that must be redone. "
            "None if the reconstruction is accepted."
        ),
    )
    rationale: str = Field(
        ...,
        description=(
            "Why the selected stage needs revision, what is wrong and what to change."
            "It must be specific enough to act on without guessing."
            "If `revise=None`, state why the reconstruction is accepted."
        ),
    )

    @model_validator(mode="after")
    def require_rationale_for_redo(self) -> Self:
        if self.revise is not None and not self.rationale.strip():
            raise ValueError("rationale is required when a stage is revised")
        return self


class ReconstructionState(TypedDict):
    semantics_state: NotRequired[FanoutReduceState]
    operations_state: NotRequired[ProposerReviewerState]
    coding_state: NotRequired[AgentState]
    audit_state: NotRequired[AgentState]

    semantic_hypothesis: NotRequired[SemanticProposal | None]
    operation_plan: NotRequired[OperationProposal | None]
    last_verification: NotRequired[VerifyOutputResult]
    audit: NotRequired[Audit | None]
    audit_reject_count: NotRequired[Annotated[int, operator.add]]


def _custom_state_types(*root_schemas: type) -> tuple[type, ...]:
    """Collect project-defined runtime types reachable from state schemas."""
    collected: dict[tuple[str, str], type] = {}
    visited: set[tuple[str, str]] = set()

    def walk(annotation: object) -> None:
        # unwrap Annotated, NotRequired, Union, list[T], dict[K, V], etc.
        for argument in get_args(annotation):
            walk(argument)

        if not isinstance(annotation, type):
            return
        if not annotation.__module__.startswith("zeroshot."):
            return

        key = (annotation.__module__, annotation.__name__)
        if key in visited:
            return
        visited.add(key)

        # TypedDict is dict at runtime, so it is not necessary to allowlist.
        # However, we need to explore its field annotations.
        if not is_typeddict(annotation):
            collected[key] = annotation

        for field_annotation in get_type_hints(
            annotation, include_extras=True
        ).values():
            walk(field_annotation)

    for root_schema in root_schemas:
        walk(root_schema)

    return tuple(collected[key] for key in sorted(collected))


CUSTOM_STATE_TYPES = _custom_state_types(ReconstructionState)
