import operator
from collections.abc import Mapping, Sequence
from typing import (
    Annotated,
    Any,
    Literal,
    NotRequired,
    Self,
    TypedDict,
    cast,
    get_args,
    get_type_hints,
)

from langchain_core.messages import AnyMessage
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import is_typeddict

from zeroshot.pipeline.messages.contracts import (
    OperationPlan,
    PlanCoverage,
    SemanticHypothesis,
)
from zeroshot.pipeline.tools import VerifyOutputResult
from zeroshot.pipeline.workflow.components.agent import AgentState
from zeroshot.pipeline.workflow.components.fanout_reduce import FanoutReduceState
from zeroshot.pipeline.workflow.components.proposer_reviewer import (
    ProposerReviewerState,
)

type ReasoningStage = Literal["semantics", "operations", "coding"]


class Audit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revise: ReasoningStage | None = Field(
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

    semantic_hypothesis: NotRequired[SemanticHypothesis | None]
    operation_plan: NotRequired[OperationPlan | None]
    plan_coverage: NotRequired[PlanCoverage | None]
    last_verification: NotRequired[VerifyOutputResult]
    audit: NotRequired[Audit | None]
    audit_reject_count: NotRequired[Annotated[int, operator.add]]


# Which agent carries the thread through each reasoning stage, and where its
# transcript is kept: the fan-out continues its head proposer as the reducer,
# the proposer-reviewer loop continues its proposer, and a plain agent is its
# own.  The one place that knows the three shapes.
_LEAD_TRANSCRIPT: Mapping[ReasoningStage, tuple[str, str | None]] = {
    "semantics": ("semantics_state", "reducer_state"),
    "operations": ("operations_state", "proposer_state"),
    "coding": ("coding_state", None),
}


def lead_transcript(
    state: ReconstructionState, stage: ReasoningStage
) -> list[AnyMessage]:
    """The transcript of the agent that carried the thread through `stage`."""
    channel, nested = _LEAD_TRANSCRIPT[stage]
    stage_state = cast(Mapping[str, Any], state).get(channel) or {}
    lead = (stage_state.get(nested) or {}) if nested is not None else stage_state
    return list(lead.get("messages") or [])


def carry_thread(
    state: ReconstructionState, thread: Sequence[AnyMessage]
) -> dict[str, Any]:
    """Broadcast `thread` to other stages."""
    seeded = {"messages": list(thread), "reported_message_count": len(thread)}

    stage_states = cast(Mapping[str, Any], state)
    update: dict[str, Any] = {}
    for channel, nested in _LEAD_TRANSCRIPT.values():
        existing = dict(stage_states.get(channel) or {})
        update[channel] = (
            {**existing, **seeded}
            if nested is None
            else {**existing, nested: {**(existing.get(nested) or {}), **seeded}}
        )
    return update


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
