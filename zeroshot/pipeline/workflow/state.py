from collections.abc import Mapping, Sequence
from typing import (
    Any,
    NotRequired,
    TypedDict,
    cast,
    get_args,
    get_type_hints,
)

from langchain_core.messages import AnyMessage
from typing_extensions import is_typeddict

from zeroshot.pipeline.messages.contracts.audit import AuditReport
from zeroshot.pipeline.messages.contracts.reconstruction import (
    CodingSubmission,
    OperationSubmission,
    ReconstructionRun,
    SemanticSubmission,
)
from zeroshot.pipeline.messages.contracts.stages import (
    PipelineStage,
    ReasoningStage,
)
from zeroshot.pipeline.workflow.components.agent import AgentState


class ReconstructionState(TypedDict):
    semantics_state: NotRequired[AgentState]
    operations_state: NotRequired[AgentState]
    coding_state: NotRequired[AgentState]
    audit_state: NotRequired[AgentState]

    reconstruction: NotRequired[ReconstructionRun]
    # Keep the concrete models inline: checkpoint type discovery walks this
    # annotation and must see every runtime submission class.
    stage_submission: NotRequired[
        SemanticSubmission | OperationSubmission | CodingSubmission | None
    ]
    stage_validation_error: NotRequired[str | None]
    stage_validation_failure_count: NotRequired[int]
    audit_report: NotRequired[AuditReport | None]


# Where each reasoning stage keeps the transcript of the agent that carried
# the thread. The one place that knows which channel belongs to which stage.
_LEAD_TRANSCRIPT: Mapping[ReasoningStage, str] = {
    PipelineStage.SEMANTICS: "semantics_state",
    PipelineStage.OPERATIONS: "operations_state",
    PipelineStage.CODING: "coding_state",
}


def lead_transcript(
    state: ReconstructionState, stage: ReasoningStage
) -> list[AnyMessage]:
    """The transcript of the agent that carried the thread through `stage`."""
    stage_state = cast(Mapping[str, Any], state).get(_LEAD_TRANSCRIPT[stage]) or {}
    return list(stage_state.get("messages") or [])


def carry_thread(
    state: ReconstructionState, thread: Sequence[AnyMessage]
) -> dict[str, Any]:
    """Broadcast `thread` to other stages."""
    seeded = {"messages": list(thread), "reported_message_count": len(thread)}

    stage_states = cast(Mapping[str, Any], state)
    return {
        channel: {**(dict(stage_states.get(channel) or {})), **seeded}
        for channel in _LEAD_TRANSCRIPT.values()
    }


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


# ReasoningStage is a Literal of enum members, which generic annotation
# traversal cannot discover as a class by itself.
CUSTOM_STATE_TYPES = _custom_state_types(ReconstructionState, PipelineStage)
