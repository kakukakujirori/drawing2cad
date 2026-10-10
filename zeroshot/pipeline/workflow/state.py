from typing import (
    NotRequired,
    TypedDict,
    get_args,
    get_type_hints,
)

from typing_extensions import is_typeddict

from zeroshot.pipeline.stages.audit.contracts import AuditReport, AuditSubmission
from zeroshot.pipeline.stages.contracts import (
    ReconstructionHistory,
    ReconstructionSnapshot,
)
from zeroshot.pipeline.stages.tickets.contracts import TicketAnswers
from zeroshot.pipeline.stages.types import PipelineStage
from zeroshot.pipeline.workflow.components.agent import AgentState


class ReconstructionState(TypedDict):
    coding_state: NotRequired[AgentState]
    audit_state: NotRequired[AgentState]

    reconstruction: NotRequired[ReconstructionHistory]
    stage_submission: NotRequired[TicketAnswers | None]
    stage_validation_error: NotRequired[str | None]
    stage_validation_failure_count: NotRequired[int]
    audit_report: NotRequired[AuditReport | None]
    audit_evidence: NotRequired[dict[str, list[str]]]


def current_snapshot(state: ReconstructionState) -> ReconstructionSnapshot:
    reconstruction = state.get("reconstruction")
    if reconstruction is None:
        raise RuntimeError("reconstruction has not been initialized")
    return reconstruction.snapshots[-1]


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
# AuditSubmission lives under AgentState's untyped structured_response field.
CUSTOM_STATE_TYPES = _custom_state_types(
    ReconstructionState, PipelineStage, AuditSubmission
)
