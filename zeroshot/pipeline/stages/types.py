from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Literal, cast


class PipelineStage(StrEnum):
    DRAWINGS = "drawings"
    SEMANTICS = "semantics"
    OPERATIONS = "operations"
    CODING = "coding"
    AUDIT = "audit"


type ReasoningStage = Literal[
    PipelineStage.DRAWINGS,
    PipelineStage.SEMANTICS,
    PipelineStage.OPERATIONS,
    PipelineStage.CODING,
]

PIPELINE_STAGES = tuple(PipelineStage)
REASONING_STAGES = cast(tuple[ReasoningStage, ...], PIPELINE_STAGES[:-1])


# Fields a stage may update, not fields that must be non-null at completion.
# In particular, a failed coding attempt may have no readable program source.
STAGE_ARTIFACT_FIELDS: Mapping[ReasoningStage, tuple[ArtifactField, ...]] = {
    PipelineStage.DRAWINGS: ("drawings",),
    PipelineStage.SEMANTICS: ("semantics",),
    PipelineStage.OPERATIONS: ("operations",),
    PipelineStage.CODING: ("program_source", "verification"),
}
type ArtifactField = Literal[
    "drawings", "semantics", "operations", "program_source", "verification"
]


def next_stage(completed: PipelineStage | None) -> PipelineStage | None:
    """The fixed pipeline successor, with ``None`` denoting a new round."""
    if completed is None:
        return PIPELINE_STAGES[0]

    index = PIPELINE_STAGES.index(completed) + 1
    return PIPELINE_STAGES[index] if index < len(PIPELINE_STAGES) else None
