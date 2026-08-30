from enum import StrEnum
from typing import Literal, cast


class PipelineStage(StrEnum):
    SEMANTICS = "semantics"
    OPERATIONS = "operations"
    CODING = "coding"
    AUDIT = "audit"


type ReasoningStage = Literal[
    PipelineStage.SEMANTICS,
    PipelineStage.OPERATIONS,
    PipelineStage.CODING,
]

PIPELINE_STAGES = tuple(PipelineStage)
REASONING_STAGES = cast(tuple[ReasoningStage, ...], PIPELINE_STAGES[:-1])


def next_stage(completed: PipelineStage | None) -> PipelineStage | None:
    """The fixed pipeline successor, with ``None`` denoting a new round."""
    if completed is None:
        return PIPELINE_STAGES[0]

    index = PIPELINE_STAGES.index(completed) + 1
    return PIPELINE_STAGES[index] if index < len(PIPELINE_STAGES) else None
