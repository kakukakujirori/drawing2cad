from collections.abc import Callable

from zeroshot.pipeline.stages.audit.stage import create_audit_stage
from zeroshot.pipeline.stages.coding.stage import create_coding_stage
from zeroshot.pipeline.stages.interpretation.stage import create_interpretation_stage
from zeroshot.pipeline.stages.operations.stage import create_operation_stage
from zeroshot.pipeline.stages.types import PipelineStage


def stage_factory(stage: PipelineStage) -> Callable:
    match stage:
        case PipelineStage.INTERPRETATION:
            return create_interpretation_stage
        case PipelineStage.OPERATIONS:
            return create_operation_stage
        case PipelineStage.CODING:
            return create_coding_stage
        case PipelineStage.AUDIT:
            return create_audit_stage
        case _:
            raise ValueError(f"unsupported stage: {stage}")
