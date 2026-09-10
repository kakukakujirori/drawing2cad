from collections.abc import Callable

from zeroshot.pipeline.stages.audit.stage import create_audit_stage
from zeroshot.pipeline.stages.coding.stage import create_coding_stage
from zeroshot.pipeline.stages.drawings.stage import create_drawing_stage
from zeroshot.pipeline.stages.operations.stage import create_operation_stage
from zeroshot.pipeline.stages.semantics.stage import create_semantic_stage
from zeroshot.pipeline.stages.types import PipelineStage


def stage_factory(stage: PipelineStage) -> Callable:
    match stage:
        case PipelineStage.SEMANTICS:
            return create_semantic_stage
        case PipelineStage.OPERATIONS:
            return create_operation_stage
        case PipelineStage.CODING:
            return create_coding_stage
        case PipelineStage.DRAWINGS:
            return create_drawing_stage
        case PipelineStage.AUDIT:
            return create_audit_stage
        case _:
            raise ValueError(f"unsupported stage: {stage}")
