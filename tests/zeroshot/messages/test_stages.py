"""The single ordered vocabulary shared by contracts and workflow routing."""

import pytest

from zeroshot.pipeline.stages.types import (
    PIPELINE_STAGES,
    REASONING_STAGES,
    PipelineStage,
    next_stage,
)


def test_the_pipeline_order_has_one_canonical_definition() -> None:
    assert PIPELINE_STAGES == (
        PipelineStage.DRAWINGS,
        PipelineStage.SEMANTICS,
        PipelineStage.OPERATIONS,
        PipelineStage.CODING,
        PipelineStage.AUDIT,
    )
    assert REASONING_STAGES == PIPELINE_STAGES[:-1]


@pytest.mark.parametrize(
    ("completed", "following"),
    [
        (None, PipelineStage.DRAWINGS),
        (PipelineStage.DRAWINGS, PipelineStage.SEMANTICS),
        (PipelineStage.SEMANTICS, PipelineStage.OPERATIONS),
        (PipelineStage.OPERATIONS, PipelineStage.CODING),
        (PipelineStage.CODING, PipelineStage.AUDIT),
        (PipelineStage.AUDIT, None),
    ],
)
def test_next_stage_follows_that_order(
    completed: PipelineStage | None,
    following: PipelineStage | None,
) -> None:
    assert next_stage(completed) is following


def test_stage_values_remain_stable_json_strings() -> None:
    assert [stage.value for stage in PIPELINE_STAGES] == [
        "drawings",
        "semantics",
        "operations",
        "coding",
        "audit",
    ]
