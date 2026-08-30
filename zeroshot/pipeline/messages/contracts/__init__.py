"""What each reasoning stage must answer with, and how it is shown downstream.

One module per stage's answer, split along the seam the pipeline already has.
`operations` reads `semantics` because a plan is made from a hypothesis and
says which of its features each step builds; nothing reads back the other way.
The names are re-exported here so a caller need not know which stage a contract
belongs to.
"""

from zeroshot.pipeline.messages.contracts.operations import (
    Operation,
    OperationPlan,
    OperationVerb,
    linearise,
    operation_heading,
    render_plan,
    resolve_reference,
)
from zeroshot.pipeline.messages.contracts.semantics import (
    VIEW_FRAME,
    Axis,
    ClaimSource,
    DrawnEntity,
    EdgeStyle,
    FeatureGeometry,
    GeometryKind,
    Parameter,
    ParameterName,
    SemanticFeature,
    SemanticHypothesis,
    View,
    ViewEvidence,
    edge_style_for_linetype,
    render_hypothesis,
    view_frame_sentence,
)
from zeroshot.pipeline.messages.contracts.stages import (
    PIPELINE_STAGES,
    REASONING_STAGES,
    PipelineStage,
    ReasoningStage,
    next_stage,
)

__all__ = [
    "PIPELINE_STAGES",
    "REASONING_STAGES",
    "VIEW_FRAME",
    "Axis",
    "ClaimSource",
    "DrawnEntity",
    "EdgeStyle",
    "FeatureGeometry",
    "GeometryKind",
    "Operation",
    "OperationPlan",
    "OperationVerb",
    "Parameter",
    "ParameterName",
    "PipelineStage",
    "ReasoningStage",
    "SemanticFeature",
    "SemanticHypothesis",
    "View",
    "ViewEvidence",
    "edge_style_for_linetype",
    "linearise",
    "next_stage",
    "operation_heading",
    "render_hypothesis",
    "render_plan",
    "resolve_reference",
    "view_frame_sentence",
]
