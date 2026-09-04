"""What each reasoning stage must answer with, and how it is shown downstream.

One module per stage's answer, split along the seam the pipeline already has.
`operations` reads `semantics` because a plan is made from a hypothesis and
says which of its features each step builds; `semantics` reads `drawings`
because a claim is made from evidence; nothing reads back the other way.
The names are re-exported here so a caller need not know which contract a name
belongs to.
"""

from zeroshot.pipeline.messages.contracts.drawings import (
    DRAWING_SUFFIXES,
    ORTHOGRAPHIC_VIEWS,
    VIEW_FRAME,
    ClaimSource,
    Dimension,
    DimensionKind,
    DrawingEvidence,
    DrawingSheet,
    DrawingSource,
    DrawnEntity,
    EdgeStyle,
    Parameter,
    ParameterName,
    View,
    edge_style_for_linetype,
    view_frame_sentence,
)
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
    Axis,
    FeatureGeometry,
    GeometryKind,
    SemanticFeature,
    SemanticHypothesis,
)
from zeroshot.pipeline.messages.contracts.stages import (
    PIPELINE_STAGES,
    REASONING_STAGES,
    PipelineStage,
    ReasoningStage,
    next_stage,
)

__all__ = [
    "DRAWING_SUFFIXES",
    "ORTHOGRAPHIC_VIEWS",
    "PIPELINE_STAGES",
    "REASONING_STAGES",
    "VIEW_FRAME",
    "Axis",
    "ClaimSource",
    "Dimension",
    "DimensionKind",
    "DrawingEvidence",
    "DrawingSheet",
    "DrawingSource",
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
    "edge_style_for_linetype",
    "linearise",
    "next_stage",
    "operation_heading",
    "render_plan",
    "resolve_reference",
    "view_frame_sentence",
]
