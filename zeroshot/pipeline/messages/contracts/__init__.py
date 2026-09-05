"""What each reasoning stage must answer with, and how it is shown downstream.

One module per stage's answer, split along the seam the pipeline already has.
`operations` reads `semantics` because a plan is made from a hypothesis and
says which of its features each step builds; nothing reads back the other way.
The names are re-exported here so a caller need not know which stage a contract
belongs to.
"""

from zeroshot.pipeline.messages.contracts.drawings import (
    VIEW_FRAME,
    DrawingEvidence,
    DrawingSheet,
    DrawingSource,
    DrawnEntity,
    EdgeStyle,
    View,
    edge_style_for_linetype,
)
from zeroshot.pipeline.messages.contracts.operations import (
    Operation,
    OperationPlan,
    OperationVerb,
    linearise,
)
from zeroshot.pipeline.messages.contracts.parameters import Parameter, ParameterName
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
    "PIPELINE_STAGES",
    "REASONING_STAGES",
    "VIEW_FRAME",
    "Axis",
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
]
