"""What each reasoning stage must answer with, and how it is shown downstream.

One module per stage's answer. They share no vocabulary -- an operation refers
to a feature by its id, not by its type -- so the split is along the seam the
pipeline already has, and the names are re-exported here so a caller need not
know which stage a contract belongs to.
"""

from zeroshot.pipeline.messages.contracts.operations import (
    Operation,
    OperationPlan,
    PlanCoverage,
    linearise,
    plan_coverage,
    render_plan,
    render_plan_coverage,
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

__all__ = [
    "VIEW_FRAME",
    "Axis",
    "ClaimSource",
    "DrawnEntity",
    "EdgeStyle",
    "FeatureGeometry",
    "GeometryKind",
    "Operation",
    "OperationPlan",
    "Parameter",
    "ParameterName",
    "PlanCoverage",
    "SemanticFeature",
    "SemanticHypothesis",
    "View",
    "ViewEvidence",
    "edge_style_for_linetype",
    "linearise",
    "plan_coverage",
    "render_hypothesis",
    "render_plan",
    "render_plan_coverage",
    "view_frame_sentence",
]
