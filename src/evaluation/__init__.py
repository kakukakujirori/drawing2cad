"""Deterministic generation evaluation and isolated CAD execution."""

# Import executor first so its worker thread limits are set before evaluator
# imports NumPy through the geometry metric module.
from .executor import CadExecutionResult, execute_cadquery

from .evaluator import (
    EvaluationConfig,
    EvaluationItem,
    aggregate_evaluation_metrics,
    evaluate_prediction,
    evaluate_predictions,
    evaluation_error_histogram,
)
from .generate import CADGenerationEvaluator
from .scoring import ScoringResult, score_sample

__all__ = [
    "CadExecutionResult",
    "EvaluationConfig",
    "EvaluationItem",
    "CADGenerationEvaluator",
    "ScoringResult",
    "aggregate_evaluation_metrics",
    "evaluate_prediction",
    "evaluate_predictions",
    "evaluation_error_histogram",
    "execute_cadquery",
    "score_sample",
]
