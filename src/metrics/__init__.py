"""Metrics used by SFT validation and later policy-training pipelines."""

from .cad import cad_error_histogram, cad_execution_metrics
from .geometry import (
    aggregate_geometry_metrics,
    align_meshes,
    bbox_dimension_error_mm,
    bbox_dimension_error_relative,
    max_bbox_error_mm,
    max_bbox_error_relative,
    normalized_voxel_iou,
    surface_chamfer_distance,
    symmetric_chamfer_distance,
)
from .language import causal_token_accuracy, exact_match_rate, mean_edit_similarity

__all__ = [
    "aggregate_geometry_metrics",
    "align_meshes",
    "bbox_dimension_error_mm",
    "bbox_dimension_error_relative",
    "cad_error_histogram",
    "cad_execution_metrics",
    "causal_token_accuracy",
    "exact_match_rate",
    "max_bbox_error_mm",
    "max_bbox_error_relative",
    "mean_edit_similarity",
    "normalized_voxel_iou",
    "surface_chamfer_distance",
    "symmetric_chamfer_distance",
]
