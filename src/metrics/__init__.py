"""Metrics used by SFT validation and later policy-training pipelines.

Importing this package registers every metric family, so a class name written
in ``evaluation.metrics`` resolves without the config naming a module. The
family modules must therefore stay import-light: CAD kernels (OCC, cadquery,
cadgenbench) are imported inside ``score``, which only ever runs in the
isolated scoring subprocess.
"""

from .base import CADMetric, MetricSample
from .bbox import BoundingBoxMetric
from .cad import cad_error_histogram, cad_execution_metrics
from .cadgenbench import CADGenBenchScoreMetric
from .chamfer import ChamferMetric
from .eccv import ECCVChallengeMetric
from .execution import CadExecutionMetric
from .geometry import (
    align_meshes,
    bbox_dimension_error_mm,
    bbox_dimension_error_relative,
    finite_float,
    max_bbox_error_mm,
    max_bbox_error_relative,
    mean_or_zero,
    median_or_zero,
    normalized_voxel_iou,
    surface_chamfer_distance,
    symmetric_chamfer_distance,
)
from .language import causal_token_accuracy, exact_match_rate, mean_edit_similarity
from .registry import (
    METRIC_REGISTRY,
    build_metrics,
    empty_metric_row,
    register_metric,
    required_artifacts,
)
from .voxel_iou import VoxelIoUMetric

__all__ = [
    "BoundingBoxMetric",
    "CADGenBenchScoreMetric",
    "CADMetric",
    "CadExecutionMetric",
    "ChamferMetric",
    "ECCVChallengeMetric",
    "METRIC_REGISTRY",
    "MetricSample",
    "VoxelIoUMetric",
    "align_meshes",
    "bbox_dimension_error_mm",
    "bbox_dimension_error_relative",
    "build_metrics",
    "cad_error_histogram",
    "cad_execution_metrics",
    "causal_token_accuracy",
    "empty_metric_row",
    "exact_match_rate",
    "finite_float",
    "max_bbox_error_mm",
    "max_bbox_error_relative",
    "mean_edit_similarity",
    "mean_or_zero",
    "median_or_zero",
    "normalized_voxel_iou",
    "register_metric",
    "required_artifacts",
    "surface_chamfer_distance",
    "symmetric_chamfer_distance",
]
