"""Bounding-box dimension errors as a selectable metric family."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.metrics.base import GT_MESH, PRED_MESH, CADMetric, MetricSample, stem
from src.metrics.geometry import (
    bbox_dimension_error_mm,
    bbox_dimension_error_relative,
    finite_float,
    mean_or_zero,
)
from src.metrics.registry import register_metric


@register_metric
@dataclass(frozen=True)
class BoundingBoxMetric(CADMetric):
    """Absolute and target-relative bounding-box dimension errors.

    A diagnostic rather than a target: the drawings carry no dimension
    information, so absolute size is not recoverable. It still exposes
    absolute-scale drift, which the shape-only metrics cannot see.
    """

    sort_dimensions: bool = True

    requires = frozenset({PRED_MESH, GT_MESH})
    row_keys = (
        "bbox_pred_mm",
        "bbox_target_mm",
        "bbox_error_mm",
        "max_bbox_error_mm",
        "max_bbox_error_relative",
    )

    def score(self, sample: MetricSample) -> Mapping[str, Any]:
        predicted = sample.pred_mesh()
        target = sample.gt_mesh()
        sort_dimensions = bool(self.sort_dimensions)
        error = bbox_dimension_error_mm(
            predicted, target, sort_dimensions=sort_dimensions
        )
        relative = bbox_dimension_error_relative(
            predicted, target, sort_dimensions=sort_dimensions
        )
        return {
            "bbox_pred_mm": np.sort(predicted.extents).tolist(),
            "bbox_target_mm": np.sort(target.extents).tolist(),
            "bbox_error_mm": error.tolist(),
            "max_bbox_error_mm": float(np.max(error)),
            "max_bbox_error_relative": float(np.max(relative)),
        }

    def reduce(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        prefix: str,
    ) -> Mapping[str, float | int]:
        absolute: list[float] = []
        relative: list[float] = []
        for row in rows:
            value = finite_float(row.get("max_bbox_error_mm"))
            if value is not None:
                absolute.append(value)
            value = finite_float(row.get("max_bbox_error_relative"))
            if value is not None:
                relative.append(value)
        head = stem(prefix)
        metrics: dict[str, float | int] = {
            f"{head}bbox_scored_n": len(absolute),
            f"{head}mean_max_bbox_error_mm": mean_or_zero(absolute),
        }
        if relative:
            metrics[f"{head}bbox_relative_scored_n"] = len(relative)
            metrics[f"{head}mean_max_bbox_error_relative"] = mean_or_zero(relative)
        return metrics


__all__ = ["BoundingBoxMetric"]
