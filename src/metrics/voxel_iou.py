"""Normalized voxel IoU as a selectable metric family."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.metrics.base import GT_MESH, PRED_MESH, CADMetric, MetricSample, stem
from src.metrics.geometry import (
    finite_float,
    mean_or_zero,
    median_or_zero,
    normalized_voxel_iou,
)
from src.metrics.registry import register_metric


@register_metric
@dataclass(frozen=True)
class VoxelIoUMetric(CADMetric):
    """Shape-only voxel IoU over centred, unit-box-normalized meshes.

    This family owns the default checkpoint monitor
    (``val/<root>/mean_iou_including_failures``), so its key names are fixed.
    """

    resolution: int = 64

    requires = frozenset({PRED_MESH, GT_MESH})
    row_keys = ("iou",)

    def score(self, sample: MetricSample) -> Mapping[str, Any]:
        return {
            "iou": normalized_voxel_iou(
                sample.pred_mesh(),
                sample.gt_mesh(),
                resolution=int(self.resolution),
            )
        }

    def reduce(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        prefix: str,
    ) -> Mapping[str, float | int]:
        # A failed or invalid prediction is a zero, not a missing value: the
        # headline mean must not improve by producing fewer scorable outputs.
        with_failures: list[float] = []
        valid_only: list[float] = []
        for row in rows:
            iou = finite_float(row.get("iou"))
            scored = bool(row.get("valid", False)) and iou is not None
            with_failures.append(iou if scored and iou is not None else 0.0)
            if scored and iou is not None:
                valid_only.append(iou)
        head = stem(prefix)
        return {
            f"{head}iou_scored_n": len(valid_only),
            f"{head}mean_iou_including_failures": mean_or_zero(with_failures),
            f"{head}mean_iou_valid_only": mean_or_zero(valid_only),
            f"{head}median_iou": median_or_zero(with_failures),
        }


__all__ = ["VoxelIoUMetric"]
