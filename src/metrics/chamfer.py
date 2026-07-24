"""Surface Chamfer distance as a selectable metric family."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.metrics.base import GT_MESH, PRED_MESH, CADMetric, MetricSample, stem
from src.metrics.geometry import (
    finite_float,
    mean_or_zero,
    surface_chamfer_distance,
)
from src.metrics.registry import register_metric


@register_metric
@dataclass(frozen=True)
class ChamferMetric(CADMetric):
    """Centre-aligned Chamfer distance, in normalized and/or absolute scale.

    Both modes are spelled out rather than relying on the function default, so
    the row key and the scale convention it names cannot drift apart. Selecting
    this family replaces the former ``compute_chamfer`` flag.
    """

    num_points: int = 8192
    seed: int = 0
    normalized: bool = True
    absolute: bool = True

    requires = frozenset({PRED_MESH, GT_MESH})
    row_keys = ("chamfer_normalized", "chamfer_mm2")

    def score(self, sample: MetricSample) -> Mapping[str, Any]:
        predicted = sample.pred_mesh()
        target = sample.gt_mesh()
        row: dict[str, Any] = {}
        if self.normalized:
            row["chamfer_normalized"] = surface_chamfer_distance(
                predicted,
                target,
                num_points=int(self.num_points),
                seed=int(self.seed),
                normalize_scale=True,
            )
        if self.absolute:
            row["chamfer_mm2"] = surface_chamfer_distance(
                predicted,
                target,
                num_points=int(self.num_points),
                seed=int(self.seed),
                normalize_scale=False,
            )
        return row

    def reduce(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        prefix: str,
    ) -> Mapping[str, float | int]:
        head = stem(prefix)
        metrics: dict[str, float | int] = {}
        for key, label in (
            ("chamfer_mm2", "chamfer_mm2"),
            ("chamfer_normalized", "chamfer_normalized"),
        ):
            values = [
                value
                for value in (finite_float(row.get(key)) for row in rows)
                if value is not None
            ]
            if values:
                metrics[f"{head}{label}_scored_n"] = len(values)
                metrics[f"{head}mean_{label}"] = mean_or_zero(values)
        return metrics


__all__ = ["ChamferMetric"]
