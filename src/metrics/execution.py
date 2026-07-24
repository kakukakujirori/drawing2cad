"""CadQuery execution and validity rates as a selectable metric family."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.metrics.base import CADMetric, MetricSample
from src.metrics.cad import cad_execution_metrics
from src.metrics.registry import register_metric


@register_metric
@dataclass(frozen=True)
class CadExecutionMetric(CADMetric):
    """Aggregate the execution outcome the pipeline already records.

    ``exec_ok``/``has_result``/``valid`` are written into every row by the
    isolated CadQuery execution itself -- they exist whether or not this family
    is selected, because other metrics gate on them. This family only decides
    whether their rates are reported, so :meth:`score` contributes nothing.
    """

    requires = frozenset()
    row_keys = ()

    def score(self, sample: MetricSample) -> Mapping[str, Any]:
        del sample
        return {}

    def reduce(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        prefix: str,
    ) -> Mapping[str, float | int]:
        return cad_execution_metrics(rows, prefix=prefix)


__all__ = ["CadExecutionMetric"]
