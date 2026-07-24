"""CADGenBench CAD Score and its component axes.

Definitions come from the installed ``cadgenbench`` package rather than from a
copy here, so this family cannot drift away from the published benchmark. The
functions used are exactly the ones the official orchestrator
(``cadgenbench.eval.evaluate``) calls; only the report rendering, which needs a
working X/VTK stack and contributes nothing numerically, is skipped. The same
approach is already taken by ``bench/oracle_terminal_blend_probe.py``.

CAD Score is the weight-renormalized mean over the axes that exist for a
fixture. Our validation samples carry no interface jig sub-volumes, so the
present axes are shape (weight 0.4) and topology (0.2), i.e. an effective 2/3
and 1/3. An invalid candidate scores zero.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.metrics.base import GT_STEP, PRED_STEP, CADMetric, MetricSample, stem
from src.metrics.geometry import finite_float, mean_or_zero
from src.metrics.registry import register_metric


_AXIS_KEYS = (
    "shape_surface_distance_f1",
    "shape_volume_iou",
    "shape_similarity_score",
    "topology_match_score",
)


@register_metric
@dataclass(frozen=True)
class CADGenBenchScoreMetric(CADMetric):
    """CAD Score plus the surface-distance F1, volume IoU and topology axes.

    ``normalize_to_gt_bbox`` rescales the prediction onto the ground truth's
    bounding box before scoring. It defaults to on because both component
    metrics are threshold-based in absolute length (the surface-distance
    threshold is 0.5 percent of the GT bounding-box diagonal) while our inputs
    carry no scale, so without it a correctly shaped prediction at the wrong
    size scores near zero. Turn it off to reproduce the benchmark's own
    absolute-scale numbers.
    """

    normalize_to_gt_bbox: bool = True
    align: bool = True

    requires = frozenset({PRED_STEP, GT_STEP})
    row_keys = (
        "cad_score",
        "shape_surface_distance_f1",
        "shape_volume_iou",
        "shape_similarity_score",
        "topology_match_score",
        "cadgenbench_is_valid",
        "cadgenbench_alignment_rmse",
        "cadgenbench_normalization_scale",
    )

    def score(self, sample: MetricSample) -> Mapping[str, Any]:
        import os

        from src.metrics._ocp_compat import ensure_ocp_static_aliases

        ensure_ocp_static_aliases()
        # CADGenBench guards each tessellation with a nested killable process.
        # Our own scoring worker already is that boundary, with a timeout that
        # covers the whole sample, so the inner pool only adds a second
        # interpreter that would need the binding shim above re-applied. Read
        # once at import of ``cadgenbench.common.validity``, hence set first.
        os.environ.setdefault("CADGENBENCH_MESH_TIMEOUT_S", "0")

        from cadgenbench.eval.evaluate import (
            GENERATION_AXIS_WEIGHTS,
            _cad_score,
            _topology_metrics_dict,
            _validation_dict,
        )
        from cadgenbench.eval.shape_similarity import compare_step_files

        gt_step = sample.gt_step()
        candidate = sample.pred_step()
        row: dict[str, Any] = {"cadgenbench_normalization_scale": None}
        if self.normalize_to_gt_bbox:
            from src.metrics.step_transform import normalize_step_to_reference

            normalized = sample.workdir / f"{sample.sample_id}.cadgenbench.step"
            normalization = normalize_step_to_reference(candidate, gt_step, normalized)
            row["cadgenbench_normalization_scale"] = normalization.scale
            candidate = normalized

        validation = _validation_dict(candidate)
        comparison = compare_step_files(
            candidate,
            gt_step,
            align=bool(self.align),
            # Keep every derived artifact inside this sample's scratch dir; the
            # default writes the aligned STEP next to the candidate, which is a
            # persisted prediction directory during training.
            aligned_output=sample.workdir / f"{sample.sample_id}.aligned.step",
        )
        topology = _topology_metrics_dict(candidate, gt_step, validation)

        row.update(
            {
                "cadgenbench_is_valid": bool(validation.get("is_valid", False)),
                "cadgenbench_alignment_rmse": _optional_float(
                    comparison.alignment_rmse
                ),
                "shape_surface_distance_f1": _optional_float(
                    comparison.scores.get("shape_surface_distance_f1")
                ),
                "shape_volume_iou": _optional_float(
                    comparison.scores.get("shape_volume_iou")
                ),
                "shape_similarity_score": _optional_float(
                    comparison.scores.get("shape_similarity_score")
                ),
                "topology_match_score": _optional_float(topology.get("score")),
                "cad_score": float(
                    _cad_score(
                        scores=comparison.scores,
                        interface_metrics={},
                        topology_metrics=topology,
                        validation=validation,
                        weights=GENERATION_AXIS_WEIGHTS,
                    )
                ),
            }
        )
        return row

    def reduce(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        prefix: str,
    ) -> Mapping[str, float | int]:
        head = stem(prefix)
        scored: list[float] = []
        with_failures: list[float] = []
        for row in rows:
            value = finite_float(row.get("cad_score"))
            # A prediction that never produced a STEP is a zero, exactly as it
            # is on the benchmark's own leaderboard.
            with_failures.append(value if value is not None else 0.0)
            if value is not None:
                scored.append(value)
        metrics: dict[str, float | int] = {
            f"{head}cad_score_scored_n": len(scored),
            f"{head}mean_cad_score": mean_or_zero(with_failures),
            f"{head}mean_cad_score_scored_only": mean_or_zero(scored),
        }
        for key in _AXIS_KEYS:
            axis = [
                value if value is not None else 0.0
                for value in (finite_float(row.get(key)) for row in rows)
            ]
            metrics[f"{head}mean_{key}"] = mean_or_zero(axis)
        return metrics


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = ["CADGenBenchScoreMetric"]
