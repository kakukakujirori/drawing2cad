"""Orchestrate isolated CAD execution and metric scoring."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
import tempfile
from typing import Iterable, Mapping

from src.evaluation.executor import execute_cadquery
from src.evaluation.scoring import score_sample
from src.metrics.base import GT_MESH, GT_STEP, PRED_MESH, PRED_STEP, CADMetric
from src.metrics.cad import cad_error_histogram
from src.metrics.registry import empty_metric_row, required_artifacts


@dataclass(frozen=True)
class EvaluationConfig:
    """Everything the scoring pipeline needs that is not a metric parameter.

    Metric-specific knobs (voxel resolution, Chamfer sample count, whether a
    prediction is normalized onto the target) live on the metric instances in
    ``metrics``, so adding a metric never widens this object.
    """

    timeout_s: float = 120.0
    tessellation_tolerance: float = 0.1
    volume_tolerance: float = 1e-6
    # Matches the per-sample budget the ECCV challenge's own evaluator allows
    # itself; its entity assignment is the slowest thing scored here.
    metric_timeout_s: float = 600.0
    metrics: tuple[CADMetric, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class EvaluationItem:
    sample_id: str
    prediction: str
    target_step_path: str | Path | None = None
    target_code: str | None = None

    def __post_init__(self) -> None:
        available = int(self.target_step_path is not None) + int(
            self.target_code is not None
        )
        if available != 1:
            raise ValueError(
                "exactly one of target_step_path and target_code is required"
            )


def _empty_row(sample_id: str, metrics: Sequence[CADMetric]) -> dict[str, object]:
    """A row for a sample nothing could be measured on.

    The execution columns are always present because the pipeline itself writes
    them and the metrics gate on them; the metric columns come from whichever
    families are selected.
    """

    return {
        "id": sample_id,
        "exec_ok": False,
        "has_result": False,
        "is_valid": False,
        "volume": 0.0,
        "valid": False,
        "error": None,
        **empty_metric_row(metrics),
    }


def _import_step_source(step_path: Path) -> str:
    return (
        f"import cadquery as cq\nresult = cq.importers.importStep({str(step_path)!r})\n"
    )


def evaluate_prediction(
    item: EvaluationItem,
    *,
    config: EvaluationConfig | None = None,
) -> dict[str, object]:
    """Execute one prediction and score it against the target with every metric.

    Artifacts are produced on demand: the union of what the selected metrics
    declare in ``requires`` decides whether the prediction is tessellated,
    exported as STEP, and whether the target is run through CadQuery at all.
    """

    cfg = config or EvaluationConfig()
    metrics = tuple(cfg.metrics)
    row = _empty_row(item.sample_id, metrics)
    needed = required_artifacts(metrics)
    target_path = None if item.target_step_path is None else Path(item.target_step_path)
    if target_path is not None and not target_path.is_file():
        row["error"] = "missing_target_step"
        return row

    with tempfile.TemporaryDirectory(prefix="drawing2cad-eval-") as temporary:
        directory = Path(temporary)
        predicted_mesh_path = (
            directory / "predicted.npz" if PRED_MESH in needed else None
        )
        predicted_step_path = (
            directory / "predicted.step" if PRED_STEP in needed else None
        )
        execution = execute_cadquery(
            item.prediction,
            timeout_s=cfg.timeout_s,
            mesh_output_path=predicted_mesh_path,
            step_output_path=predicted_step_path,
            tessellation_tolerance=cfg.tessellation_tolerance,
            volume_tolerance=cfg.volume_tolerance,
        )
        row.update(asdict(execution))
        if not execution.valid:
            return row
        if predicted_mesh_path is not None and not predicted_mesh_path.is_file():
            row["error"] = "missing_predicted_mesh"
            return row
        if predicted_step_path is not None and not predicted_step_path.is_file():
            row["error"] = "missing_predicted_step"
            return row

        # A target given as a STEP file is already the artifact a STEP metric
        # wants, so it is used as is. Anything that must be produced goes
        # through the same process boundary as the prediction, and malformed
        # kernel input therefore cannot crash the training process either.
        target_mesh_path = directory / "target.npz" if GT_MESH in needed else None
        exported_target_step = (
            directory / "target.step"
            if GT_STEP in needed and target_path is None
            else None
        )
        target_step_path = target_path if GT_STEP in needed else None
        if exported_target_step is not None:
            target_step_path = exported_target_step
        if target_mesh_path is not None or exported_target_step is not None:
            if item.target_code is not None:
                target_source = item.target_code
            else:
                assert target_path is not None
                target_source = _import_step_source(target_path)
            target_execution = execute_cadquery(
                target_source,
                timeout_s=cfg.timeout_s,
                mesh_output_path=target_mesh_path,
                step_output_path=exported_target_step,
                tessellation_tolerance=cfg.tessellation_tolerance,
                volume_tolerance=cfg.volume_tolerance,
            )
            if not target_execution.valid:
                row["error"] = f"target:{target_execution.error or 'invalid_shape'}"
                return row
            for produced in (target_mesh_path, exported_target_step):
                if produced is not None and not produced.is_file():
                    row["error"] = "target:missing_artifact"
                    return row

        scoring = score_sample(
            item.sample_id,
            workdir=directory / "scoring",
            metrics=metrics,
            pred_mesh_path=predicted_mesh_path,
            gt_mesh_path=target_mesh_path,
            pred_step_path=predicted_step_path,
            gt_step_path=target_step_path,
            timeout_s=cfg.metric_timeout_s,
        )
        row.update(scoring.columns)
        if scoring.error is not None:
            row["error"] = f"scoring:{scoring.error}"
        elif scoring.metric_errors:
            # Partial failure: the metrics that did run keep their columns, and
            # the histogram still shows that something went wrong here.
            row["error"] = (
                "metric:"
                + "; ".join(
                    f"{name}: {message}"
                    for name, message in sorted(scoring.metric_errors.items())
                )[:500]
            )
        else:
            row["error"] = None
    return row


def evaluate_predictions(
    items: Iterable[EvaluationItem],
    *,
    config: EvaluationConfig | None = None,
    on_item: Callable[[int], None] | None = None,
    max_workers: int | None = None,
) -> list[dict[str, object]]:
    """Evaluate an item sequence and return rows in the supplied order.

    ``on_item`` is invoked with ``1`` after each prediction is scored so callers
    can drive a progress bar over the slow isolated-execution loop.

    ``max_workers`` controls concurrency. Each ``evaluate_prediction`` is
    independent (its own tempdir and its own ``spawn`` subprocesses) and spends
    nearly all its wall-clock time blocked in ``process.join``, so a thread pool
    overlaps those waits without touching shared state. ``None`` or a value
    ``<= 1`` keeps the original strictly-sequential behavior. Output order
    always matches input order regardless of completion order.
    """

    materialized = list(items)
    if not materialized:
        return []
    workers = 1 if max_workers is None else max_workers
    if workers <= 1 or len(materialized) == 1:
        rows: list[dict[str, object]] = []
        for item in materialized:
            rows.append(evaluate_prediction(item, config=config))
            if on_item is not None:
                on_item(1)
        return rows

    results: list[dict[str, object] | None] = [None] * len(materialized)
    with ThreadPoolExecutor(max_workers=min(workers, len(materialized))) as pool:
        future_to_index = {
            pool.submit(evaluate_prediction, item, config=config): index
            for index, item in enumerate(materialized)
        }
        for future in as_completed(future_to_index):
            results[future_to_index[future]] = future.result()
            if on_item is not None:
                on_item(1)
    # Every slot is filled once as_completed drains; assert to satisfy typing.
    return [row for row in results if row is not None]


def aggregate_evaluation_metrics(
    rows: Iterable[Mapping[str, object]],
    *,
    metrics: Sequence[CADMetric],
    prefix: str = "val",
) -> dict[str, float | int]:
    """Merge every selected metric's aggregates under log-ready keys."""

    materialized = list(rows)
    merged: dict[str, float | int] = {}
    for metric in metrics:
        merged.update(metric.reduce(materialized, prefix=prefix))
    return merged


def evaluation_error_histogram(rows: Iterable[Mapping[str, object]]) -> dict[str, int]:
    return cad_error_histogram(rows)


__all__ = [
    "EvaluationConfig",
    "EvaluationItem",
    "aggregate_evaluation_metrics",
    "evaluate_prediction",
    "evaluate_predictions",
    "evaluation_error_histogram",
]
