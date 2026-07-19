"""Orchestrate isolated CAD execution and geometry scoring."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
import tempfile
from typing import Iterable, Mapping

import numpy as np

from src.evaluation.executor import execute_cadquery
from src.metrics.cad import cad_error_histogram, cad_execution_metrics
from src.metrics.geometry import (
    aggregate_geometry_metrics,
    bbox_dimension_error_mm,
    surface_chamfer_distance,
    translation_aligned_voxel_iou,
)


@dataclass(frozen=True)
class EvaluationConfig:
    timeout_s: float = 120.0
    tessellation_tolerance: float = 0.1
    volume_tolerance: float = 1e-6
    voxel_resolution: int = 64
    compute_normalized_iou: bool = False
    compute_chamfer: bool = True
    chamfer_points: int = 8192
    chamfer_seed: int = 0


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


def _load_mesh(path: Path):
    import trimesh

    with np.load(path, allow_pickle=False) as archive:
        vertices = np.asarray(archive["vertices"], dtype=np.float64)
        faces = np.asarray(archive["faces"], dtype=np.int64)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _empty_row(sample_id: str) -> dict[str, object]:
    return {
        "id": sample_id,
        "exec_ok": False,
        "has_result": False,
        "is_valid": False,
        "volume": 0.0,
        "valid": False,
        "error": None,
        "iou": None,
        "iou_normalized": None,
        "bbox_pred_mm": None,
        "bbox_target_mm": None,
        "bbox_error_mm": None,
        "max_bbox_error_mm": None,
        "chamfer_mm2": None,
        "chamfer_normalized": None,
    }


def evaluate_prediction(
    item: EvaluationItem,
    *,
    config: EvaluationConfig | None = None,
) -> dict[str, object]:
    """Execute and score one prediction against target CadQuery code or STEP."""

    cfg = config or EvaluationConfig()
    row = _empty_row(item.sample_id)
    target_path = None if item.target_step_path is None else Path(item.target_step_path)
    if target_path is not None and not target_path.is_file():
        row["error"] = "missing_target_step"
        return row

    with tempfile.TemporaryDirectory(prefix="drawing2cad-eval-") as temporary:
        directory = Path(temporary)
        predicted_mesh_path = directory / "predicted.npz"
        target_mesh_path = directory / "target.npz"
        execution = execute_cadquery(
            item.prediction,
            timeout_s=cfg.timeout_s,
            mesh_output_path=predicted_mesh_path,
            tessellation_tolerance=cfg.tessellation_tolerance,
            volume_tolerance=cfg.volume_tolerance,
        )
        row.update(asdict(execution))
        if not execution.valid or not predicted_mesh_path.is_file():
            return row

        # STEP import/tessellation also passes through the same process boundary;
        # malformed kernel input therefore cannot crash the training process.
        if item.target_code is not None:
            target_source = item.target_code
        else:
            assert target_path is not None
            target_source = (
                "import cadquery as cq\n"
                f"result = cq.importers.importStep({str(target_path)!r})\n"
            )
        target_execution = execute_cadquery(
            target_source,
            timeout_s=cfg.timeout_s,
            mesh_output_path=target_mesh_path,
            tessellation_tolerance=cfg.tessellation_tolerance,
            volume_tolerance=cfg.volume_tolerance,
        )
        if not target_execution.valid or not target_mesh_path.is_file():
            row["error"] = f"target:{target_execution.error or 'invalid_shape'}"
            return row

        try:
            predicted_mesh = _load_mesh(predicted_mesh_path)
            target_mesh = _load_mesh(target_mesh_path)
            bbox_error = bbox_dimension_error_mm(predicted_mesh, target_mesh)
            row["bbox_pred_mm"] = np.sort(predicted_mesh.extents).tolist()
            row["bbox_target_mm"] = np.sort(target_mesh.extents).tolist()
            row["bbox_error_mm"] = bbox_error.tolist()
            row["max_bbox_error_mm"] = float(np.max(bbox_error))
            row["iou"] = translation_aligned_voxel_iou(
                predicted_mesh,
                target_mesh,
                resolution=cfg.voxel_resolution,
            )
            if cfg.compute_normalized_iou:
                row["iou_normalized"] = translation_aligned_voxel_iou(
                    predicted_mesh,
                    target_mesh,
                    resolution=cfg.voxel_resolution,
                    normalize_scale=True,
                )
            if cfg.compute_chamfer:
                row["chamfer_mm2"] = surface_chamfer_distance(
                    predicted_mesh,
                    target_mesh,
                    num_points=cfg.chamfer_points,
                    seed=cfg.chamfer_seed,
                )
                row["chamfer_normalized"] = surface_chamfer_distance(
                    predicted_mesh,
                    target_mesh,
                    num_points=cfg.chamfer_points,
                    seed=cfg.chamfer_seed,
                    normalize_scale=True,
                )
            row["error"] = None
        except Exception as error:
            message = " ".join(str(error).split())
            row["error"] = f"geometry:{type(error).__name__}: {message}"[:500]
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
    independent (its own tempdir and its own ``spawn`` CadQuery subprocesses) and
    spends nearly all its wall-clock time blocked in ``process.join``, so a
    thread pool overlaps those waits without touching shared state. ``None`` or
    a value ``<= 1`` keeps the original strictly-sequential behavior. Output
    order always matches input order regardless of completion order.
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
    prefix: str = "val",
) -> dict[str, float | int]:
    """Merge CAD rates and geometry metrics under log-ready keys."""

    materialized = list(rows)
    return {
        **cad_execution_metrics(materialized, prefix=prefix),
        **aggregate_geometry_metrics(materialized, prefix=prefix),
    }


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
