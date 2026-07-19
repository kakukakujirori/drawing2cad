"""Translation-aligned, scale-preserving mesh geometry metrics."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.floating[Any]]


def _mesh_extents(mesh_or_extents: object) -> FloatArray:
    raw = getattr(mesh_or_extents, "extents", mesh_or_extents)
    extents = np.asarray(raw, dtype=np.float64)
    if extents.shape != (3,):
        raise ValueError(f"mesh extents must have shape [3], got {extents.shape}")
    if not np.all(np.isfinite(extents)) or np.any(extents < 0.0):
        raise ValueError(f"mesh extents must be finite and non-negative, got {extents}")
    return extents


def bbox_dimension_error_mm(
    predicted: object,
    target: object,
    *,
    sort_dimensions: bool = True,
) -> FloatArray:
    """Return absolute per-axis bounding-box dimension errors in millimetres.

    Sorted dimensions are the default because CAD orientation can differ while
    the part dimensions remain equivalent.
    """

    pred = _mesh_extents(predicted)
    gt = _mesh_extents(target)
    if sort_dimensions:
        pred, gt = np.sort(pred), np.sort(gt)
    return np.abs(pred - gt)


def max_bbox_error_mm(
    predicted: object,
    target: object,
    *,
    sort_dimensions: bool = True,
) -> float:
    return float(
        np.max(
            bbox_dimension_error_mm(
                predicted, target, sort_dimensions=sort_dimensions
            )
        )
    )


def _aligned_mesh(mesh: Any, *, normalize_scale: bool) -> Any:
    aligned = mesh.copy()
    bounds = np.asarray(aligned.bounds, dtype=np.float64)
    if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)):
        raise ValueError("mesh must have finite [2, 3] bounds")
    if normalize_scale:
        aligned.apply_translation(-(bounds[0] + bounds[1]) / 2.0)
        extent = float(np.max(_mesh_extents(aligned)))
        if extent <= 1e-12:
            raise ValueError("cannot normalize a mesh with zero maximum extent")
        aligned.apply_scale(1.0 / extent)
    else:
        aligned.apply_translation(-bounds[0])
    return aligned


def _voxel_indices(mesh: Any, pitch: float) -> set[tuple[int, int, int]]:
    grid = mesh.voxelized(pitch)
    try:
        grid = grid.fill()
    except Exception:
        # Some trimesh voxel backends cannot fill a non-watertight mesh. Surface
        # occupancy is still a deterministic fallback for the standalone metric.
        pass
    points = np.asarray(grid.points, dtype=np.float64)
    if points.size == 0:
        return set()
    indices = np.floor(points / pitch + 1e-6).astype(np.int64)
    return {tuple(int(value) for value in row) for row in indices}


def translation_aligned_voxel_iou(
    predicted_mesh: Any,
    target_mesh: Any,
    *,
    resolution: int = 64,
    normalize_scale: bool = False,
) -> float:
    """Compute bbox-min-aligned voxel IoU in a shared scale-preserving grid.

    With ``normalize_scale=False`` the pitch is measured in millimetres from the
    target maximum extent, so incorrect predicted dimensions reduce IoU. The
    optional normalized mode is provided only for comparison with unit-box CAD
    benchmarks.
    """

    if isinstance(resolution, bool) or not isinstance(resolution, int) or resolution <= 0:
        raise ValueError(f"resolution must be a positive integer, got {resolution!r}")
    pred = _aligned_mesh(predicted_mesh, normalize_scale=normalize_scale)
    gt = _aligned_mesh(target_mesh, normalize_scale=normalize_scale)
    target_extent = 1.0 if normalize_scale else float(np.max(_mesh_extents(gt)))
    if target_extent <= 1e-12:
        raise ValueError("target mesh has zero maximum extent")
    pitch = target_extent / resolution
    pred_voxels = _voxel_indices(pred, pitch)
    target_voxels = _voxel_indices(gt, pitch)
    if not pred_voxels or not target_voxels:
        return 0.0
    union = pred_voxels | target_voxels
    return float(len(pred_voxels & target_voxels) / len(union)) if union else 0.0


def symmetric_chamfer_distance(
    predicted_points: ArrayLike,
    target_points: ArrayLike,
) -> float:
    """Return the sum of bidirectional mean squared nearest-neighbour distances."""

    pred = np.asarray(predicted_points, dtype=np.float64)
    gt = np.asarray(target_points, dtype=np.float64)
    for label, points in (("predicted", pred), ("target", gt)):
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            raise ValueError(f"{label} points must have non-empty shape [N, 3]")
        if not np.all(np.isfinite(points)):
            raise ValueError(f"{label} points must be finite")
    try:
        from scipy.spatial import cKDTree

        pred_to_gt = cKDTree(gt).query(pred, k=1)[0]
        gt_to_pred = cKDTree(pred).query(gt, k=1)[0]
        return float(np.mean(pred_to_gt**2) + np.mean(gt_to_pred**2))
    except ImportError:
        return float(
            np.mean(_nearest_squared_distances(pred, gt))
            + np.mean(_nearest_squared_distances(gt, pred))
        )


def _nearest_squared_distances(
    query: FloatArray,
    reference: FloatArray,
    *,
    chunk_size: int = 1024,
) -> FloatArray:
    result = np.empty(len(query), dtype=np.float64)
    for start in range(0, len(query), chunk_size):
        delta = query[start : start + chunk_size, None, :] - reference[None, :, :]
        result[start : start + chunk_size] = np.min(np.sum(delta * delta, axis=-1), axis=1)
    return result


def surface_chamfer_distance(
    predicted_mesh: Any,
    target_mesh: Any,
    *,
    num_points: int = 8192,
    seed: int = 0,
    normalize_scale: bool = False,
) -> float:
    """Sample mesh surfaces and compute translation-aligned Chamfer distance.

    The absolute result uses squared millimetres. In normalized mode each mesh is
    independently divided by its maximum bbox extent.
    """

    if isinstance(num_points, bool) or not isinstance(num_points, int) or num_points <= 0:
        raise ValueError(f"num_points must be a positive integer, got {num_points!r}")
    import trimesh

    pred = _aligned_mesh(predicted_mesh, normalize_scale=normalize_scale)
    gt = _aligned_mesh(target_mesh, normalize_scale=normalize_scale)
    pred_points, _ = trimesh.sample.sample_surface(pred, num_points, seed=seed)
    gt_points, _ = trimesh.sample.sample_surface(gt, num_points, seed=seed)
    return symmetric_chamfer_distance(pred_points, gt_points)


def aggregate_geometry_metrics(
    rows: Iterable[Mapping[str, object]],
    *,
    prefix: str = "val",
) -> dict[str, float | int]:
    """Aggregate geometry rows, scoring failed/invalid predictions as zero IoU."""

    materialized = list(rows)
    total = len(materialized)
    iou_with_failures: list[float] = []
    valid_ious: list[float] = []
    bbox_errors: list[float] = []
    chamfer_mm2: list[float] = []
    chamfer_normalized: list[float] = []
    normalized_ious: list[float] = []
    for row in materialized:
        raw_iou = row.get("iou")
        valid = bool(row.get("valid", False))
        iou = _finite_float(raw_iou)
        iou_with_failures.append(iou if valid and iou is not None else 0.0)
        if valid and iou is not None:
            valid_ious.append(iou)
        normalized_iou = _finite_float(row.get("iou_normalized"))
        if valid and normalized_iou is not None:
            normalized_ious.append(normalized_iou)
        bbox = _finite_float(row.get("max_bbox_error_mm"))
        if bbox is not None:
            bbox_errors.append(bbox)
        chamfer = _finite_float(row.get("chamfer_mm2"))
        if chamfer is not None:
            chamfer_mm2.append(chamfer)
        normalized = _finite_float(row.get("chamfer_normalized"))
        if normalized is not None:
            chamfer_normalized.append(normalized)

    stem = f"{prefix}/" if prefix else ""
    metrics: dict[str, float | int] = {
        f"{stem}iou_scored_n": len(valid_ious),
        f"{stem}mean_iou_including_failures": _mean(iou_with_failures),
        f"{stem}mean_iou_valid_only": _mean(valid_ious),
        f"{stem}median_iou": _median(iou_with_failures),
        f"{stem}bbox_scored_n": len(bbox_errors),
        f"{stem}mean_max_bbox_error_mm": _mean(bbox_errors),
    }
    if chamfer_mm2:
        metrics[f"{stem}chamfer_mm2_scored_n"] = len(chamfer_mm2)
        metrics[f"{stem}mean_chamfer_mm2"] = _mean(chamfer_mm2)
    if chamfer_normalized:
        metrics[f"{stem}chamfer_normalized_scored_n"] = len(chamfer_normalized)
        metrics[f"{stem}mean_chamfer_normalized"] = _mean(chamfer_normalized)
    if normalized_ious:
        metrics[f"{stem}iou_normalized_scored_n"] = len(normalized_ious)
        metrics[f"{stem}mean_iou_normalized"] = _mean(normalized_ious)
    if total == 0:
        # Keep the checkpoint monitor stable even for an accidentally empty subset.
        metrics[f"{stem}mean_iou_including_failures"] = 0.0
    return metrics


def _finite_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _median(values: list[float]) -> float:
    return float(np.median(values)) if values else 0.0


__all__ = [
    "aggregate_geometry_metrics",
    "bbox_dimension_error_mm",
    "max_bbox_error_mm",
    "surface_chamfer_distance",
    "symmetric_chamfer_distance",
    "translation_aligned_voxel_iou",
]
