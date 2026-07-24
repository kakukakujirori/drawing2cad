"""Shape-only mesh geometry metrics.

The drawings carry no dimension annotations and their sheet scale never reaches
the model (``SampleMetadata.drawing_scale`` is parsed but unused), so absolute
size is not recoverable from the input. Every headline metric here therefore
compares shape alone: both meshes are centred on their bounding-box centre and
divided by their own maximum extent. Absolute-millimetre variants remain
available where they cost nothing and stay useful as diagnostics.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.floating[Any]]
IntArray = NDArray[np.int64]


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
            bbox_dimension_error_mm(predicted, target, sort_dimensions=sort_dimensions)
        )
    )


def bbox_dimension_error_relative(
    predicted: object,
    target: object,
    *,
    sort_dimensions: bool = True,
) -> FloatArray:
    """Return per-axis bounding-box errors as a fraction of the target's size.

    Divided by the target's maximum extent so the result is comparable across
    parts and across datasets stored in different units, which the millimetre
    version is not.
    """

    reference = float(np.max(_mesh_extents(target)))
    if reference <= 1e-12:
        raise ValueError("target mesh has zero maximum extent")
    absolute = bbox_dimension_error_mm(
        predicted, target, sort_dimensions=sort_dimensions
    )
    return absolute / reference


def max_bbox_error_relative(
    predicted: object,
    target: object,
    *,
    sort_dimensions: bool = True,
) -> float:
    return float(
        np.max(
            bbox_dimension_error_relative(
                predicted, target, sort_dimensions=sort_dimensions
            )
        )
    )


def _centered_mesh(mesh: Any, *, normalize_scale: bool) -> Any:
    aligned = mesh.copy()
    bounds = np.asarray(aligned.bounds, dtype=np.float64)
    if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)):
        raise ValueError("mesh must have finite [2, 3] bounds")
    aligned.apply_translation(-(bounds[0] + bounds[1]) / 2.0)
    if normalize_scale:
        extent = float(np.max(_mesh_extents(aligned)))
        if extent <= 1e-12:
            raise ValueError("cannot normalize a mesh with zero maximum extent")
        aligned.apply_scale(1.0 / extent)
    return aligned


def align_meshes(
    predicted_mesh: Any,
    target_mesh: Any,
    *,
    normalize_scale: bool = True,
) -> tuple[Any, Any]:
    """Return copies of both meshes placed in a shared comparison frame.

    Both are translated so their bounding-box centre sits at the origin. With
    ``normalize_scale`` each is additionally divided by its own maximum extent,
    which discards size entirely and leaves a shape-only comparison in a unit
    box. Centring is the single alignment convention: every metric in this
    module goes through here so they cannot drift apart.
    """

    return (
        _centered_mesh(predicted_mesh, normalize_scale=normalize_scale),
        _centered_mesh(target_mesh, normalize_scale=normalize_scale),
    )


def _voxel_indices(mesh: Any, pitch: float) -> IntArray:
    """Return the unique occupied cell indices as an ``[N, 3]`` array."""

    grid = mesh.voxelized(pitch)
    try:
        grid = grid.fill()
    except Exception:
        # Some trimesh voxel backends cannot fill a non-watertight mesh. Surface
        # occupancy is still a deterministic fallback for the standalone metric.
        pass
    points = np.asarray(grid.points, dtype=np.float64)
    if points.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    indices = np.floor(points / pitch + 1e-6).astype(np.int64)
    return np.unique(indices, axis=0)


def _cell_set_iou(predicted_cells: IntArray, target_cells: IntArray) -> float:
    """Return the IoU of two sets of unique integer cell indices.

    The three axes are packed into one integer key so the set operations run in
    numpy. Holding the cells as Python tuples instead costs roughly 200 bytes
    each, which at the default resolution is two orders of magnitude more memory
    than the grids themselves. The packing cannot overflow here: both meshes are
    normalized before voxelization, so each axis spans at most ``resolution + 2``
    cells and a resolution large enough to overflow an int64 key could not be
    voxelized in the first place.
    """

    if predicted_cells.size == 0 or target_cells.size == 0:
        return 0.0
    low = np.minimum(predicted_cells.min(axis=0), target_cells.min(axis=0))
    span = np.maximum(predicted_cells.max(axis=0), target_cells.max(axis=0)) - low + 1
    strides = np.array([span[1] * span[2], span[2], 1], dtype=np.int64)
    predicted_keys = ((predicted_cells - low) * strides).sum(axis=1)
    target_keys = ((target_cells - low) * strides).sum(axis=1)
    intersection = int(
        np.intersect1d(predicted_keys, target_keys, assume_unique=True).size
    )
    union = predicted_keys.size + target_keys.size - intersection
    return float(intersection / union) if union else 0.0


def normalized_voxel_iou(
    predicted_mesh: Any,
    target_mesh: Any,
    *,
    resolution: int = 64,
) -> float:
    """Compute voxel IoU over centred, unit-box-normalized meshes.

    Both meshes go through :func:`align_meshes`, so each spans at most one unit
    along its longest axis and the pitch is exactly ``1 / resolution``. The
    occupied-cell count is therefore bounded by ``(resolution + 1) ** 3`` per
    mesh no matter how large the prediction is in its own units -- the metric
    cannot be made to allocate an unbounded grid by a bad prediction.

    Size is deliberately discarded: the drawings carry no dimension information,
    so a prediction can only be scored on shape. Use the millimetre bounding-box
    errors alongside this to see absolute-scale drift.
    """

    if (
        isinstance(resolution, bool)
        or not isinstance(resolution, int)
        or resolution <= 0
    ):
        raise ValueError(f"resolution must be a positive integer, got {resolution!r}")
    pred, gt = align_meshes(predicted_mesh, target_mesh, normalize_scale=True)
    pitch = 1.0 / resolution
    return _cell_set_iou(_voxel_indices(pred, pitch), _voxel_indices(gt, pitch))


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
        result[start : start + chunk_size] = np.min(
            np.sum(delta * delta, axis=-1), axis=1
        )
    return result


def surface_chamfer_distance(
    predicted_mesh: Any,
    target_mesh: Any,
    *,
    num_points: int = 8192,
    seed: int = 0,
    normalize_scale: bool = True,
) -> float:
    """Sample mesh surfaces and compute a centre-aligned Chamfer distance.

    Normalized by default, matching the voxel IoU. The absolute mode is kept
    because it costs nothing: both the surface sampling and the nearest-neighbour
    search are sized by ``num_points`` alone, so unlike voxelization this metric
    never allocates in proportion to the meshes' physical size. Its result is in
    squared millimetres and is only interpretable when the input carries scale.
    """

    if (
        isinstance(num_points, bool)
        or not isinstance(num_points, int)
        or num_points <= 0
    ):
        raise ValueError(f"num_points must be a positive integer, got {num_points!r}")
    import trimesh

    pred, gt = align_meshes(
        predicted_mesh, target_mesh, normalize_scale=normalize_scale
    )
    pred_points, _ = trimesh.sample.sample_surface(pred, num_points, seed=seed)
    gt_points, _ = trimesh.sample.sample_surface(gt, num_points, seed=seed)
    return symmetric_chamfer_distance(pred_points, gt_points)


def finite_float(value: object) -> float | None:
    """Return ``value`` as a finite float, or ``None`` when it is not one.

    Shared by the metric families that aggregate these rows: a column is
    missing (``None``), non-numeric, or NaN/inf for exactly the samples that
    could not be scored, and those must drop out of a mean rather than poison
    it.
    """

    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def mean_or_zero(values: Sequence[float]) -> float:
    """Mean of the scored values, or zero when nothing could be scored.

    Zero keeps the checkpoint monitor a real number even for an empty or
    fully-failed validation subset.
    """

    return float(np.mean(values)) if len(values) else 0.0


def median_or_zero(values: Sequence[float]) -> float:
    return float(np.median(values)) if len(values) else 0.0


__all__ = [
    "align_meshes",
    "bbox_dimension_error_mm",
    "bbox_dimension_error_relative",
    "max_bbox_error_mm",
    "max_bbox_error_relative",
    "normalized_voxel_iou",
    "surface_chamfer_distance",
    "symmetric_chamfer_distance",
    "finite_float",
    "mean_or_zero",
    "median_or_zero",
]
