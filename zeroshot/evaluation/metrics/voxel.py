"""Shape-only voxel IoU between a predicted and a ground-truth STEP."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from zeroshot.evaluation.load_canonical_mesh import load_step_mesh, normalized


def _occupied_cells(mesh: Any, pitch: float) -> np.ndarray:
    """Return the unique filled cell indices, on a lattice shared by all meshes.

    Taken from world-space cell centres rather than the grid's own
    ``sparse_indices``, which are relative to each mesh's own bounds and so are
    not comparable between two meshes.
    """

    points = np.asarray(mesh.voxelized(pitch).fill().points, dtype=np.float64)
    if points.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    return np.unique(np.floor(points / pitch + 1e-6).astype(np.int64), axis=0)


def _cell_iou(predicted_cells: np.ndarray, target_cells: np.ndarray) -> float:
    """IoU of two sets of unique integer cell indices."""

    if predicted_cells.size == 0 or target_cells.size == 0:
        return 0.0
    union = len(np.unique(np.concatenate([predicted_cells, target_cells]), axis=0))
    intersection = len(predicted_cells) + len(target_cells) - union
    return intersection / union


def score_voxel(
    pred_step: Path,
    gt_step: Path,
    resolution: int = 64,
) -> dict[str, Any]:
    """Return the voxel IoU over centred, unit-box-normalized meshes."""

    if isinstance(resolution, bool) or not isinstance(resolution, int):
        raise TypeError(f"resolution must be an integer, got {resolution!r}")
    if resolution <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}")

    pitch = 1.0 / resolution
    return {
        "voxel_iou": _cell_iou(
            _occupied_cells(normalized(load_step_mesh(pred_step)), pitch),
            _occupied_cells(normalized(load_step_mesh(gt_step)), pitch),
        )
    }


__all__ = ["score_voxel"]
