"""Squared Chamfer as in CAD-Recode (sum of two directional means) and squared Hausdorff.

`chamfer` is in reference_extent units, while CAD-Recode normalizes the longest
side to 1. `chamfer_diag` uses a unit GT bbox diagonal, as IterCAD does.
"""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from zeroshot.evaluation.preprocess import PreparedPair

DISTANCE_METRICS = frozenset({"chamfer", "hausdorff", "chamfer_diag"})


def point_distances(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    clouds = [np.asarray(points, dtype=np.float64) for points in (pred, gt)]
    for cloud in clouds:
        if cloud.ndim != 2 or cloud.shape[1] != 3 or not len(cloud):
            raise ValueError("point clouds must be nonempty (N, 3) arrays")
        if not np.isfinite(cloud).all():
            raise ValueError("point clouds must contain finite coordinates")
    pred, gt = clouds
    p_to_g = np.square(cKDTree(gt).query(pred, k=1)[0])
    g_to_p = np.square(cKDTree(pred).query(gt, k=1)[0])
    if not np.isfinite(p_to_g).all() or not np.isfinite(g_to_p).all():
        raise ValueError("nearest-neighbour squared distances are nonfinite")
    return {
        "chamfer": float(p_to_g.mean() + g_to_p.mean()),
        "hausdorff": float(max(p_to_g.max(), g_to_p.max())),
    }


def score_surface_distance(
    pair: PreparedPair, *, sample_points: int = 8192, seed: int = 0
) -> dict[str, float]:
    """Seed each side separately, so identical meshes still sample differently."""
    if (
        isinstance(sample_points, bool)
        or not isinstance(sample_points, int)
        or sample_points < 1
    ):
        raise ValueError("sample_points must be a positive integer")
    pred, _ = trimesh.sample.sample_surface(pair.pred_mesh, sample_points, seed=seed)
    gt, _ = trimesh.sample.sample_surface(pair.gt_mesh, sample_points, seed=seed + 1)
    scores = point_distances(pred, gt)
    gt_diagonal2 = float(np.sum(np.square(pair.gt_mesh.extents)))
    return {
        **scores,
        "chamfer_diag": scores["chamfer"] / gt_diagonal2,
    }
