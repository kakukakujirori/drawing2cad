"""CAD-Recode volume IoU using double-precision Manifold mesh booleans.

Reference: filaPro/cad-recode@03e3262119b38939feaa44b8368ad8db99243d47,
demo.ipynb. Multipart inputs represent their solid union, including cavities.
"""

from __future__ import annotations

import manifold3d as manifold
import numpy as np
import trimesh

from zeroshot.pipeline.verification.solid_mesh import require_solid, to_manifold

# Manifold Booleans on coincident faces misstate volumes by about 1e-4.
_IOU_SLACK = 1e-3


def manifold_iou(pred: manifold.Manifold, gt: manifold.Manifold) -> float:
    """Intersection volume / union volume, without voxelization."""
    require_solid(pred)
    require_solid(gt)
    intersection = pred ^ gt
    if intersection.status() != manifold.Error.NoError:
        raise ValueError(f"Manifold intersection: {intersection.status()}")
    overlap = intersection.volume()
    union = pred.volume() + gt.volume() - overlap
    value = overlap / union
    if not np.isfinite(value) or value < -_IOU_SLACK or value > 1 + _IOU_SLACK:
        raise ValueError(f"invalid mesh IoU: {value}")
    return float(np.clip(value, 0, 1))


def mesh_iou(pred: trimesh.Trimesh, gt: trimesh.Trimesh) -> float:
    """IoU of already aligned meshes. Does not normalize either input."""
    return manifold_iou(to_manifold(pred), to_manifold(gt))
