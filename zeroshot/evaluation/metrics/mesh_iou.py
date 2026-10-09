"""CAD-Recode volume IoU using double-precision Manifold mesh booleans.

Reference: filaPro/cad-recode@03e3262119b38939feaa44b8368ad8db99243d47,
demo.ipynb. Multipart inputs represent their solid union, including cavities.
"""

from __future__ import annotations

import manifold3d as manifold
import numpy as np
import trimesh

# Manifold Booleans on coincident faces misstate volumes by about 1e-4.
_IOU_SLACK = 1e-3


def to_manifold(mesh: trimesh.Trimesh) -> manifold.Manifold:
    """Require a finite, nonempty solid; Manifold decides mesh validity."""
    if not len(mesh.vertices) or not len(mesh.faces):
        raise ValueError("empty triangle mesh")
    if not np.isfinite(mesh.vertices).all():
        raise ValueError("nonfinite mesh vertices")
    native = manifold.Manifold(
        manifold.Mesh64(
            vert_properties=np.ascontiguousarray(mesh.vertices, dtype=np.float64),
            tri_verts=np.ascontiguousarray(mesh.faces, dtype=np.uint64),
        )
    )
    require_solid(native)
    return native


def require_solid(solid: manifold.Manifold) -> None:
    if solid.status() != manifold.Error.NoError:
        raise ValueError(f"Manifold: {solid.status()}")
    volume = solid.volume()
    if solid.is_empty() or not np.isfinite(volume) or volume <= 0:
        raise ValueError("mesh has no positive finite solid volume")


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
