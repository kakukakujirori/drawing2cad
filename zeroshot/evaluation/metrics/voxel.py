"""Shape-only voxel IoU between a predicted and a ground-truth STEP."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np

# Passed with OCC's `isRelative` flag, which scales the chord error by each
# edge's own size; that is what makes solids stored in different units mesh
# alike. Do not pre-multiply it by the part's size as well.
_LINEAR_DEFLECTION = 0.01
_ANGULAR_DEFLECTION = 0.1


def _load_mesh(step_path: Path) -> Any:
    """Triangulate a STEP into one mesh, in the file's own units."""

    import trimesh
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.StlAPI import StlAPI_Writer

    reader = STEPControl_Reader()
    if reader.ReadFile(str(step_path)) != IFSelect_RetDone:
        raise ValueError(f"failed to read STEP file {step_path}")
    reader.TransferRoots()
    shape = reader.OneShape()

    BRepMesh_IncrementalMesh(shape, _LINEAR_DEFLECTION, True, _ANGULAR_DEFLECTION, True)
    with tempfile.NamedTemporaryFile(suffix=".stl") as stl_file:
        writer = StlAPI_Writer()
        writer.SetASCIIMode(False)
        writer.Write(shape, stl_file.name)
        return trimesh.load_mesh(stl_file.name)


def _normalized(mesh: Any) -> Any:
    """Centre a mesh on its bounding box and scale its longest side to 1."""

    aligned = mesh.copy()
    bounds = np.asarray(aligned.bounds, dtype=np.float64)
    aligned.apply_translation(-(bounds[0] + bounds[1]) / 2.0)
    extent = float(np.max(aligned.extents))
    if extent <= 1e-12:
        raise ValueError("cannot normalize a mesh with zero maximum extent")
    aligned.apply_scale(1.0 / extent)
    return aligned


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
            _occupied_cells(_normalized(_load_mesh(pred_step)), pitch),
            _occupied_cells(_normalized(_load_mesh(gt_step)), pitch),
        )
    }


__all__ = ["score_voxel"]
