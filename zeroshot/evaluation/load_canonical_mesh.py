"""Read a STEP as a mesh, and put a mesh on a unit box at the origin.

Shared so that solids read for different purposes still triangulate alike.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np

# Passed with OCC's `isRelative` flag, which scales the chord error by each
# edge's own size; that is what makes solids stored in different units mesh
# alike. Do not pre-multiply it by the part's size as well.
LINEAR_DEFLECTION = 0.01
ANGULAR_DEFLECTION = 0.1


def load_step_mesh(step_path: Path) -> Any:
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

    BRepMesh_IncrementalMesh(shape, LINEAR_DEFLECTION, True, ANGULAR_DEFLECTION, True)
    with tempfile.NamedTemporaryFile(suffix=".stl") as stl_file:
        writer = StlAPI_Writer()
        writer.SetASCIIMode(False)
        writer.Write(shape, stl_file.name)
        return trimesh.load_mesh(stl_file.name)


def centre_and_extent(mesh: Any) -> tuple[np.ndarray, float]:
    """The centre of a mesh's bounding box, and its longest side."""

    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    centre = (bounds[0] + bounds[1]) / 2.0
    extent = float((bounds[1] - bounds[0]).max())
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError("mesh has no extent")
    return centre, extent


def normalized(mesh: Any) -> Any:
    """Centre a mesh on its bounding box and scale its longest side to 1."""

    centre, extent = centre_and_extent(mesh)
    canonical = mesh.copy()
    canonical.apply_translation(-centre)
    canonical.apply_scale(1.0 / extent)
    return canonical


__all__ = [
    "ANGULAR_DEFLECTION",
    "LINEAR_DEFLECTION",
    "centre_and_extent",
    "load_step_mesh",
    "normalized",
]
