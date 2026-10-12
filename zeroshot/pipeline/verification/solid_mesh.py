"""Mesh a solid as the scorer does; the pipeline checks each build with the same code.

`mesh_tolerance` is BRepMesh's deflection relative to edge size.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import manifold3d as manifold
import numpy as np
import trimesh

# Hole area, relative to the solid's surface area, that counts as a zero-area face.
_SLIVER_AREA = 1e-9


@dataclass(frozen=True)
class PreprocessConfig:
    reference_extent: float = 1.8
    mesh_tolerance: float = 0.001
    angular_tolerance: float = 0.1

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


def shape_bbox(shape: Any) -> tuple[np.ndarray, np.ndarray]:
    """Tight analytic bbox, without triangulation or tolerance padding."""
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    box = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape.wrapped, box, False, False)
    bounds = np.asarray(box.Get(), dtype=np.float64).reshape(2, 3)
    extents = bounds[1] - bounds[0]
    if not np.isfinite(bounds).all() or np.max(extents) <= 0:
        raise ValueError("STEP has no finite positive bbox extent")
    return bounds.mean(axis=0), extents


def transform_shape(shape: Any, matrix: np.ndarray, translation: np.ndarray) -> Any:
    """Uniform similarity via gp_Trsf preserves analytic B-Rep surface types."""
    import cadquery as cq
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Trsf

    affine = np.column_stack((matrix, translation))
    transform = gp_Trsf()
    transform.SetValues(*affine.ravel().tolist())
    return cq.Shape.cast(
        BRepBuilderAPI_Transform(shape.wrapped, transform, True).Shape()
    )


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


def close_sliver_holes(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Fan-fill zero-area holes left by faces that BRepMesh cannot triangulate.

    Other holes stay open for Manifold to reject.
    """
    half_edges = mesh.faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    # Collapsed triangles add self-loops, which Manifold drops anyway.
    half_edges = half_edges[half_edges[:, 0] != half_edges[:, 1]]
    open_edges = half_edges[
        trimesh.grouping.group_rows(np.sort(half_edges, axis=1), require_count=1)
    ]
    # A patch traverses each open half-edge backwards.
    following = dict(zip(open_edges[:, 1].tolist(), open_edges[:, 0].tolist()))
    if len(following) != len(open_edges) or set(following) != set(following.values()):
        return mesh
    patches = []
    while following:
        loop = [next(iter(following))]
        while (vertex := following.pop(loop[-1])) != loop[0]:
            loop.append(vertex)
        fan = np.array([(loop[0], a, b) for a, b in pairwise(loop[1:])])
        if trimesh.triangles.area(mesh.vertices[fan]).sum() <= _SLIVER_AREA * mesh.area:
            patches.append(fan)
    if not patches:
        return mesh
    return trimesh.Trimesh(
        mesh.vertices, np.vstack([mesh.faces, *patches]), process=False
    )


def _triangles(solid: Any, config: PreprocessConfig) -> trimesh.Trimesh:
    vertices, faces = solid.tessellate(config.mesh_tolerance, config.angular_tolerance)
    mesh = trimesh.Trimesh(
        vertices=[vertex.toTuple() for vertex in vertices],
        faces=faces,
        process=True,
    )
    return close_sliver_holes(mesh)


def solid_mesh(shape: Any, config: PreprocessConfig) -> manifold.Manifold:
    parts = [to_manifold(_triangles(solid, config)) for solid in shape.Solids()]
    # Unlike summing component volumes, union does not double-count overlaps.
    combined = manifold.Manifold.batch_boolean(parts, manifold.OpType.Add)
    require_solid(combined)
    return combined


# ---------------------------------------------------------------------------
# Build check: whether the scorer can mesh a build, and where it cannot.
# ---------------------------------------------------------------------------


class UnjoinedMeshError(ValueError):
    """Manifold rejected the mesh; `regions` box, in mm, where its surface fails to join."""

    def __init__(self, reason: str, regions: list[tuple[np.ndarray, np.ndarray]]):
        super().__init__(reason)
        self.regions = regions


def _unjoined_regions(
    shape: Any, config: PreprocessConfig
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Box each connected run of edges not shared by two oppositely wound triangles."""
    regions = []
    for solid in shape.Solids():
        mesh = _triangles(solid, config)
        uses = Counter(map(tuple, mesh.edges.tolist()))
        unpaired = [
            edge
            for edge, count in uses.items()
            if edge[0] != edge[1] and (count != 1 or uses[edge[::-1]] != 1)
        ]
        if not unpaired:
            continue
        for run in trimesh.graph.connected_components(np.array(unpaired), min_len=1):
            points = mesh.vertices[run]
            regions.append((points.min(axis=0), points.max(axis=0)))
    return sorted(regions, key=lambda box: float(np.linalg.norm(box[1] - box[0])))


def require_closed_mesh(shape: Any) -> None:
    """Mesh `shape` as the scorer meshes a prediction; raise UnjoinedMeshError if it fails.

    Scaled by its own longest side, where the scorer uses the target's.
    """
    config = PreprocessConfig()
    centre, extents = shape_bbox(shape)
    scale = config.reference_extent / max(extents)
    scaled = transform_shape(shape, scale * np.eye(3), -scale * centre)
    try:
        solid_mesh(scaled, config)
    except ValueError as error:
        regions = [
            (low / scale + centre, high / scale + centre)
            for low, high in _unjoined_regions(scaled, config)
        ]
        raise UnjoinedMeshError(str(error), regions) from error
