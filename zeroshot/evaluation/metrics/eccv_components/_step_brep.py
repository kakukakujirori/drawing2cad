"""Labeled B-Rep point clouds and incidence matrices sampled from a STEP file.

Ported from the ECCV 2026 CAD Challenge's own evaluator, bundled with the
challenge data at ``data/eccv2026-cad-challenge-data/examples/min_eval/eval.py``
(see ``data/eccv2026-cad-challenge-data/ACKNOWLEDGEMENTS_AND_LICENSES.md`` for
its terms). The sampling densities, meshing deflections and entity caps are the
official values and must not be tuned: they define the metric, because the
leaderboard's F1 threshold is an absolute distance against clouds sampled this
way.

The one deliberate deviation is that surface sampling is seeded. The official
script leaves it to the global random state, which makes its own smoke test
report slightly different numbers on every run; a validation metric watched
across training steps has to be reproducible.

This module imports pythonocc (``OCC.Core``), so it must only ever be imported
inside the isolated scoring subprocess.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Meshing parameters for STEP triangulation (official values).
STEP_LINEAR_DEFLECTION = 0.01
STEP_ANGULAR_DEFLECTION = 0.1
# Surface samples per unit area, and edge samples per unit length.
SAMPLES_PER_AREA = 1000
SAMPLES_PER_LENGTH = 100
# Entity caps: beyond these the pairwise assignment below stops being tractable.
MAX_FACES = 5000
MAX_EDGES = 10000
MAX_VERTS = 10000
# Both densities are per unit length/area, so the point count grows with the
# square of a part's physical size. That is harmless for the challenge's own
# targets (always 1.8 units long) but not for a solid in millimetres or a
# wildly oversized prediction, which would otherwise sample hundreds of
# millions of points. Scoring such a part is refused rather than silently
# thinned out, because thinning would change the metric.
MAX_FACE_SAMPLES = 4_000_000


@dataclass(frozen=True)
class StepBRep:
    """Sampled boundary representation of one STEP file."""

    face_pc: np.ndarray
    face_labels: np.ndarray
    edge_pc: np.ndarray
    edge_labels: np.ndarray
    vertex_pc: np.ndarray
    vertex_labels: np.ndarray
    fe_matrix: np.ndarray
    ev_matrix: np.ndarray
    n_faces: int
    n_edges: int
    n_verts: int


def _face_triangles(face) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(vertices, triangles)`` of a meshed face, or ``None``."""

    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopLoc import TopLoc_Location

    location = TopLoc_Location()
    triangulation = BRep_Tool.Triangulation(face, location)
    if triangulation is None:
        return None
    transformation = location.Transformation()
    node_count = triangulation.NbNodes()
    vertices = np.empty((node_count, 3), dtype=np.float64)
    for index in range(1, node_count + 1):
        point = triangulation.Node(index).Transformed(transformation)
        vertices[index - 1] = (point.X(), point.Y(), point.Z())
    triangle_count = triangulation.NbTriangles()
    triangles = np.empty((triangle_count, 3), dtype=np.int64)
    for index in range(1, triangle_count + 1):
        first, second, third = triangulation.Triangle(index).Get()
        triangles[index - 1] = (first - 1, second - 1, third - 1)
    return vertices, triangles


def _sample_edge_points(edge) -> np.ndarray | None:
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.GCPnts import GCPnts_AbscissaPoint, GCPnts_UniformAbscissa

    try:
        curve = BRepAdaptor_Curve(edge)
        length = GCPnts_AbscissaPoint.Length(curve)
    except Exception:  # noqa: BLE001
        return None
    if not np.isfinite(length) or length <= 0:
        return None
    count = max(2, int(length * SAMPLES_PER_LENGTH))
    abscissa = GCPnts_UniformAbscissa(curve, count)
    if not abscissa.IsDone() or abscissa.NbPoints() < 1:
        return None
    points = np.empty((abscissa.NbPoints(), 3), dtype=np.float64)
    for index in range(1, abscissa.NbPoints() + 1):
        point = curve.Value(abscissa.Parameter(index))
        points[index - 1] = (point.X(), point.Y(), point.Z())
    return points


def _read_step(step_path: str | Path):
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.STEPControl import STEPControl_Reader

    step_path = Path(step_path)
    if not step_path.is_file():
        raise ValueError(f"File {step_path} does not exist.")
    reader = STEPControl_Reader()
    if reader.ReadFile(str(step_path)) != IFSelect_RetDone:
        raise ValueError(f"Failed to read STEP file {step_path}")
    reader.TransferRoots()
    return reader.OneShape()


def load_step_brep(
    step_path: str | Path,
    *,
    seed: int = 0,
) -> StepBRep:
    """Triangulate a STEP file and sample per-entity labeled point clouds.

    The densities are per unit of the file, so it must already be normalized
    the way the challenge's targets are.
    """

    import trimesh
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_VERTEX
    from OCC.Core.TopExp import TopExp_Explorer, topexp
    from OCC.Core.TopoDS import topods
    from OCC.Core.TopTools import TopTools_IndexedMapOfShape

    step_path = Path(step_path)
    shape = _read_step(step_path)
    BRepMesh_IncrementalMesh(
        shape,
        STEP_LINEAR_DEFLECTION,
        True,
        STEP_ANGULAR_DEFLECTION,
        True,
    )

    face_map = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, TopAbs_FACE, face_map)
    edge_map = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, TopAbs_EDGE, edge_map)
    vertex_map = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, TopAbs_VERTEX, vertex_map)

    n_faces, n_edges, n_verts = face_map.Size(), edge_map.Size(), vertex_map.Size()
    for count, cap, label in (
        (n_faces, MAX_FACES, "faces"),
        (n_edges, MAX_EDGES, "edges"),
        (n_verts, MAX_VERTS, "vertices"),
    ):
        if count > cap:
            raise ValueError(f"Too many {label} in {step_path}: {count} > {cap}")

    face_chunks = []
    sampled = 0
    for index in range(1, n_faces + 1):
        triangulated = _face_triangles(topods.Face(face_map.FindKey(index)))
        if triangulated is None:
            continue
        vertices, triangles = triangulated
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
        if mesh.area <= 0 or len(mesh.faces) == 0:
            continue
        count = max(1, int(mesh.area * SAMPLES_PER_AREA))
        sampled += count
        if sampled > MAX_FACE_SAMPLES:
            raise ValueError(
                f"{step_path} needs more than {MAX_FACE_SAMPLES} surface samples"
            )
        points, _ = trimesh.sample.sample_surface(mesh, count, seed=seed + index)
        labels = np.full((len(points), 1), index - 1, dtype=np.float64)
        face_chunks.append(np.concatenate([points, labels], axis=1))
    if not face_chunks:
        raise ValueError(f"No triangulated faces found in {step_path}")
    face_cloud = np.concatenate(face_chunks, axis=0)

    edge_chunks = []
    for index in range(1, n_edges + 1):
        edge = topods.Edge(edge_map.FindKey(index))
        if BRep_Tool.Degenerated(edge):
            continue
        points = _sample_edge_points(edge)
        if points is None or len(points) == 0:
            continue
        labels = np.full((len(points), 1), index - 1, dtype=np.float64)
        edge_chunks.append(np.concatenate([points, labels], axis=1))
    if edge_chunks:
        edge_cloud = np.concatenate(edge_chunks, axis=0)
        edge_pc = edge_cloud[:, :3].astype(np.float64)
        edge_labels = edge_cloud[:, 3].astype(np.int32)
    else:
        edge_pc = np.zeros((0, 3), dtype=np.float64)
        edge_labels = np.zeros((0,), dtype=np.int32)

    vertex_pc = np.empty((n_verts, 3), dtype=np.float64)
    for index in range(1, n_verts + 1):
        point = BRep_Tool.Pnt(topods.Vertex(vertex_map.FindKey(index)))
        vertex_pc[index - 1] = (point.X(), point.Y(), point.Z())

    fe_matrix = np.zeros((n_faces, n_edges), dtype=np.uint8)
    for index in range(1, n_faces + 1):
        explorer = TopExp_Explorer(topods.Face(face_map.FindKey(index)), TopAbs_EDGE)
        while explorer.More():
            edge_index = edge_map.FindIndex(explorer.Current())
            if edge_index > 0:
                fe_matrix[index - 1, edge_index - 1] = 1
            explorer.Next()

    ev_matrix = np.zeros((n_edges, n_verts), dtype=np.uint8)
    for index in range(1, n_edges + 1):
        explorer = TopExp_Explorer(topods.Edge(edge_map.FindKey(index)), TopAbs_VERTEX)
        while explorer.More():
            vertex_index = vertex_map.FindIndex(explorer.Current())
            if vertex_index > 0:
                ev_matrix[index - 1, vertex_index - 1] = 1
            explorer.Next()

    return StepBRep(
        face_pc=face_cloud[:, :3].astype(np.float64),
        face_labels=face_cloud[:, 3].astype(np.int32),
        edge_pc=edge_pc,
        edge_labels=edge_labels,
        vertex_pc=vertex_pc,
        vertex_labels=np.arange(n_verts, dtype=np.int32),
        fe_matrix=fe_matrix,
        ev_matrix=ev_matrix,
        n_faces=n_faces,
        n_edges=n_edges,
        n_verts=n_verts,
    )


__all__ = [
    "MAX_EDGES",
    "MAX_FACES",
    "MAX_VERTS",
    "SAMPLES_PER_AREA",
    "SAMPLES_PER_LENGTH",
    "STEP_ANGULAR_DEFLECTION",
    "STEP_LINEAR_DEFLECTION",
    "StepBRep",
    "load_step_brep",
]
