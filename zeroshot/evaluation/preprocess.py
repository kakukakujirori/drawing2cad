"""Centre both STEPs on their bboxes, scale both by reference_extent / longest GT
side, and take the maximum-mesh-IoU pose among the 24 cube rotations.

`mesh_tolerance` is BRepMesh's deflection relative to edge size.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise, permutations, product
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from zeroshot.evaluation.metrics.mesh_iou import (
    manifold_iou,
    require_solid,
    to_manifold,
)

# Hole area, relative to the solid's surface area, that counts as a zero-area face.
_SLIVER_AREA = 1e-9

# Poses this close to the best IoU are the same geometry; re-meshing a rotated
# solid moves its IoU by up to about 5e-4.
IOU_TIE_TOLERANCE = 1e-3

# The 24 proper rotations of a cube: signed axis permutations with det 1.
ORIENTATIONS = [
    matrix
    for axes in permutations(range(3))
    for signs in product((-1.0, 1.0), repeat=3)
    if np.linalg.det(matrix := np.diag(signs) @ np.eye(3)[list(axes)]) > 0
]


class EvaluationError(RuntimeError):
    """A GT, preprocessing, alignment or metric error, not a generation failure."""


class InvalidPredictionError(ValueError):
    """No usable predicted solid was generated."""


class AlignmentError(EvaluationError):
    """The required maximum-IoU pose could not be determined."""


@dataclass(frozen=True)
class PreprocessConfig:
    reference_extent: float = 1.8
    mesh_tolerance: float = 0.001
    angular_tolerance: float = 0.1

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class PreparedPair:
    pred_shape: Any
    gt_shape: Any
    pred_mesh: trimesh.Trimesh
    gt_mesh: trimesh.Trimesh
    iou: float
    rotation_index: int
    # The prediction in every pose whose IoU ties the best, best first.
    tied_shapes: dict[int, Any]
    metadata: dict[str, Any]


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


def read_solid(step_path: str | Path, *, prediction: bool = False) -> Any:
    import cadquery as cq

    error_type = InvalidPredictionError if prediction else EvaluationError
    try:
        path = Path(step_path)
        if not path.is_file():
            raise ValueError(f"STEP not found: {path}")
        shapes = cq.importers.importStep(str(path)).vals()
        shape = shapes[0] if len(shapes) == 1 else cq.Compound.makeCompound(shapes)
        if not shape.isValid() or not shape.Solids():
            raise ValueError("STEP is not a valid solid B-Rep")
        # Compound.Volume() raises StopIteration on some nested STEP compounds.
        volume = sum(solid.Volume() for solid in shape.Solids())
        if not np.isfinite(volume) or volume <= 0:
            raise ValueError("STEP has no positive finite volume")
        shape_bbox(shape)
        return shape
    except Exception as error:
        raise error_type(f"{step_path}: {error}") from error


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


def _solid_mesh(shape: Any, config: PreprocessConfig) -> Any:
    import manifold3d as manifold

    parts = []
    for solid in shape.Solids():
        vertices, faces = solid.tessellate(
            config.mesh_tolerance, config.angular_tolerance
        )
        mesh = trimesh.Trimesh(
            vertices=[vertex.toTuple() for vertex in vertices],
            faces=faces,
            process=True,
        )
        parts.append(to_manifold(close_sliver_holes(mesh)))
    # Unlike summing component volumes, union does not double-count overlaps.
    combined = manifold.Manifold.batch_boolean(parts, manifold.OpType.Add)
    require_solid(combined)
    return combined


def _to_trimesh(solid: Any) -> trimesh.Trimesh:
    mesh = solid.to_mesh64()
    return trimesh.Trimesh(
        vertices=np.asarray(mesh.vert_properties[:, :3]),
        faces=np.asarray(mesh.tri_verts),
        process=False,
    )


def orientation_ious(pred: Any, gt: Any) -> list[float]:
    """Mesh IoU of each cube rotation of ``pred``; fail closed, never fall back to CD."""
    try:
        return [
            manifold_iou(pred.transform(np.column_stack((rotation, np.zeros(3)))), gt)
            for rotation in ORIENTATIONS
        ]
    except Exception as error:
        raise AlignmentError(f"24-pose mesh-IoU alignment failed: {error}") from error


def prepare_pair(
    pred_step: str | Path,
    gt_step: str | Path,
    config: PreprocessConfig | None = None,
) -> PreparedPair:
    """Read, centre, jointly scale and align. Invalid pred and evaluator errors differ."""
    config = config or PreprocessConfig()
    gt = read_solid(gt_step)
    pred = read_solid(pred_step, prediction=True)
    try:
        gt_centre, gt_extents = shape_bbox(gt)
        pred_centre, pred_extents = shape_bbox(pred)
        scale = float(config.reference_extent / max(gt_extents))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("shared GT scale is not finite and positive")
        gt = transform_shape(gt, scale * np.eye(3), -scale * gt_centre)
        pred = transform_shape(pred, scale * np.eye(3), -scale * pred_centre)
        gt_native = _solid_mesh(gt, config)
        pred_native = _solid_mesh(pred, config)
        scores = orientation_ious(pred_native, gt_native)
        index = int(np.argmax(scores))
        tied = [index] + [
            i
            for i, score in enumerate(scores)
            if i != index and score >= scores[index] - IOU_TIE_TOLERANCE
        ]
        tied_shapes = {
            i: transform_shape(pred, ORIENTATIONS[i], np.zeros(3)) for i in tied
        }
        rotation = ORIENTATIONS[index]
        pred_native = pred_native.transform(np.column_stack((rotation, np.zeros(3))))
        return PreparedPair(
            pred_shape=tied_shapes[index],
            gt_shape=gt,
            pred_mesh=_to_trimesh(pred_native),
            gt_mesh=_to_trimesh(gt_native),
            iou=scores[index],
            rotation_index=index,
            tied_shapes=tied_shapes,
            metadata={
                "reference_extent": config.reference_extent,
                "shared_scale": scale,
                "gt_bbox_extents_mm": gt_extents.tolist(),
                "pred_bbox_extents_mm": pred_extents.tolist(),
                "gt_bbox_centre_mm": gt_centre.tolist(),
                "pred_bbox_centre_mm": pred_centre.tolist(),
                "rotation_index": index,
                "rotation": rotation.tolist(),
                "alignment_objective": "maximum_mesh_iou",
                "orientation_ious": scores,
                "tied_rotation_indices": tied,
            },
        )
    except EvaluationError:
        raise
    except Exception as error:
        raise EvaluationError(f"shared preprocessing failed: {error}") from error
