"""Turn a predicted solid to face the way the ground truth does, as the
challenge does before it measures anything.

Orientation only: the metric families recentre and rescale each side themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
from pathlib import Path
from typing import Any

import numpy as np

from zeroshot.evaluation.load_canonical_mesh import centre_and_extent, load_step_mesh

_SAMPLE_POINTS = 4096


def cube_orientations() -> list[np.ndarray]:
    """The 24 rotations that map a cube onto itself."""

    # Signed permutations of the axes number 48; `det == 1` drops the half that
    # are reflections, which no rigid motion can reach.
    return [
        matrix
        for permutation in permutations(range(3))
        for signs in product((-1, 1), repeat=3)
        if round(
            float(np.linalg.det(matrix := _signed_permutation(permutation, signs)))
        )
        == 1
    ]


def _signed_permutation(
    permutation: tuple[int, ...], signs: tuple[int, ...]
) -> np.ndarray:
    matrix = np.zeros((3, 3), dtype=np.float64)
    for row, axis in enumerate(permutation):
        matrix[row, axis] = signs[row]
    return matrix


ORIENTATIONS = cube_orientations()


def _chamfer(a: np.ndarray, b: np.ndarray) -> float:
    """What the search below minimises: how far apart two surfaces sit."""

    from scipy.spatial import cKDTree

    # Not the forked `chamfer`, which reports 0.0 here: an empty cloud would
    # then win every minimum.
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    return float((cKDTree(b).query(a)[0].mean() + cKDTree(a).query(b)[0].mean()) / 2.0)


# Two orientations this close apart are not distinguishable by a surface
# sample of this size; the band is reported, never used to choose.
_TIE_RATIO = 1.05


@dataclass(frozen=True)
class Alignment:
    """Which orientation was chosen, how close it came, and how clear the win.

    `tied` counts the orientations that came out indistinguishable; above one,
    the pose is a draw the sampling settled rather than a measurement.
    """

    rotation_index: int
    chamfer: float
    tied: int = 1

    @property
    def rotation(self) -> np.ndarray:
        return ORIENTATIONS[self.rotation_index]


def best_orientation(
    predicted_points: np.ndarray,
    target_points: np.ndarray,
    predicted_centre: np.ndarray,
    target_centre: np.ndarray,
    scale: float,
) -> Alignment:
    """Pick the orientation whose Chamfer distance to the target is smallest.

    A solid of revolution ties about its axis, so sampling noise picks the
    winner; the challenge chooses that way too, and `tied` reports it.
    """

    # Centre and scale first, so the comparison is about pose alone.
    predicted = (predicted_points - predicted_centre) * scale
    target = target_points - target_centre
    distances = np.array(
        [_chamfer((rotation @ predicted.T).T, target) for rotation in ORIENTATIONS]
    )
    index = int(np.argmin(distances))
    best = float(distances[index])
    return Alignment(
        rotation_index=index,
        chamfer=best,
        tied=int((distances <= best * _TIE_RATIO).sum()),
    )


def _is_rotation(matrix: np.ndarray) -> bool:
    matrix = np.asarray(matrix, dtype=np.float64)
    return (
        matrix.shape == (3, 3)
        and bool(np.allclose(matrix @ matrix.T, np.eye(3)))
        and bool(np.isclose(float(np.linalg.det(matrix)), 1.0))
    )


def _sampled(mesh: Any, count: int, seed: int) -> np.ndarray:
    import trimesh

    points, _ = trimesh.sample.sample_surface(mesh, count, seed=seed)
    return np.asarray(points, dtype=np.float64)


def align_step(
    pred_step: Path,
    gt_step: Path,
    output_step: Path,
    sample_points: int = _SAMPLE_POINTS,
    seed: int = 0,
) -> Alignment:
    """Write `pred_step` rotated into the target's pose, and say which pose."""

    predicted_mesh = load_step_mesh(pred_step)
    target_mesh = load_step_mesh(gt_step)
    predicted_centre, predicted_extent = centre_and_extent(predicted_mesh)
    target_centre, target_extent = centre_and_extent(target_mesh)

    alignment = best_orientation(
        _sampled(predicted_mesh, sample_points, seed + 1),
        _sampled(target_mesh, sample_points, seed),
        predicted_centre,
        target_centre,
        target_extent / predicted_extent,
    )
    write_rotated_step(pred_step, output_step, alignment.rotation)
    return alignment


def write_rotated_step(
    source_step: Path, output_step: Path, rotation: np.ndarray
) -> None:
    """Rotate a STEP about the origin, keeping every surface its own kind."""

    import cadquery as cq

    # OCP, not the OCC.Core bindings the mesh loader uses: they wrap the same
    # C++ classes in objects CadQuery will not accept.
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Trsf

    # `gp_Trsf.SetValues` coerces instead of refusing: a shear becomes the
    # identity, and `diag(2, 1, 1)` a uniform scale of 1.26.
    if not _is_rotation(rotation):
        raise ValueError(f"not a rotation:\n{rotation}")

    transform = gp_Trsf()
    transform.SetValues(
        *(
            value
            for row in range(3)
            for value in (*(float(item) for item in rotation[row]), 0.0)
        )
    )
    shape = cq.importers.importStep(str(source_step)).val()
    rotated = cq.Shape(BRepBuilderAPI_Transform(shape.wrapped, transform, True).Shape())

    output_step.parent.mkdir(parents=True, exist_ok=True)
    cq.exporters.export(rotated, str(output_step), exportType="STEP")


__all__ = [
    "ORIENTATIONS",
    "Alignment",
    "align_step",
    "best_orientation",
    "cube_orientations",
    "write_rotated_step",
]
