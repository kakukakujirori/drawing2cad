from pathlib import Path

import numpy as np
import pytest

from zeroshot.evaluation.align_orientation import (
    ORIENTATIONS,
    align_step,
    cube_orientations,
    write_rotated_step,
)

_QUARTER_TURNS = {
    "x": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
    "y": np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),
    "z": np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
}


@pytest.fixture(scope="module")
def lopsided_step(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A solid no rotation but the identity maps onto itself.

    The box in ``solids`` is symmetric under a half turn about each axis, which
    leaves the best orientation genuinely ambiguous; that is a real property of
    the metric but it cannot tell a working search from a broken one.
    """

    import cadquery as cq

    path = tmp_path_factory.mktemp("alignment") / "lopsided.step"
    cq.exporters.export(
        (
            cq.Workplane("XY")
            .box(30, 20, 10)
            .faces(">Z")
            .workplane()
            .pushPoints([(10, 5)])
            .hole(4)
            .faces(">X")
            .workplane()
            .pushPoints([(2, 1)])
            .hole(3)
            .edges("|Z and >X")
            .fillet(2)
        ),
        str(path),
        exportType="STEP",
    )
    return path


def _topology(step_path: Path) -> tuple[int, int, int]:
    import cadquery as cq

    shape = cq.importers.importStep(str(step_path)).val()
    return len(shape.Faces()), len(shape.Edges()), len(shape.Vertices())


def test_the_orientations_are_the_rotations_of_a_cube() -> None:
    """Signed permutations number 48; the reflections among them are not poses
    any solid can be put into."""
    orientations = cube_orientations()

    assert len(orientations) == 24
    assert all(round(float(np.linalg.det(r))) == 1 for r in orientations)
    assert any(np.allclose(r, np.eye(3)) for r in orientations)


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_a_rotated_prediction_is_put_back(
    axis: str, lopsided_step: Path, tmp_path: Path
) -> None:
    rotation = _QUARTER_TURNS[axis]
    rotated = tmp_path / f"rotated_{axis}.step"
    write_rotated_step(lopsided_step, rotated, rotation)

    alignment = align_step(rotated, lopsided_step, tmp_path / f"aligned_{axis}.step")

    assert np.allclose(alignment.rotation @ rotation, np.eye(3))


def test_a_prediction_already_in_pose_is_left_alone(
    lopsided_step: Path, tmp_path: Path
) -> None:
    alignment = align_step(lopsided_step, lopsided_step, tmp_path / "aligned.step")

    assert np.allclose(alignment.rotation, np.eye(3))
    assert np.allclose(ORIENTATIONS[alignment.rotation_index], alignment.rotation)


def test_a_solid_of_revolution_reports_its_pose_as_a_draw(tmp_path: Path) -> None:
    """Chamfer cannot see a seam, so it cannot place a turned part about its
    axis. The pose it picks is the sampling's, and any score that depends on
    pose is then a coin toss -- which the caller has to be able to notice."""
    import cadquery as cq

    turned = tmp_path / "turned.step"
    cq.exporters.export(
        cq.Workplane("XZ").circle(8).extrude(40).faces(">Y").hole(6),
        str(turned),
        exportType="STEP",
    )

    alignment = align_step(turned, turned, tmp_path / "aligned.step")

    assert alignment.tied > 1


def test_an_unambiguous_pose_reports_no_draw(
    lopsided_step: Path, tmp_path: Path
) -> None:
    alignment = align_step(lopsided_step, lopsided_step, tmp_path / "aligned.step")

    assert alignment.tied == 1


def test_the_chosen_pose_does_not_move_with_the_seed(
    lopsided_step: Path, tmp_path: Path
) -> None:
    """The sample count has to leave the winner clear of the sampling noise.

    At 4096 it did not: half this dataset's samples changed pose between seeds,
    and on one of them the two poses were worth 0.18 and 0.67 mean F1.
    """
    rotated = tmp_path / "rotated.step"
    write_rotated_step(lopsided_step, rotated, _QUARTER_TURNS["y"])

    chosen = {
        align_step(
            rotated, lopsided_step, tmp_path / f"aligned{seed}.step", seed=seed
        ).rotation_index
        for seed in range(3)
    }

    assert len(chosen) == 1


def test_rotating_keeps_every_surface_its_own_kind(
    lopsided_step: Path, tmp_path: Path
) -> None:
    """The metrics match faces, edges and vertices one to one.

    A transform carrying a scale makes OCC rebuild each face as a NURBS patch,
    which changes those counts and would be scored as a different part.
    """
    rotated = tmp_path / "rotated.step"
    write_rotated_step(lopsided_step, rotated, _QUARTER_TURNS["x"])

    assert _topology(rotated) == _topology(lopsided_step)


def test_a_non_rigid_transform_is_refused(lopsided_step: Path, tmp_path: Path) -> None:
    """The guarantee above is enforced by OCC, not by convention here."""
    with pytest.raises(ValueError, match="not a rotation"):
        write_rotated_step(
            lopsided_step, tmp_path / "scaled.step", np.diag([2.0, 1.0, 1.0])
        )
