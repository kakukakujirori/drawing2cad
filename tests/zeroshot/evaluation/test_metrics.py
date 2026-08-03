from pathlib import Path

import pytest

from zeroshot.evaluation.metrics import score_eccv, score_voxel
from zeroshot.evaluation.metrics.eccv_components import reference_frame


def test_voxel_self_identity_is_exact(solids: dict[str, Path]) -> None:
    assert score_voxel(solids["box"], solids["box"])["voxel_iou"] == 1.0


def test_voxel_ignores_scale(solids: dict[str, Path]) -> None:
    """The chord error reaches OCC with its own ``isRelative`` flag set.

    Take that away, or scale it by the part's size a second time, and the
    triangulation -- and so the IoU -- depends on the units the file uses.
    """

    assert score_voxel(solids["sphere"], solids["sphere10"])["voxel_iou"] == 1.0


def test_voxel_separates_shapes(solids: dict[str, Path]) -> None:
    assert 0.0 < score_voxel(solids["box"], solids["sphere"])["voxel_iou"] < 0.5


def test_voxel_returns_scalars(solids: dict[str, Path]) -> None:
    row = score_voxel(solids["box"], solids["box"])
    assert all(key.startswith("voxel_") for key in row)
    assert all(isinstance(value, (int, float)) for value in row.values())


@pytest.mark.parametrize(
    "resolution, expected",
    [(0, ValueError), (-1, ValueError), (True, TypeError), (1.5, TypeError)],
)
def test_voxel_rejects_a_bad_resolution(
    solids: dict[str, Path],
    resolution: object,
    expected: type[Exception],
) -> None:
    with pytest.raises(expected):
        score_voxel(solids["box"], solids["box"], resolution)  # type: ignore[arg-type]


def test_eccv_self_identity(solids: dict[str, Path]) -> None:
    row = score_eccv(solids["box"], solids["box"])
    assert row["eccv_surface_f1"] == pytest.approx(1.0, abs=1e-3)
    assert row["eccv_edge_f1"] == pytest.approx(1.0, abs=1e-3)
    assert row["eccv_vertex_f1"] == pytest.approx(1.0, abs=1e-3)


def test_eccv_is_deterministic(solids: dict[str, Path]) -> None:
    assert score_eccv(solids["box"], solids["fillet"]) == score_eccv(
        solids["box"], solids["fillet"]
    )


def test_eccv_separates_shapes(solids: dict[str, Path]) -> None:
    assert score_eccv(solids["box"], solids["sphere"])["eccv_surface_f1"] == 0.0


def test_eccv_scores_a_partial_match_between(solids: dict[str, Path]) -> None:
    assert 0.0 < score_eccv(solids["box"], solids["fillet"])["eccv_surface_f1"] < 1.0


def test_eccv_returns_prefixed_scalars(solids: dict[str, Path]) -> None:
    row = score_eccv(solids["box"], solids["box"])
    assert all(key.startswith("eccv_") for key in row)
    assert all(isinstance(value, (int, float)) for value in row.values())


def test_eccv_rejects_an_unreadable_step(
    solids: dict[str, Path], junk_step: Path
) -> None:
    with pytest.raises(ValueError):
        score_eccv(junk_step, solids["box"])


def test_reference_frame_is_not_fooled_by_a_control_hull(
    solids: dict[str, Path],
) -> None:
    """A frame built from an untriangulated ``Add`` shrinks the whole solid.

    The absolute match threshold then covers a far larger share of the part,
    and unrelated shapes score as matches.
    """

    assert reference_frame(solids["loft"], 1.8).scale == pytest.approx(
        1.8 / 20.0, rel=1e-3
    )
