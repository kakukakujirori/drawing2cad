import pickle
from pathlib import Path

import numpy as np
import pytest

from zeroshot.evaluation import align_orientation, normalize_brep, run_scoring
from zeroshot.evaluation.metrics import score_eccv
from zeroshot.evaluation.run_scoring import (
    ScoreStatus,
    StepScorer,
    latest_verified_step,
    score_run,
)


def test_scores_a_valid_pair(solids: dict[str, Path]) -> None:
    report = StepScorer().score(solids["box"], solids["box"])
    assert report.status is ScoreStatus.OK
    assert report.errors == {}
    assert report.metrics["voxel_iou"] == 1.0


def test_every_column_is_prefixed_by_its_family(solids: dict[str, Path]) -> None:
    for name, call in StepScorer().families().items():
        columns = call(solids["box"], solids["box"])
        assert columns, f"{name} produced no columns"
        assert all(column.startswith(f"{name}_") for column in columns)


def test_a_family_returns_its_most_telling_column_first(
    solids: dict[str, Path],
) -> None:
    """Reports show one column per family and take the first one.

    Reordering these dicts is therefore not cosmetic: it moves what the run
    table leads with.
    """
    leading = {
        name: next(iter(call(solids["box"], solids["box"])))
        for name, call in StepScorer().families().items()
    }

    assert leading == {"eccv": "eccv_mean_f1", "voxel": "voxel_iou"}


def test_the_leading_column_is_the_mean_of_the_four_official_axes(
    solids: dict[str, Path],
) -> None:
    """It is the challenge's own per-sample score, so it has to be those four
    and not, say, the surface/edge/vertex three or the precision-weighted mix.
    """
    columns = StepScorer().families()["eccv"](solids["box"], solids["sphere"])
    axes = ("surface", "edge", "vertex", "topology")

    assert columns["eccv_mean_f1"] == pytest.approx(
        sum(columns[f"eccv_{axis}_f1"] for axis in axes) / 4
    )


def test_a_missing_prediction_is_reported(
    tmp_path: Path, solids: dict[str, Path]
) -> None:
    report = StepScorer().score(tmp_path / "absent.step", solids["box"])
    assert report.status is ScoreStatus.NO_PREDICTION
    assert report.metrics == {}


def test_a_missing_target_raises(tmp_path: Path, solids: dict[str, Path]) -> None:
    """A run outcome is reported; an operator's mistake is not."""

    with pytest.raises(FileNotFoundError):
        StepScorer().score(solids["box"], tmp_path / "absent.step")


def test_a_malformed_prediction_fails(solids: dict[str, Path], junk_step: Path) -> None:
    report = StepScorer().score(junk_step, solids["box"])
    assert report.status is ScoreStatus.FAILED
    # Alignment reads the same unreadable file, and says so under its own key.
    assert set(report.errors) == {"align", "eccv", "voxel"}
    assert report.metrics == {}


def test_a_timeout_is_not_a_metric_failure(solids: dict[str, Path]) -> None:
    report = StepScorer(timeout_s=0.01).score(solids["box"], solids["box"])
    assert report.status is ScoreStatus.TIMEOUT
    assert set(report.errors) == {"scorer"}


def test_one_failing_family_keeps_the_others(
    monkeypatch: pytest.MonkeyPatch, solids: dict[str, Path]
) -> None:
    def boom(*args: object, **kwargs: object) -> dict[str, float]:
        raise RuntimeError("boom")

    monkeypatch.setattr(run_scoring, "score_eccv", boom)
    columns, errors = StepScorer()._run_families(solids["box"], solids["box"])

    assert set(errors) == {"eccv"}
    metric_columns = {c for c in columns if not c.startswith("align_")}
    assert metric_columns and all(c.startswith("voxel_") for c in metric_columns)


def test_alignment_that_fails_still_leaves_a_score(
    monkeypatch: pytest.MonkeyPatch, solids: dict[str, Path]
) -> None:
    """The pose is a correction, not a precondition.

    A solid that cannot be re-posed is still a solid, and reporting nothing for
    it would lose a measurement the families were able to take.
    """

    def boom(*args: object, **kwargs: object) -> object:
        raise RuntimeError("boom")

    monkeypatch.setattr(align_orientation, "align_step", boom)
    columns, errors = StepScorer()._run_families(solids["box"], solids["box"])

    assert set(errors) == {"align"}
    assert columns["voxel_iou"] == 1.0
    assert "align_rotation_index" not in columns


def test_a_prediction_in_a_different_pose_is_scored_where_it_belongs(
    tmp_path: Path, solids: dict[str, Path]
) -> None:
    """Every family is orientation-sensitive, so alignment precedes all of them.

    A reconstruction that is right about the wrong axis used to score near
    zero, which is a statement about the modeller's axis convention rather
    than about the part.
    """

    quarter_turn_about_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    )
    rotated = tmp_path / "rotated.step"
    align_orientation.write_rotated_step(
        solids["fillet"], rotated, quarter_turn_about_x
    )

    scored = StepScorer().score(rotated, solids["fillet"])
    # The families do not align; that the scorer does is what this asserts.
    unaligned = score_eccv(rotated, solids["fillet"])

    assert scored.metrics["eccv_mean_f1"] > 0.99
    assert scored.metrics["voxel_iou"] > 0.99
    assert unaligned["eccv_mean_f1"] < scored.metrics["eccv_mean_f1"]


def _face_count(step_path: Path) -> int:
    import cadquery as cq

    return len(cq.importers.importStep(str(step_path)).val().Faces())


def test_splitting_closed_faces_is_off_unless_asked(
    tmp_path: Path, solids: dict[str, Path]
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    kept = StepScorer()._split_closed(solids["sphere"], scratch)

    assert StepScorer().split_closed_faces is False
    assert _face_count(kept) > _face_count(solids["sphere"])


def test_splitting_gives_a_seamed_face_the_two_the_target_would_have(
    tmp_path: Path,
) -> None:
    """CadQuery leaves a full cylinder whole; SolidWorks halves it at the seam.

    The metric assigns faces one to one, so a prediction with half the target's
    faces cannot recall more than half of them however right its geometry is.
    """
    import cadquery as cq

    cylinder = tmp_path / "cylinder.step"
    cq.exporters.export(
        cq.Workplane("XY").circle(5).extrude(20), str(cylinder), exportType="STEP"
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    split = StepScorer()._split_closed(cylinder, scratch)

    assert _face_count(cylinder) == 3
    assert _face_count(split) == 4


def test_normalization_that_fails_still_leaves_a_score(
    monkeypatch: pytest.MonkeyPatch, solids: dict[str, Path]
) -> None:
    def boom(*args: object, **kwargs: object) -> object:
        raise RuntimeError("boom")

    monkeypatch.setattr(normalize_brep, "split_closed_faces", boom)
    columns, errors = StepScorer(split_closed_faces=True)._run_families(
        solids["box"], solids["box"]
    )

    assert set(errors) == {"normalize"}
    assert columns["voxel_iou"] == 1.0


def test_the_scorer_survives_the_spawn_boundary() -> None:
    scorer = StepScorer(seed=3, voxel_resolution=32)
    assert pickle.loads(pickle.dumps(scorer)) == scorer


def test_a_non_positive_timeout_is_refused() -> None:
    with pytest.raises(ValueError):
        StepScorer(timeout_s=0)


def test_picks_the_final_attempt(make_run_dir, solids: dict[str, Path]) -> None:
    run_dir = make_run_dir({"000": solids["box"], "001": solids["sphere"]})
    found = latest_verified_step(run_dir)
    assert found is not None and found.parent.name == "001"


def test_a_final_attempt_without_a_solid_is_no_prediction(
    make_run_dir, solids: dict[str, Path]
) -> None:
    """The run submitted its last attempt, not the last one that worked."""

    run_dir = make_run_dir({"000": solids["box"], "001": None})
    assert latest_verified_step(run_dir) is None


def test_an_earlier_attempt_is_used_when_allowed(
    make_run_dir, solids: dict[str, Path]
) -> None:
    run_dir = make_run_dir({"000": solids["box"], "001": None})
    found = latest_verified_step(run_dir, last_only=False)
    assert found is not None and found.parent.name == "000"


def test_no_attempts_directory(tmp_path: Path) -> None:
    assert latest_verified_step(tmp_path) is None


def test_an_empty_attempts_directory(make_run_dir) -> None:
    assert latest_verified_step(make_run_dir({})) is None


def test_ignores_non_numeric_attempt_names(
    make_run_dir, solids: dict[str, Path]
) -> None:
    run_dir = make_run_dir({"000": solids["box"], "final": solids["sphere"]})
    found = latest_verified_step(run_dir)
    assert found is not None and found.parent.name == "000"


def test_score_run_records_what_produced_the_numbers(
    make_run_dir, solids: dict[str, Path]
) -> None:
    run_dir = make_run_dir({"000": solids["box"]})
    document = score_run(run_dir, solids["box"], StepScorer(seed=7))

    assert document["status"] == "OK"
    assert document["target_step"] == str(solids["box"])
    assert document["last_only"] is True
    assert document["scorer"]["seed"] == 7  # type: ignore[index]


def test_score_run_reports_a_run_that_submitted_nothing(
    make_run_dir, solids: dict[str, Path]
) -> None:
    document = score_run(make_run_dir({}), solids["box"], StepScorer())
    assert document["status"] == "NO_PREDICTION"
    assert document["pred_step"] is None
