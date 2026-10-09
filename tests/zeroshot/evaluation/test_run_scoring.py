import pickle
from pathlib import Path

import pytest

from zeroshot.evaluation.run_scoring import (
    ScoreStatus,
    StepScorer,
    latest_verified_step,
    score_run,
)

# Supervision checks need no ECCV or Ortho2CAD columns.
_FAST = {"include_eccv": False, "include_ortho2cad": False}


def test_scores_a_valid_pair(solids: dict[str, Path]) -> None:
    report = StepScorer(**_FAST).score(solids["box"], solids["box"])
    assert report.status is ScoreStatus.OK
    assert report.errors == {}
    assert report.metrics["mesh_iou"] == pytest.approx(1.0)


def test_each_family_leads_with_its_most_telling_column(
    solids: dict[str, Path],
) -> None:
    """Reports show each family's first column, so metric order is not cosmetic."""
    metrics = StepScorer().score(solids["box"], solids["fillet"]).metrics
    leading = {
        family: next(n for n in metrics if n == family or n.startswith(f"{family}_"))
        for family in StepScorer.families()
    }

    assert leading == {
        "mesh": "mesh_iou",
        "chamfer": "chamfer",
        "hausdorff": "hausdorff",
        "eccv": "eccv_mean_f1",
        "ortho2cad": "ortho2cad_iou",
    }


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


def test_a_timeout_is_not_a_metric_failure(solids: dict[str, Path]) -> None:
    report = StepScorer(timeout_s=0.01).score(solids["box"], solids["box"])
    assert report.status is ScoreStatus.TIMEOUT
    assert set(report.errors) == {"scorer"}


def test_the_scorer_survives_the_spawn_boundary() -> None:
    scorer = StepScorer(seed=3, sample_points=16)
    assert pickle.loads(pickle.dumps(scorer)) == scorer


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


def test_picks_the_latest_round_before_its_latest_coding_attempt(
    tmp_path: Path,
    solids: dict[str, Path],
) -> None:
    root = tmp_path / "run" / "workspace" / "attempts"
    old = root / "round_000" / "coding" / "009"
    new = root / "round_001" / "coding" / "000"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    old.joinpath("output.step").write_bytes(solids["box"].read_bytes())
    new.joinpath("output.step").write_bytes(solids["sphere"].read_bytes())

    found = latest_verified_step(tmp_path / "run")

    assert found == new / "output.step"


def test_score_run_records_what_produced_the_numbers(
    make_run_dir, solids: dict[str, Path]
) -> None:
    run_dir = make_run_dir({"000": solids["box"]})
    document = score_run(run_dir, solids["box"], StepScorer(seed=7, **_FAST))

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
