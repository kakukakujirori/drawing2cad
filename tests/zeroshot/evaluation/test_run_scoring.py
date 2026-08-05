import pickle
from pathlib import Path

import pytest

from zeroshot.evaluation import run_scoring
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

    assert leading == {"eccv": "eccv_surface_f1", "voxel": "voxel_iou"}


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
    assert set(report.errors) == {"eccv", "voxel"}
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
    assert columns and all(column.startswith("voxel_") for column in columns)


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
