import pickle
from pathlib import Path

import pytest

from zeroshot.evaluation.run_scoring import (
    ScoreStatus,
    StepScorer,
    format_metric,
    format_score,
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


def test_every_headline_is_a_column_the_scorer_returns(
    solids: dict[str, Path],
) -> None:
    """A renamed column would otherwise drop out of every report unnoticed."""
    metrics = StepScorer().score(solids["box"], solids["fillet"]).metrics

    assert set(StepScorer.headlines()) <= set(metrics)


def test_ortho2cad_iou_is_scored_but_not_a_headline(solids: dict[str, Path]) -> None:
    metrics = StepScorer().score(solids["box"], solids["box"]).metrics

    assert "ortho2cad_iou" in metrics
    assert "ortho2cad_iou" not in StepScorer.headlines()


def test_the_report_shows_hausdorff_to_six_places_and_the_f1_breakdown() -> None:
    document = {
        "status": "OK",
        "pred_step": "pred.step",
        "metrics": {
            "hausdorff": 0.0123456789,
            "eccv_mean_f1": 0.5,
            "eccv_surface_f1": 0.8,
            "eccv_edge_f1": 0.6,
            "eccv_vertex_f1": 0.4,
            "eccv_topology_f1": 0.2,
            "eccv_surface_precision": 0.9,
            "ortho2cad_iou": 0.7,
        },
        "errors": {},
    }

    lines = format_score(document).splitlines()

    assert "  hausdorff          0.012346" in lines
    assert [line.split()[0] for line in lines[3:]] == [
        "eccv_mean_f1",
        "eccv_surface_f1",
        "eccv_edge_f1",
        "eccv_vertex_f1",
        "eccv_topology_f1",
    ]
    assert not any("ortho2cad" in line or "precision" in line for line in lines)


def test_a_value_that_would_show_as_zero_keeps_three_figures() -> None:
    assert format_metric(0.000012345) == "1.23e-05"
    assert format_metric(0.000012345, decimals=6) == "0.000012"
    assert format_metric(0.0, decimals=6) == "0.000000"


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
    found, _ = latest_verified_step(run_dir)
    assert found is not None and found.parent.name == "001"


def test_a_final_attempt_without_a_solid_is_no_prediction(
    make_run_dir, solids: dict[str, Path]
) -> None:
    """The run submitted its last attempt, not the last one that worked."""

    run_dir = make_run_dir({"000": solids["box"], "001": None})
    assert latest_verified_step(run_dir)[0] is None


def test_an_earlier_attempt_is_used_when_allowed(
    make_run_dir, solids: dict[str, Path]
) -> None:
    run_dir = make_run_dir({"000": solids["box"], "001": None})
    found, _ = latest_verified_step(run_dir, last_only=False)
    assert found is not None and found.parent.name == "000"


def test_skips_newer_attempts_that_fail_the_step_checks(
    make_run_dir, solids: dict[str, Path], junk_step: Path
) -> None:
    run_dir = make_run_dir({"000": solids["box"], "001": junk_step})

    found, skipped = latest_verified_step(run_dir, last_only=False)
    document = score_run(run_dir, solids["box"], StepScorer(**_FAST), last_only=False)

    assert found is not None and found.parent.name == "000"
    assert "Failed to import STEP" in skipped["round_000/coding/001"]
    assert document["pred_step"] == str(found)
    assert document["skipped_attempts"] == skipped
    assert latest_verified_step(run_dir, last_only=True)[0] is None


def test_no_attempts_directory(tmp_path: Path) -> None:
    assert latest_verified_step(tmp_path)[0] is None


def test_an_empty_attempts_directory(make_run_dir) -> None:
    assert latest_verified_step(make_run_dir({}))[0] is None


def test_ignores_non_numeric_attempt_names(
    make_run_dir, solids: dict[str, Path]
) -> None:
    run_dir = make_run_dir({"000": solids["box"], "final": solids["sphere"]})
    found, _ = latest_verified_step(run_dir)
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

    found, _ = latest_verified_step(tmp_path / "run")

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
