"""The scorer, the pipeline score.json seam, and run-level CD aggregation."""

import json
import os
from dataclasses import asdict
from pathlib import Path

import cadquery as cq
import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from zeroshot import run_pipeline
from zeroshot.evaluation.aggregate_run import (
    SampleRow,
    Terminal,
    collect,
    format_report,
    headline_columns,
    notes,
    summarize,
)
from zeroshot.evaluation.preprocess import AlignmentError
from zeroshot.evaluation.run_scoring import (
    ScoreReport,
    ScoreStatus,
    StepScorer,
    score_run,
)


def _box(tmp_path):
    path = tmp_path / "gt.step"
    cq.Workplane("XY").box(30, 20, 10).val().exportStep(str(path))
    return path


def _run(tmp_path, sample_id, step=None):
    run = tmp_path / sample_id
    attempt = run / "workspace" / "attempts" / "round_000" / "coding" / "000"
    attempt.mkdir(parents=True)
    if step:
        attempt.joinpath("output.step").write_bytes(step.read_bytes())
    events = [
        {
            "event": "verification",
            "timestamp_ms": 0,
            "data": {"report": {"status": "VERIFIED" if step else "FAILED"}},
        },
        {"event": "run_completed", "timestamp_ms": 1, "data": {"duration_ms": 1}},
    ]
    run.joinpath("events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events)
    )
    return run


def test_default_config_uses_corrected_target_and_new_scorer():
    directory = Path(__file__).resolve().parents[3] / "zeroshot" / "configs"
    with initialize_config_dir(config_dir=str(directory), version_base="1.3"):
        config = compose(config_name="default")
    assert config.sample.target_step_path == "data/test_vlm/target_step_ori/000364.step"
    scorer = instantiate(config.evaluation.scorer)
    assert isinstance(scorer, StepScorer)
    assert scorer.reference_extent == 1.8


def test_pipeline_score_and_aggregate_include_failed_builds(tmp_path):
    gt = _box(tmp_path)
    _run(tmp_path, "000364", gt)
    directory = Path(__file__).resolve().parents[3] / "zeroshot" / "configs"
    with initialize_config_dir(config_dir=str(directory), version_base="1.3"):
        config = compose(
            config_name="default",
            overrides=[
                f"artifact_root={tmp_path}",
                f"sample.target_step_path={gt}",
                "evaluation.scorer.auc_tr_min_cd=0.01",
                "evaluation.scorer.auc_tr_max_cd=1.0",
            ],
        )
    run_pipeline.score(config)
    good = json.loads((tmp_path / "000364" / "score.json").read_text())
    assert good["build_valid"] is True
    assert good["metrics"]["mesh_iou"] == pytest.approx(1)
    assert {"chamfer", "hausdorff", "eccv_mean_f1", "ortho2cad_iou"} <= good[
        "metrics"
    ].keys()
    empty = _run(tmp_path, "empty")
    report = score_run(
        empty, gt, instantiate(config.evaluation.scorer), last_only=False
    )
    empty.joinpath("score.json").write_text(json.dumps(report))
    rows = collect(tmp_path)
    summary = summarize(rows)
    assert summary.itercad["total_samples"] == 2
    assert summary.itercad["invalid_predictions"] == 1
    assert summary.itercad["auc_tr"] == pytest.approx(0.5)
    assert summary.itercad["mean_cd"] == good["metrics"]["chamfer_diag"]
    assert summary.itercad["auc_tr_min_cd"] == 0.01
    assert summary.metrics["mesh_iou"].overall == pytest.approx(0.5)
    assert summary.metrics["chamfer"].overall is None
    rendered = format_report(
        rows, summary, headline_columns(rows, StepScorer().families())
    )
    assert "AUC-TR" in rendered and "Med. CD" in rendered
    assert "generation_failure" in rendered
    json.dumps(asdict(summary), allow_nan=False)


def test_scorer_distinguishes_model_and_evaluator_failures(tmp_path):
    gt = _box(tmp_path)
    junk = tmp_path / "junk.step"
    junk.write_text("not STEP")
    scorer = StepScorer(include_eccv=False, include_ortho2cad=False)
    invalid = scorer.score(junk, gt).as_dict()
    assert invalid["status"] == "GENERATION_FAILED"
    assert invalid["build_valid"] is False
    assert invalid["errors"]["prediction"]["kind"] == "generation_failure"
    broken = scorer.score(gt, junk).as_dict()
    assert broken["status"] == "FAILED"
    assert broken["build_valid"] is None
    assert broken["errors"]["preprocess"]["kind"] == "evaluation_error"
    missing = scorer.score(tmp_path / "missing.step", gt).as_dict()
    assert missing["status"] == "NO_PREDICTION" and missing["build_valid"] is False


def test_alignment_failure_at_pipeline_scorer_keeps_only_ortho2cad(
    tmp_path, monkeypatch
):
    gt = _box(tmp_path)

    def failed(*args):
        raise AlignmentError("Boolean failed")

    monkeypatch.setattr("zeroshot.evaluation.preprocess.orientation_ious", failed)
    report = StepScorer()._run_families(gt, gt).as_dict()
    assert report["status"] == "PARTIAL"
    assert report["metrics"].keys() == {"ortho2cad_iou"}
    assert report["errors"]["alignment"]["kind"] == "evaluation_error"


def test_aggregate_keeps_partial_cd_and_refuses_zero_for_evaluator_errors():
    good = SampleRow(
        "good",
        Terminal.COMPLETED,
        score_status="PARTIAL",
        metrics={"chamfer": 0.001, "chamfer_diag": 0.0002, "mesh_iou": 0.8},
        build_valid=True,
        score_errors={"eccv": {"kind": "evaluation_error", "message": "sampling"}},
    )
    empty = SampleRow(
        "empty",
        Terminal.COMPLETED,
        score_status="GENERATION_FAILED",
        build_valid=False,
    )
    summary = summarize([good, empty])
    assert summary.itercad["mean_cd"] == 0.0002
    assert summary.metrics["chamfer"].scored == 0.001
    assert summary.metrics["chamfer"].overall is None
    assert summary.metrics["mesh_iou"].overall == pytest.approx(0.4)
    timeout = SampleRow(
        "timeout",
        Terminal.COMPLETED,
        score_status="TIMEOUT",
        score_errors={"scorer": {"kind": "evaluation_error", "message": "timeout"}},
    )
    summary = summarize([good, empty, timeout])
    assert summary.itercad["total_samples"] == 3
    assert summary.itercad["invalid_predictions"] == 1
    assert summary.itercad["auc_tr"] is None
    assert summary.itercad["mean_cd"] is None
    assert summary.metrics["mesh_iou"].overall is None


def test_a_sample_without_score_json_stays_out_of_the_metrics():
    good = SampleRow(
        "good",
        Terminal.COMPLETED,
        score_status="OK",
        metrics={"chamfer_diag": 0.0, "mesh_iou": 0.8},
        build_valid=True,
    )
    no_target = SampleRow("017", Terminal.COMPLETED)
    summary = summarize([good, no_target])
    assert summary.itercad["total_samples"] == 1
    assert summary.itercad["auc_tr"] == pytest.approx(1)
    assert summary.metrics["mesh_iou"].overall == pytest.approx(0.8)
    assert any("017  no score.json" in line for line in notes([good, no_target]))


def test_aggregate_rejects_different_protocols_and_auc_configs():
    shared = SampleRow(
        "shared",
        Terminal.COMPLETED,
        score_status="OK",
        metrics={"chamfer": 0},
        build_valid=True,
        protocol={"reference_extent": 1.8},
    )
    unversioned = SampleRow(
        "old", Terminal.COMPLETED, score_status="OK", metrics={"mesh_iou": 1}
    )
    with pytest.raises(ValueError, match="protocol"):
        summarize([shared, unversioned])
    rows = [
        SampleRow(
            str(i),
            Terminal.COMPLETED,
            auc_options={"min_cd": value},
        )
        for i, value in enumerate([1e-5, 1e-4])
    ]
    with pytest.raises(ValueError, match="AUC"):
        summarize(rows)


class _CrashScorer(StepScorer):
    def _run_families(self, pred_step, gt_step):
        os._exit(7)


class _LargeReportScorer(StepScorer):
    def _run_families(self, pred_step, gt_step):
        return ScoreReport(ScoreStatus.OK, details={"payload": "x" * 262144})


def test_native_crash_is_an_evaluator_error(tmp_path):
    gt = _box(tmp_path)
    report = _CrashScorer().score(gt, gt).as_dict()
    assert report["status"] == "FAILED"
    assert report["build_valid"] is None
    assert report["errors"]["scorer"]["kind"] == "evaluation_error"
    assert "exitcode=7" in report["errors"]["scorer"]["message"]


def test_large_reports_do_not_deadlock_the_scoring_pipe(tmp_path):
    gt = _box(tmp_path)
    report = _LargeReportScorer(timeout_s=10).score(gt, gt)
    assert report.status is ScoreStatus.OK
    assert len(report.details["payload"]) == 262144


@pytest.mark.parametrize(
    "options",
    [
        {"timeout_s": 0},
        {"reference_extent": float("nan")},
        {"sample_points": 0},
        {"auc_tr_num_points": 1},
        {"auc_tr_min_cd": 1},
    ],
)
def test_bad_scoring_configuration_is_refused(options):
    with pytest.raises(ValueError):
        StepScorer(**options)
