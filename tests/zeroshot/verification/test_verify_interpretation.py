import json
from pathlib import Path

import pytest
from PIL import Image

from tests.zeroshot.verification.test_interpretation_validation import raster_case
from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.stages.drawings.contracts import (
    DrawingSheet,
    DrawingSource,
)
from zeroshot.pipeline.stages.interpretation.contracts import (
    DrawingInterpretation,
    View,
)
from zeroshot.pipeline.verification.attempts import AttemptStore
from zeroshot.pipeline.verification.verify_interpretation import InterpretationVerifier
from zeroshot.pipeline.workflow.middleware import VerifyOnWriteMiddleware


def _case(tmp_path: Path, measurements=None):
    data = (
        raster_case(tmp_path)
        if measurements is None
        else raster_case(tmp_path, measurements)
    ).model_dump()
    Image.new("RGB", (1200, 1400), "white").save(tmp_path / "source.png")
    data["views"].insert(
        0,
        {
            "name": "view_page",
            "role": "full_page",
            "file": "/work/source.png",
            "region": {"view": "view_page", "box_px": [0, 0, 1200, 1400]},
            "dimensions": [],
        },
    )
    workdir = SandboxWorkdir(tmp_path)
    verifier = InterpretationVerifier(
        workdir,
        AttemptStore(workdir, round_source=lambda: 0),
        DrawingSource(
            sheets=[
                DrawingSheet(
                    name="sheet_page",
                    role=View.FULL_PAGE,
                    crop_of=None,
                    scale=1,
                    file=str(tmp_path / "source.png"),
                    evidence=[],
                    dimensions=[],
                )
            ]
        ),
    )
    return verifier, DrawingInterpretation.model_validate(data)


def test_first_round_is_unseeded_and_revisions_keep_the_complete_baseline(tmp_path):
    verifier, candidate = _case(tmp_path)
    verifier.source_path.write_text("stale")
    verifier.reset(None)
    assert not verifier.source_path.exists()
    verifier.reset(candidate)
    assert (
        DrawingInterpretation.model_validate_json(verifier.source_path.read_bytes())
        == candidate
    )
    assert not verifier.confirmed


def test_verification_fills_the_main_file_and_keeps_unadvertised_debug_records(
    tmp_path,
):
    verifier, candidate = _case(tmp_path, [(4.2, 42), (10, 100), (20, 200), (30, 50)])
    verifier.reset(None)
    payload = candidate.model_dump_json()
    verifier.source_path.write_text(payload)
    result = verifier.verify()
    assert result.confirmed and not verifier.confirmed
    feedback = verifier.feedback()[0]["text"]
    accepted = verifier.accepted_interpretation
    assert accepted is not None
    assert accepted.views[1].scale == pytest.approx(0.1)
    assert accepted.features[0].parameters["length"] is None
    assert '"inliers": "3/4"' in feedback
    assert "inlier_fraction" not in feedback
    assert '"outliers": ["dim_length_3"]' in feedback
    attempt = tmp_path / "attempts/round_000/interpretation/000"
    report = json.loads((attempt / "_interpretation_validation_log.json").read_text())
    summary = json.loads(feedback.splitlines()[-1])
    assert summary.keys() == report.keys()
    assert summary["reports"].keys() == report["reports"].keys()
    assert len(report["reports"]["view_front"]["measurements"]) == 4
    assert (attempt / "_interpretation_raw.json").read_text() == payload
    assert (
        DrawingInterpretation.model_validate_json(
            (attempt / "interpretation.json").read_bytes()
        )
        == accepted
    )
    assert (
        verifier.source_path.read_bytes()
        == (attempt / "interpretation.json").read_bytes()
    )
    assert verifier.source_path.read_text() != payload
    assert "_interpretation_raw" not in feedback
    assert "_interpretation_validation_log" not in feedback
    assert "attempts/" not in feedback
    assert not (attempt / "validated_interpretation.json").exists()
    assert verifier.verify() is result
    assert len(list(attempt.parent.iterdir())) == 1


def test_automatic_enrichment_does_not_trigger_another_verification(tmp_path):
    verifier, candidate = _case(tmp_path)
    middleware = VerifyOnWriteMiddleware(verifier, fingerprint=verifier.source_digest)
    payload = candidate.model_dump_json()
    verifier.source_path.write_text(payload)
    update = middleware.before_model({}, None)
    assert update is not None and verifier.confirmed
    assert verifier.source_path.read_text() != payload
    assert middleware.before_model({}, None) is None
    assert verifier.verify().attempt_id == "000"
    assert (
        tmp_path / "attempts/round_000/interpretation/000/_interpretation_raw.json"
    ).read_text() == payload


def test_failed_writeback_preserves_the_submission(tmp_path, monkeypatch):
    verifier, candidate = _case(tmp_path)
    payload = candidate.model_dump_json()
    verifier.source_path.write_text(payload)

    def fail_replace(self, target):
        raise OSError("writeback failed")

    monkeypatch.setattr(Path, "replace", fail_replace)
    feedback = verifier.feedback()[0]["text"]
    assert "writeback failed" in feedback
    assert not verifier.confirmed
    assert verifier.accepted_interpretation is None
    assert verifier.source_path.read_text() == payload
    attempt = tmp_path / "attempts/round_000/interpretation/000"
    assert (attempt / "_interpretation_raw.json").read_text() == payload
    assert json.loads((attempt / "_interpretation_validation_log.json").read_text())[
        "errors"
    ]


def test_crop_edits_trigger_feedback_and_do_not_reuse_previous_acceptance(tmp_path):
    verifier, candidate = _case(tmp_path)
    verifier.reset(candidate)
    middleware = VerifyOnWriteMiddleware(verifier, fingerprint=verifier.source_digest)
    verifier.feedback()
    assert verifier.confirmed
    (tmp_path / "front.png").write_bytes(b"broken image")
    assert not verifier.confirmed
    failed = verifier.verify()
    assert failed.attempt_id == "001" and not failed.confirmed
    assert verifier.accepted_interpretation is None
    update = middleware.before_model({}, None)
    assert update is not None and "invalid" in update["messages"][0].text
    Image.new("RGB", (1200, 1400), "gray").save(tmp_path / "front.png")
    update = middleware.before_model({}, None)
    assert update is not None and verifier.confirmed
    assert verifier.verify().attempt_id == "002"


def test_invalid_json_and_missing_original_page_are_rejected_and_recorded(tmp_path):
    verifier, candidate = _case(tmp_path)
    verifier.reset(None)
    verifier.source_path.write_text("{")
    assert "Invalid JSON" in verifier.feedback()[0]["text"]
    assert verifier.accepted_interpretation is None
    attempt = tmp_path / "attempts/round_000/interpretation/000"
    assert (attempt / "_interpretation_raw.json").read_text() == "{"
    assert not (attempt / "interpretation.json").exists()
    assert verifier.source_path.read_text() == "{"
    assert json.loads((attempt / "_interpretation_validation_log.json").read_text())[
        "errors"
    ]
    data = candidate.model_dump()
    data["views"].pop(0)
    verifier.source_path.write_text(json.dumps(data))
    assert "Retain each original input file" in verifier.feedback()[0]["text"]
    assert not verifier.confirmed


def test_absent_calibration_stays_valid_and_reports_zero_measurements(tmp_path):
    verifier, candidate = _case(tmp_path, [])
    verifier.reset(candidate)
    feedback = verifier.feedback()[0]["text"]
    assert verifier.confirmed
    assert verifier.accepted_interpretation.views[1].scale is None
    assert '"status": "insufficient_evidence"' in feedback
    assert '"inliers": "0/0"' in feedback
