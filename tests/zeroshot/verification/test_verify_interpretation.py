import json
from pathlib import Path

import pytest
from PIL import Image

from tests.zeroshot.verification.test_interpretation_validation import raster_case
from zeroshot.pipeline.messages.manifest import register_view
from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.stages.interpretation.contracts import (
    UNDECIDED,
    DrawingInterpretation,
    View,
)
from zeroshot.pipeline.stages.interpretation.verify import InterpretationVerifier
from zeroshot.pipeline.verification.attempts import AttemptStore
from zeroshot.pipeline.workflow.middleware import VerifyOnWriteMiddleware


def _case(
    tmp_path: Path,
    measurements=None,
    *,
    pictorial: str | None = None,
    input_role: View = View.FULL_PAGE,
):
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
            "role": input_role,
            "file": "/work/source.png",
            "region": {"view": "view_page", "box_px": [0, 0, 1200, 1400]},
            "dimensions": [],
        },
    )
    given = [register_view("view_page", input_role, tmp_path / "source.png")]
    if pictorial is not None:
        Image.new("RGB", (60, 40), "white").save(tmp_path / f"{pictorial}.png")
        given.append(
            register_view(
                f"view_{pictorial}", View.PERSPECTIVE, tmp_path / f"{pictorial}.png"
            )
        )
    workdir = SandboxWorkdir(tmp_path)
    verifier = InterpretationVerifier(
        workdir,
        AttemptStore(workdir, round_source=lambda: 0),
        given,
    )
    seed = DrawingInterpretation(datum=UNDECIDED, views=given, features=[])
    return verifier, DrawingInterpretation.model_validate(data), seed


def test_first_round_seeds_the_input_and_revisions_keep_the_complete_baseline(tmp_path):
    verifier, candidate, seed = _case(tmp_path)
    verifier.source_path.write_text("stale")
    verifier.reset(seed)
    seeded = DrawingInterpretation.model_validate_json(
        verifier.source_path.read_bytes()
    )
    assert seeded == seed
    assert [view.name for view in seeded.views] == ["view_page"]
    verifier.reset(candidate)
    assert (
        DrawingInterpretation.model_validate_json(verifier.source_path.read_bytes())
        == candidate
    )
    assert not verifier.confirmed


def test_verification_fills_the_main_file_and_keeps_unadvertised_debug_records(
    tmp_path,
):
    verifier, candidate, seed = _case(
        tmp_path, [(4.2, 42), (10, 100), (20, 200), (30, 50)]
    )
    verifier.reset(seed)
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
    verifier, candidate, _ = _case(tmp_path)
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
    verifier, candidate, _ = _case(tmp_path)
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
    verifier, candidate, _ = _case(tmp_path)
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


def test_json_too_deep_to_parse_is_rejected_not_raised(tmp_path):
    verifier, _, seed = _case(tmp_path)
    verifier.reset(seed)
    verifier.source_path.write_text('{"views": ' + "[" * 10000 + "]" * 10000 + "}")
    assert "Invalid JSON: recursion limit exceeded" in verifier.feedback()[0]["text"]


def test_invalid_json_and_missing_original_page_are_rejected_and_recorded(tmp_path):
    verifier, candidate, seed = _case(tmp_path)
    verifier.reset(seed)
    verifier.source_path.write_text("{")
    assert "interpretation.json:1:2 Invalid JSON" in verifier.feedback()[0]["text"]
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


def test_a_schema_error_points_at_its_key_in_the_written_file(tmp_path):
    verifier, candidate, _ = _case(tmp_path)
    data = candidate.model_dump(mode="json")
    data["features"][0]["center"] = [0, 0]
    text = json.dumps(data, indent=2)
    verifier.source_path.write_text(text)
    (error,) = json.loads(verifier.feedback()[0]["text"].splitlines()[-1])["errors"]
    position, rest = error.split(" ", 1)
    _, line, column = position.split(":")
    assert text.splitlines()[int(line) - 1][int(column) - 1 :].startswith('"center"')
    name = candidate.features[0].name
    assert rest == f"$.features[0].center ({name}): Extra inputs are not permitted"


def test_a_check_after_parsing_points_at_its_key_in_the_written_file(tmp_path):
    """The schema is not the only thing that rejects a file; locate the rest too."""
    verifier, candidate, _ = _case(tmp_path)
    data = candidate.model_dump(mode="json")
    data["datum"] = f"origin {UNDECIDED}"
    text = json.dumps(data, indent=2)
    verifier.source_path.write_text(text)

    (error,) = json.loads(verifier.feedback()[0]["text"].splitlines()[-1])["errors"]

    position, rest = error.split(" ", 1)
    _, line, column = position.split(":")
    assert text.splitlines()[int(line) - 1][int(column) - 1 :].startswith('"datum"')
    assert rest.startswith(f"$.datum: datum still holds {UNDECIDED}")


def test_a_pictorial_input_is_not_required_to_come_back_as_a_full_page(tmp_path):
    """A pictorial fixes no axes, so leaving it undeclared loses no coordinate."""
    verifier, candidate, seed = _case(tmp_path, pictorial="hlg")
    verifier.reset(seed)
    verifier.source_path.write_text(candidate.model_dump_json())

    assert "Retain each original input file" not in verifier.feedback()[0]["text"]
    assert verifier.confirmed


@pytest.mark.parametrize("role", [View.FRONT, View.TOP, View.SECTION, View.UNKNOWN])
def test_registered_roles_survive_dimension_readings_and_enrichment(tmp_path, role):
    verifier, candidate, _ = _case(tmp_path, input_role=role)
    candidate.views[0].dimensions = candidate.views[1].dimensions
    candidate.views[1].dimensions = []
    verifier.reset(candidate)

    verifier.feedback()
    accepted = verifier.accepted_interpretation
    assert accepted is not None
    assert accepted.views[0].role == role
    assert accepted.views[0].scale == pytest.approx(0.1)
    assert accepted.views[0].region.box_uv is not None
    verifier.reset(accepted)
    assert verifier.verify().confirmed


@pytest.mark.parametrize("changed", ["name", "file", "role", "reference", "bounds"])
def test_original_identity_and_full_file_region_cannot_be_rewritten(tmp_path, changed):
    verifier, candidate, _ = _case(tmp_path, input_role=View.FRONT)
    data = candidate.model_dump()
    original = data["views"][0]
    if changed == "name":
        original["name"] = original["region"]["view"] = "view_renamed"
    elif changed == "file":
        original["file"] = "/work/front.png"
    elif changed == "role":
        original["role"] = "full_page"
    elif changed == "reference":
        original["region"]["view"] = "view_front"
    else:
        original["region"]["box_px"] = [0, 0, 100, 100]
    verifier.source_path.write_text(json.dumps(data))

    assert (
        "registered name, role and full-file region" in verifier.feedback()[0]["text"]
    )
    assert not verifier.confirmed


@pytest.mark.parametrize(
    ("role", "accepted"),
    [
        (View.FRONT, True),
        (View.RIGHT, True),
        (View.SECTION, False),
        (View.UNKNOWN, False),
    ],
)
def test_a_full_page_input_requires_an_orthographic_view(tmp_path, role, accepted):
    verifier, candidate, _ = _case(tmp_path)
    candidate.views[1].role = role
    verifier.source_path.write_text(candidate.model_dump_json())

    text = verifier.feedback()[0]["text"]

    assert verifier.confirmed is accepted
    assert ("full_page input is an unsplit page" in text) is not accepted


def test_a_single_view_may_reuse_the_full_page_file_and_bounds(tmp_path):
    verifier, candidate, _ = _case(tmp_path)
    candidate.views[1].role = View.SECTION
    data = candidate.model_dump()
    data["views"].append(
        {
            "name": "view_single",
            "role": "front",
            "file": "/work/source.png",
            "region": {"view": "view_page", "box_px": [0, 0, 1200, 1400]},
            "dimensions": [],
        }
    )
    verifier.source_path.write_text(json.dumps(data))

    verifier.feedback()

    assert verifier.confirmed


def test_the_full_page_cannot_be_relabelled_as_the_view(tmp_path):
    verifier, candidate, _ = _case(tmp_path)
    candidate.views[0].role = View.FRONT
    verifier.source_path.write_text(candidate.model_dump_json())

    assert "Retain each original input file" in verifier.feedback()[0]["text"]
    assert not verifier.confirmed


def test_retained_pictorial_keeps_its_registration(tmp_path):
    verifier, candidate, seed = _case(tmp_path, pictorial="hlg")
    pictorial = seed.views[-1].model_copy(update={"file": "/work/hlg.png"})
    candidate.views.append(pictorial)
    verifier.reset(candidate)
    assert verifier.verify().confirmed
    candidate.views[-1].role = View.FULL_PAGE
    verifier.reset(candidate)
    assert not verifier.verify().confirmed


def test_absent_calibration_stays_valid_and_reports_zero_measurements(tmp_path):
    verifier, candidate, _ = _case(tmp_path, [])
    verifier.reset(candidate)
    feedback = verifier.feedback()[0]["text"]
    assert verifier.confirmed
    assert verifier.accepted_interpretation.views[1].scale is None
    assert '"status": "insufficient_evidence"' in feedback
    assert '"inliers": "0/0"' in feedback
