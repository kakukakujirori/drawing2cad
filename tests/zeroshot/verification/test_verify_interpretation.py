import json
from pathlib import Path

import pytest
from PIL import Image

from tests.zeroshot.contracts import UNTURNED
from tests.zeroshot.verification.test_interpretation_validation import (
    dxf_case,
    raster_case,
)
from zeroshot.pipeline.messages.manifest import register_view
from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.stages.interpretation.contracts import (
    UNDECIDED,
    DrawingInterpretation,
    DrawingView,
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
            "u_axis": UNTURNED.get(input_role, (None, None))[0],
            "v_axis": UNTURNED.get(input_role, (None, None))[1],
        },
    )
    data["views"][1]["region"] = {"view": "view_page", "box_px": [0, 0, 1200, 1400]}
    given = [
        register_view(
            "view_page",
            input_role,
            tmp_path / "source.png",
            axes=UNTURNED.get(input_role),
        )
    ]
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


def _as_role(view: DrawingView, role: View) -> DrawingView:
    """The same sheet under another role, with that role's own axes."""
    u_axis, v_axis = UNTURNED.get(role, (None, None))
    return view.model_copy(update={"role": role, "u_axis": u_axis, "v_axis": v_axis})


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
    Image.new("RGB", (1200, 1400), "white").save(tmp_path / "front.png")
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
    data["views"][0]["region"]["view"] = "view_front"
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
        data["views"][1]["region"]["view"] = "view_renamed"
    elif changed == "file":
        original["file"] = "/work/front.png"
    elif changed == "role":
        original.update(role="full_page", u_axis=None, v_axis=None)
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
    candidate.views[1] = _as_role(candidate.views[1], role)
    verifier.source_path.write_text(candidate.model_dump_json())

    text = verifier.feedback()[0]["text"]

    assert verifier.confirmed is accepted
    assert ("full_page input is an unsplit page" in text) is not accepted


@pytest.mark.parametrize(
    ("box_px", "accepted"),
    [([0, 0, 1200, 1400], True), ([100, 100, 900, 1100], False)],
)
@pytest.mark.parametrize("page_first", [True, False])
def test_full_page_file_reuse_requires_full_page_bounds(
    tmp_path, box_px, accepted, page_first
):
    verifier, candidate, _ = _case(tmp_path)
    candidate.views[1] = _as_role(candidate.views[1], View.SECTION)
    data = candidate.model_dump()
    data["views"].append(
        {
            "name": "view_single",
            "role": "front",
            "file": "/work/source.png",
            "region": {"view": "view_page", "box_px": box_px},
            "dimensions": [],
            "u_axis": "+x",
            "v_axis": "+z",
        }
    )
    if not page_first:
        data["views"] = data["views"][1:] + data["views"][:1]
    verifier.source_path.write_text(json.dumps(data))

    text = verifier.feedback()[0]["text"]

    assert verifier.confirmed is accepted
    if not accepted:
        child_index = 2 if page_first else 1
        assert f"$.views[{child_index}].file: view_single (front)" in text
        assert "full-page orthographic view" in text
        assert "Save a crop" in text


def test_the_full_page_cannot_be_relabelled_as_the_view(tmp_path):
    verifier, candidate, _ = _case(tmp_path)
    candidate.views[0] = _as_role(candidate.views[0], View.FRONT)
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


@pytest.mark.parametrize("change", ["resize", "rotate", "pixel"])
def test_crop_must_match_parent_pixels_even_on_first_submission(tmp_path, change):
    verifier, candidate, _ = _case(tmp_path)
    with Image.open(tmp_path / "source.png") as source:
        source.putpixel((450, 750), (0, 0, 0))
        source.save(tmp_path / "source.png")
        crop = source.copy()
    if change == "resize":
        crop = crop.resize((2400, 2800))
    elif change == "rotate":
        crop = crop.rotate(180)
    else:
        crop.putpixel((600, 800), (0, 0, 0))
    crop.save(tmp_path / "front.png")
    submitted = candidate.model_dump_json()
    verifier.source_path.write_text(submitted)

    text = verifier.feedback()[0]["text"]

    assert not verifier.confirmed
    assert "$.views[1].file" in text and "unmodified 1:1 crop" in text
    assert verifier.source_path.read_text() == submitted
    # Correcting only the pixels invalidates the cached failure and is accepted.
    with Image.open(tmp_path / "source.png") as source:
        source.convert("RGBA").save(tmp_path / "front.png")
    verifier.feedback()
    assert verifier.confirmed
    # A later image-only overwrite also invalidates acceptance.
    crop.save(tmp_path / "front.png")
    assert not verifier.confirmed
    text = verifier.feedback()[0]["text"]
    assert (
        "image_size disagrees" if change == "resize" else "unmodified 1:1 crop"
    ) in text
    assert not verifier.confirmed


@pytest.mark.parametrize("cycle", [False, True])
def test_new_raster_cannot_replace_its_parent_with_self_reference_or_a_cycle(
    tmp_path, cycle
):
    verifier, candidate, _ = _case(tmp_path)
    data = candidate.model_dump()
    data["views"][1]["region"]["view"] = "view_front"
    if cycle:
        child = dict(data["views"][1], name="view_child", dimensions=[])
        child["region"] = {"view": "view_front", "box_px": [0, 0, 1200, 1400]}
        data["views"].append(child)
        data["views"][1]["region"]["view"] = "view_child"
    verifier.source_path.write_text(json.dumps(data))

    text = verifier.feedback()[0]["text"]

    assert not verifier.confirmed
    assert "$.views[1].region" in text and "lead to an original input view" in text


def test_nested_crops_can_reference_an_accepted_parent_view(tmp_path):
    verifier, candidate, _ = _case(tmp_path)
    child = candidate.views[1].model_copy(
        update={"name": "view_child", "file": "/work/child.png", "dimensions": []}
    )
    child.region = child.region.model_copy(
        update={"view": "view_front", "box_px": (100, 200, 800, 1000)}
    )
    with Image.open(tmp_path / "front.png") as parent:
        parent.crop(child.region.box_px).save(tmp_path / "child.png")
    candidate.views.append(child)
    verifier.source_path.write_text(candidate.model_dump_json())

    verifier.feedback()

    assert verifier.confirmed


@pytest.mark.parametrize("child_reference", [None, "view_front", "view_detail"])
def test_native_dxf_input_and_derived_dxf_do_not_require_raster_crop_checks(
    tmp_path, child_reference
):
    candidate = dxf_case(tmp_path)
    original = register_view(
        "view_front", View.FRONT, tmp_path / "front.dxf", axes=("+x", "+z")
    )
    if child_reference is not None:
        # A separate native drawing can cite its source or retain its own frame;
        # neither introduces raster resize/pixel requirements.
        (tmp_path / "detail.dxf").write_bytes((tmp_path / "front.dxf").read_bytes())
        child = candidate.views[0].model_copy(
            update={
                "name": "view_detail",
                "role": View.DETAIL,
                "file": "/work/detail.dxf",
                "region": candidate.views[0].region.model_copy(
                    update={"view": child_reference}
                ),
                "dimensions": [],
                "u_axis": None,
                "v_axis": None,
            }
        )
        candidate.views.append(child)
    workdir = SandboxWorkdir(tmp_path)
    verifier = InterpretationVerifier(
        workdir, AttemptStore(workdir, round_source=lambda: 0), [original]
    )
    verifier.reset(candidate)

    verifier.feedback()

    assert verifier.confirmed
    accepted = verifier.accepted_interpretation
    assert accepted is not None
    assert all(
        view.image_size is None and view.scale is None for view in accepted.views
    )
    assert accepted.views[-1].region.box_uv == candidate.views[-1].region.box_uv
