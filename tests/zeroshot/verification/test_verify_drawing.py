from pathlib import Path, PurePosixPath

import pytest

from tests.zeroshot.contracts import drawing, evidence
from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.stages.drawings.contracts import (
    CropOf,
    DrawingSource,
    View,
    unread_sheet,
)
from zeroshot.pipeline.verification import AttemptStore, DrawingVerifier


def _baseline() -> DrawingSource:
    return DrawingSource(
        sheets=[unread_sheet("sheet_page", View.FULL_PAGE, "/work/source.png")]
    )


def _candidate() -> DrawingSource:
    front = (
        drawing()
        .sheets[0]
        .model_copy(
            update={
                "crop_of": CropOf(sheet="sheet_page", box=[0.0, 0.0, 10.0, 10.0]),
                "file": "/work/front.png",
            }
        )
    )
    return DrawingSource(sheets=[*_baseline().sheets, front])


def _verifier(tmp_path: Path) -> DrawingVerifier:
    (tmp_path / "source.png").write_bytes(b"source")
    (tmp_path / "front.png").write_bytes(b"front")
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    return DrawingVerifier(
        workdir=workdir,
        attempt_store=AttemptStore(workdir, round_source=lambda: 0),
        feedback_presentation_mode="path",
    )


def test_a_round_is_seeded_from_the_accepted_baseline(tmp_path: Path) -> None:
    verifier = _verifier(tmp_path)

    verifier.reset(_baseline())

    assert (
        DrawingSource.model_validate_json(
            (tmp_path / "drawing.json").read_text(encoding="utf-8")
        )
        == _baseline()
    )


def test_a_distinct_json_is_frozen_rendered_and_accepted_after_feedback(
    tmp_path: Path,
) -> None:
    verifier = _verifier(tmp_path)
    candidate = _candidate()
    verifier.reset(_baseline())
    verifier.source_path.write_text(candidate.model_dump_json(), encoding="utf-8")

    result, manifest = verifier.verify()

    attempt = tmp_path / "attempts" / "round_000" / "drawing" / "000"
    assert result.confirmed is True
    assert verifier.confirmed is False
    assert manifest is not None and manifest.drawing is not None
    assert (
        DrawingSource.model_validate_json(
            (attempt / "drawing.json").read_text(encoding="utf-8")
        )
        == candidate
    )
    assert (attempt / "front.dxf").is_file()
    assert (attempt / "front.png").is_file()

    feedback = verifier.feedback()

    assert verifier.confirmed is True
    assert verifier.accepted_drawing == candidate
    assert len(list((attempt.parent).iterdir())) == 1
    text = "\n".join(block["text"] for block in feedback if block["type"] == "text")
    assert "/work/attempts/round_000/drawing/000/front.png" in text


def test_invalid_json_is_preserved_as_a_failed_render_attempt(tmp_path: Path) -> None:
    verifier = _verifier(tmp_path)
    verifier.reset(_baseline())
    verifier.source_path.write_text("{", encoding="utf-8")

    feedback = verifier.feedback()

    attempt = tmp_path / "attempts" / "round_000" / "drawing" / "000"
    assert (attempt / "drawing.json").read_text(encoding="utf-8") == "{"
    assert verifier.confirmed is False
    assert verifier.accepted_drawing is None
    assert "Invalid JSON" in "\n".join(
        block["text"] for block in feedback if block["type"] == "text"
    )


def test_a_crop_must_retain_its_own_source_file(tmp_path: Path) -> None:
    verifier = _verifier(tmp_path)
    candidate = _candidate()
    candidate.sheets[-1].file = candidate.sheets[0].file
    verifier.reset(_baseline())
    verifier.source_path.write_text(candidate.model_dump_json(), encoding="utf-8")

    feedback = verifier.feedback()

    assert verifier.confirmed is False
    assert "cropped view must use its own file" in "\n".join(
        block["text"] for block in feedback if block["type"] == "text"
    )


@pytest.mark.parametrize(
    "reading",
    [
        evidence("line", start=[-2.0, 1.0], end=[5.0, 1.0]),
        evidence(
            "polyline",
            vertices=[1.0, 1.0, 12.0, 1.0, 12.0, 2.0],
        ),
        evidence("circle", center=[12.0, 5.0], radius=3.0),
    ],
)
def test_a_cropped_view_rejects_drawn_geometry_outside_its_local_bounds(
    reading,
) -> None:
    candidate = _candidate()
    candidate.sheets[-1].evidence = [reading]

    assert "outside this cropped view's 10 x 10 mm local bounds" in "\n".join(
        DrawingVerifier._validate_sheet_bounds(candidate.sheets[-1])
    )


def test_attempts_are_numbered_independently_by_round_and_stage(
    tmp_path: Path,
) -> None:
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    current_round = [0]
    store = AttemptStore(workdir, round_source=lambda: current_round[0])

    assert store.issue("drawing")[0] == "000"
    assert store.issue("drawing")[0] == "001"
    assert store.issue("coding")[0] == "000"
    current_round[0] = 1
    assert store.issue("drawing")[0] == "000"
    assert store.sandbox_attempt_dir(1, "coding", "007") == PurePosixPath(
        "/work/attempts/round_001/coding/007"
    )
    assert store.latest_sandbox_attempt_dir("drawing", 1) == PurePosixPath(
        "/work/attempts/round_001/drawing/000"
    )
    assert store.latest_sandbox_attempt_dir("coding", 1) == PurePosixPath(
        "/work/attempts/round_000/coding/000"
    )


@pytest.mark.parametrize(
    "root",
    [PurePosixPath(""), PurePosixPath(".."), PurePosixPath("nested/attempts")],
)
def test_attempt_store_rejects_a_non_basename_root(
    tmp_path: Path,
    root: PurePosixPath,
) -> None:
    with pytest.raises(ValueError, match="directory basename"):
        AttemptStore(
            SandboxWorkdir(host_bind_dir=tmp_path),
            round_source=lambda: 0,
            root_dirname=root,
        )
