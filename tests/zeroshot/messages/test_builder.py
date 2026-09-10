"""What a turn is shown of the run's files, and what it is never shown.

Input instructions and verification feedback translate host paths before the
model receives them. Most of what is checked here is an absence: no host path, no
file contents, and nothing about a sheet the run does not offer.
"""

import base64
from pathlib import Path
from typing import Literal

import pytest
from langchain_core.messages.content import ContentBlock

from zeroshot.pipeline.messages.artifact import (
    ArtifactPresenter,
    build_feedback_message_blocks,
)
from zeroshot.pipeline.messages.manifest import FeedbackManifest, InputManifest
from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.stages._base.prompt import StageInstructions
from zeroshot.pipeline.stages.drawings.contracts import (
    CropOf,
    DrawingSheet,
    DrawingSource,
    View,
    unread_sheet,
)


def _write(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _drawing_sheet(tmp_path: Path) -> DrawingSheet:
    return unread_sheet(
        "sheet_drawing",
        View.FULL_PAGE,
        _write(tmp_path / "input.dxf", b"RAW_DXF_MUST_NOT_BE_IN_PROMPT"),
    )


def _pictorial(tmp_path: Path, stem: str) -> DrawingSheet:
    return unread_sheet(
        f"sheet_{stem}",
        View.PERSPECTIVE,
        _write(tmp_path / f"{stem}.png", stem.encode()),
    )


def _input_manifest(tmp_path: Path, *sheets: DrawingSheet) -> InputManifest:
    return InputManifest(
        sample_id="sample-1",
        drawing=DrawingSource(sheets=list(sheets) or [_drawing_sheet(tmp_path)]),
    )


def _input_blocks(
    manifest: InputManifest,
    workdir: SandboxWorkdir,
    *,
    mode: Literal["path", "image"] = "path",
) -> list[ContentBlock]:
    return (
        StageInstructions(
            input_artifact=manifest.drawing,
            input_presentation_mode=mode,
            prompt_context={},
            workdir=workdir,
        )
        .create_artifact_message()
        .content_blocks
    )


def _feedback_manifest(
    *,
    sheets: tuple[DrawingSheet, ...] = (),
    errors: dict[str, str] | None = None,
) -> FeedbackManifest:
    return FeedbackManifest(
        verification_id="verification-1",
        drawing=DrawingSource(sheets=list(sheets)) if sheets else None,
        errors=errors or {},
    )


def _presenter(
    *, input_mode: str = "path", feedback_mode: str = "none"
) -> ArtifactPresenter:
    return ArtifactPresenter(
        input_mode=input_mode,  # type: ignore[arg-type]
        feedback_mode=feedback_mode,  # type: ignore[arg-type]
    )


@pytest.fixture
def workdir(tmp_path: Path) -> SandboxWorkdir:
    return SandboxWorkdir(host_bind_dir=tmp_path)


def _text(blocks: list[ContentBlock]) -> str:
    return "\n".join(block["text"] for block in blocks if block["type"] == "text")


@pytest.mark.parametrize(
    ("input_mode", "feedback_mode"),
    [("unknown", "none"), ("path", "unknown"), ("none", "none")],
)
def test_a_mode_the_presenter_cannot_honour_is_refused(
    input_mode: str, feedback_mode: str
) -> None:
    """`none` is a feedback outcome only: a run always offers its input."""
    with pytest.raises(ValueError):
        _presenter(input_mode=input_mode, feedback_mode=feedback_mode)


def test_a_sheet_is_named_where_the_model_can_open_it_and_nowhere_else(
    tmp_path: Path, workdir: SandboxWorkdir
) -> None:
    manifest = _input_manifest(tmp_path)
    (host,) = manifest.drawing.paths()

    text = _text(_input_blocks(manifest, workdir))

    assert str(workdir.host_to_sandbox_path(host)) in text
    assert str(host) not in text
    assert "RAW_DXF_MUST_NOT_BE_IN_PROMPT" not in text


def test_a_sheet_is_announced_with_the_view_it_holds(
    tmp_path: Path, workdir: SandboxWorkdir
) -> None:
    """Without the role a pictorial reads as one more sheet to measure."""
    manifest = _input_manifest(
        tmp_path, _drawing_sheet(tmp_path), _pictorial(tmp_path, "hlg")
    )

    text = _text(_input_blocks(manifest, workdir))

    assert "- sheet_drawing (full_page):" in text
    assert "- sheet_hlg (perspective):" in text


def test_a_full_page_says_its_views_are_told_apart_by_position(
    tmp_path: Path, workdir: SandboxWorkdir
) -> None:
    manifest = _input_manifest(tmp_path)

    text = _text(_input_blocks(manifest, workdir))

    assert "where they sit on the page" in text


def test_the_frame_is_not_said_here_because_it_belongs_to_the_round(
    tmp_path: Path, workdir: SandboxWorkdir
) -> None:
    """StageInstructions supplies the shared coordinate convention separately."""
    manifest = _input_manifest(tmp_path)

    text = _text(_input_blocks(manifest, workdir))

    assert "up=+y" not in text


def test_input_lists_files_without_stage_specific_measurement_advice(
    tmp_path: Path, workdir: SandboxWorkdir
) -> None:
    """Drawing measurement advice belongs to the drawings stage guidelines."""
    vector = _input_manifest(
        tmp_path, _drawing_sheet(tmp_path), _pictorial(tmp_path, "hlg")
    )
    raster = _input_manifest(
        tmp_path,
        unread_sheet(
            "sheet_drawing", View.FULL_PAGE, _write(tmp_path / "page.png", b"PNG")
        ),
    )

    vector_text = _text(_input_blocks(vector, workdir))
    raster_text = _text(_input_blocks(raster, workdir))

    assert "/work/input.dxf" in vector_text
    assert "/work/hlg.png" in vector_text
    assert "/work/page.png" in raster_text
    for text in (vector_text, raster_text):
        assert "ezdxf" not in text
        assert "OpenCV" not in text


def test_a_sheet_cut_out_of_another_is_named_like_any_other(
    tmp_path: Path, workdir: SandboxWorkdir
) -> None:
    """A cut-out is saved to a file of its own, so it is opened the same way."""
    page = _drawing_sheet(tmp_path)
    crop = DrawingSheet(
        name="sheet_front",
        role=View.FRONT,
        label=None,
        crop_of=CropOf(sheet=page.name, box=[0.0, 0.0, 10.0, 10.0]),
        scale=1.0,
        file=str(_write(tmp_path / "front.png", b"FRONT")),
        evidence=[],
        dimensions=[],
    )
    manifest = _input_manifest(tmp_path, page, crop)

    text = _text(_input_blocks(manifest, workdir))

    assert "- sheet_front (front):" in text


def test_path_mode_attaches_nothing_and_image_mode_attaches_every_raster(
    tmp_path: Path, workdir: SandboxWorkdir
) -> None:
    manifest = _input_manifest(
        tmp_path, _drawing_sheet(tmp_path), _pictorial(tmp_path, "hlg")
    )

    by_path = _input_blocks(manifest, workdir)
    by_image = _input_blocks(manifest, workdir, mode="image")

    assert [block["type"] for block in by_path] == ["text"]
    assert [block["type"] for block in by_image] == ["text", "text", "image"]
    image = by_image[2]
    assert image["type"] == "image"
    assert base64.b64decode(image["base64"]) == b"hlg"
    assert image["mime_type"] == "image/png"


def test_a_verification_that_drew_nothing_is_shown_as_nothing(
    tmp_path: Path, workdir: SandboxWorkdir
) -> None:
    manifest = _feedback_manifest()

    assert build_feedback_message_blocks(manifest, workdir, mode="path") == []


def test_a_projected_sheet_is_named_and_a_missing_one_is_explained(
    tmp_path: Path, workdir: SandboxWorkdir
) -> None:
    drawn = unread_sheet(
        "sheet_drawing",
        View.FULL_PAGE,
        _write(tmp_path / "feedback.dxf", b"FEEDBACK_DXF"),
    )
    manifest = _feedback_manifest(
        sheets=(drawn,), errors={"sheet_hlg": "renderer failed"}
    )

    text = _text(build_feedback_message_blocks(manifest, workdir, mode="path"))

    assert str(workdir.host_to_sandbox_path(Path(drawn.file or ""))) in text
    assert str(drawn.file) not in text
    assert "- sheet_hlg: unavailable (renderer failed)" in text


def test_withholding_the_feedback_also_withholds_why_it_is_missing(
    tmp_path: Path, workdir: SandboxWorkdir
) -> None:
    manifest = _feedback_manifest(errors={"sheet_hlg": "boom"})

    blocks = build_feedback_message_blocks(manifest, workdir, mode="none")

    assert blocks == []
