"""How the run's files are announced, on the way in and on the way back.

Both sides are a `DrawingSource`, so one set of cases covers them: which files
are named, which ride as images, and what the message says about the format it
is handing over.  A sheet's bytes never appear unless the mode asked for them.
"""

import base64
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest

from zeroshot.pipeline.messages import (
    ArtifactPresenter,
    DrawingSheet,
    DrawingSource,
    FeedbackManifest,
    InputManifest,
    View,
)
from zeroshot.pipeline.messages.artifact import _MIME_TYPES, _SandboxSheet
from zeroshot.pipeline.messages.contracts.drawings import DRAWING_SUFFIXES
from zeroshot.pipeline.sandbox import SandboxWorkdir

STYLES = ("style-a", "style-b", "style-c")


def _write(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _input_manifest(
    tmp_path: Path,
    drawing: DrawingSource | None = None,
) -> InputManifest:
    return InputManifest(
        sample_id="sample-1",
        drawing=drawing
        or DrawingSource(
            sheets=[
                DrawingSheet(
                    role=View.UNKNOWN,
                    label="drawing",
                    file=_write(
                        tmp_path / "input.dxf",
                        b"RAW_DRAWING_MUST_NOT_BE_IN_PROMPT",
                    ),
                )
            ]
        ),
    )


def _feedback_manifest(
    tmp_path: Path,
    *,
    render_count: int = 0,
    include_dxf: bool = False,
    errors: dict[str, str] | None = None,
) -> FeedbackManifest:
    sheets = [
        DrawingSheet(
            role=View.PERSPECTIVE,
            label=style,
            file=_write(
                tmp_path / f"feedback-{index}.png",
                f"feedback-{style}".encode(),
            ),
        )
        for index, style in enumerate(STYLES[:render_count])
    ]
    if include_dxf:
        sheets.insert(
            0,
            DrawingSheet(
                role=View.UNKNOWN,
                label="drawing",
                file=_write(tmp_path / "feedback.dxf", b"FEEDBACK_DXF"),
            ),
        )
    return FeedbackManifest(
        verification_id="verification-1",
        drawing=DrawingSource(sheets=sheets) if sheets else None,
        errors=errors or {},
    )


def _presenter(
    *,
    input_mode: str = "path",
    feedback_mode: str = "none",
) -> ArtifactPresenter:
    return ArtifactPresenter(
        input_mode=input_mode,  # type: ignore[arg-type]
        feedback_mode=feedback_mode,  # type: ignore[arg-type]
    )


@pytest.fixture
def workdir(tmp_path: Path) -> SandboxWorkdir:
    return SandboxWorkdir(host_bind_dir=tmp_path)


def _text(blocks: list[dict[str, Any]]) -> str:
    return "\n".join(block["text"] for block in blocks if block.get("type") == "text")


def test_rejects_an_unknown_input_mode() -> None:
    with pytest.raises(ValueError, match="input_mode"):
        _presenter(input_mode="none")


def test_rejects_an_unknown_feedback_mode() -> None:
    with pytest.raises(ValueError, match="feedback_mode"):
        _presenter(feedback_mode="unknown")


def test_initial_message_contains_the_drawing_path_but_not_the_drawing(
    tmp_path: Path,
    workdir: SandboxWorkdir,
) -> None:
    blocks = _presenter().build_input_message_blocks(_input_manifest(tmp_path), workdir)

    text = _text(blocks)
    assert "input.dxf" in text
    assert "RAW_DRAWING_MUST_NOT_BE_IN_PROMPT" not in text


def test_an_undivided_sheet_is_announced_as_one_carrying_every_view(
    tmp_path: Path,
    workdir: SandboxWorkdir,
) -> None:
    """Nothing has split it, so the model is told to separate the views by
    where they sit rather than by anything the file records."""
    blocks = _presenter().build_input_message_blocks(_input_manifest(tmp_path), workdir)

    text = _text(blocks)
    assert "carries every view at once" in text
    assert "not separated by layer or by file" in text


def test_split_sheets_are_named_by_role_and_narrow_the_frame(
    tmp_path: Path,
    workdir: SandboxWorkdir,
) -> None:
    """The frame sentence is in front of the model every turn, so it names the
    views the sample holds and no others."""
    manifest = _input_manifest(
        tmp_path,
        DrawingSource(
            sheets=[
                DrawingSheet(
                    role=View.FRONT, file=_write(tmp_path / "front.png", b"F")
                ),
                DrawingSheet(role=View.TOP, file=_write(tmp_path / "top.png", b"T")),
            ]
        ),
    )

    text = _text(_presenter().build_input_message_blocks(manifest, workdir))

    assert "- front:" in text and "- top:" in text
    assert "Front is right=+x" in text and "Top is right=+x" in text
    assert "Back is" not in text and "Left is" not in text
    assert "carries every view at once" not in text


def test_a_render_offered_beside_the_drawing_is_named_but_adds_no_frame(
    tmp_path: Path,
    workdir: SandboxWorkdir,
) -> None:
    """A pictorial is a sheet like any other, and `label` is what tells two of
    them apart. Nothing is lifted from it, so it contributes no axes."""
    manifest = _input_manifest(
        tmp_path,
        DrawingSource(
            sheets=[
                DrawingSheet(
                    role=View.UNKNOWN,
                    label="drawing",
                    file=_write(tmp_path / "input.dxf", b"DXF"),
                ),
                DrawingSheet(
                    role=View.PERSPECTIVE,
                    label="hlg_perspective",
                    file=_write(tmp_path / "hlg.png", b"P"),
                ),
            ],
        ),
    )

    text = _text(_presenter().build_input_message_blocks(manifest, workdir))

    assert "- hlg_perspective:" in text
    # No orthographic sheet was split out, so the fallback names all six.
    assert text.count(";") >= 5


def test_a_raster_sheet_rides_in_the_message_when_asked_for(
    tmp_path: Path,
    workdir: SandboxWorkdir,
) -> None:
    sheet = _write(tmp_path / "sheet.png", b"PNG_BYTES")
    manifest = _input_manifest(
        tmp_path,
        DrawingSource(
            sheets=[
                DrawingSheet(
                    role=View.UNKNOWN,
                    label="drawing",
                    file=sheet,
                )
            ]
        ),
    )

    blocks = _presenter(input_mode="image").build_input_message_blocks(
        manifest, workdir
    )

    images = [block for block in blocks if block.get("type") == "image"]
    assert len(images) == 1
    assert base64.b64decode(images[0]["base64"]) == b"PNG_BYTES"


def test_a_vector_sheet_is_never_attached_as_an_image(
    tmp_path: Path,
    workdir: SandboxWorkdir,
) -> None:
    """`image` is a request, not an instruction: a DXF has no pixels."""
    blocks = _presenter(input_mode="image").build_input_message_blocks(
        _input_manifest(tmp_path), workdir
    )

    assert not [block for block in blocks if block.get("type") == "image"]


def test_the_message_says_how_the_format_given_is_read(
    tmp_path: Path,
    workdir: SandboxWorkdir,
) -> None:
    """A stage's guidelines describe the job; only this message knows whether
    the job is done by reading a file or by measuring an image."""
    vector = _text(
        _presenter().build_input_message_blocks(_input_manifest(tmp_path), workdir)
    )
    raster = _text(
        _presenter().build_input_message_blocks(
            _input_manifest(
                tmp_path,
                DrawingSource(
                    sheets=[
                        DrawingSheet(
                            role=View.UNKNOWN,
                            label="drawing",
                            file=_write(tmp_path / "sheet.png", b"P"),
                        )
                    ]
                ),
            ),
            workdir,
        )
    )

    assert "ezdxf" in vector and "Measure" not in vector
    assert "Measure" in raster and "ezdxf" not in raster


def test_feedback_none_presents_nothing(
    tmp_path: Path,
    workdir: SandboxWorkdir,
) -> None:
    blocks = _presenter(feedback_mode="none").build_feedback_message_blocks(
        _feedback_manifest(tmp_path, render_count=3, include_dxf=True), workdir
    )

    assert blocks == []


def test_feedback_without_artifacts_has_no_presentation_blocks(
    tmp_path: Path,
    workdir: SandboxWorkdir,
) -> None:
    blocks = _presenter(feedback_mode="path").build_feedback_message_blocks(
        _feedback_manifest(tmp_path), workdir
    )

    assert blocks == []


@pytest.mark.parametrize("render_count", [0, 1, 2, 3])
def test_feedback_names_every_sheet_the_verification_drew(
    tmp_path: Path,
    workdir: SandboxWorkdir,
    render_count: int,
) -> None:
    manifest = _feedback_manifest(tmp_path, render_count=render_count, include_dxf=True)

    text = _text(
        _presenter(feedback_mode="path").build_feedback_message_blocks(
            manifest, workdir
        )
    )

    assert "feedback.dxf" in text
    for style in STYLES[:render_count]:
        assert f"- {style}:" in text
    for style in STYLES[render_count:]:
        assert style not in text


def test_feedback_explains_what_it_could_not_draw(
    tmp_path: Path,
    workdir: SandboxWorkdir,
) -> None:
    """A sheet that was never produced cannot carry its own explanation, so the
    reason is keyed by the label it would have had."""
    manifest = _feedback_manifest(
        tmp_path,
        render_count=1,
        errors={"style-b": "renderer failed"},
    )

    text = _text(
        _presenter(feedback_mode="path").build_feedback_message_blocks(
            manifest, workdir
        )
    )

    assert "- style-a:" in text
    assert "- style-b: unavailable (renderer failed)" in text


def test_feedback_attaches_its_renders_when_asked_for(
    tmp_path: Path,
    workdir: SandboxWorkdir,
) -> None:
    manifest = _feedback_manifest(tmp_path, render_count=2)

    blocks = _presenter(feedback_mode="image").build_feedback_message_blocks(
        manifest, workdir
    )

    images = [block for block in blocks if block.get("type") == "image"]
    assert [base64.b64decode(image["base64"]) for image in images] == [
        b"feedback-style-a",
        b"feedback-style-b",
    ]


def test_a_file_outside_the_workdir_is_refused(
    tmp_path: Path,
) -> None:
    """The sandbox namespace is what the model is given; a path outside it
    could not be opened even if it were named."""
    outside = _write(tmp_path / "outside" / "sheet.dxf", b"DXF")
    (tmp_path / "inside").mkdir(parents=True, exist_ok=True)
    workdir = SandboxWorkdir(host_bind_dir=tmp_path / "inside")
    manifest = InputManifest(
        sample_id="sample-1",
        drawing=DrawingSource(
            sheets=[
                DrawingSheet(
                    role=View.UNKNOWN,
                    label="drawing",
                    file=outside,
                )
            ]
        ),
    )

    with pytest.raises(ValueError, match="must be under"):
        _presenter().build_input_message_blocks(manifest, workdir)


def test_a_presented_sheet_carries_every_field_of_the_sheet_it_came_from() -> None:
    """Only the file changes on the way into a message, and it changes type.
    A field the sandbox copy leaves out is one a message can never say."""
    assert {field.name for field in fields(_SandboxSheet)} == set(
        DrawingSheet.model_fields
    )


def test_the_only_sheet_without_a_mime_type_is_a_vector_one() -> None:
    """`images()` skips a sheet with no mime type rather than refusing it, and
    that is exact only while the sheets it can skip are exactly the vectors."""
    assert set(_MIME_TYPES) | {".dxf"} == set(DRAWING_SUFFIXES)
