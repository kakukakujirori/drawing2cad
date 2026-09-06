"""What a manifest holds, and what it refuses to hold.

The file checks are here rather than in the drawing contract because a
`DrawingSource` is also what a stage answers with, where a path is a claim
about the sandbox rather than something the host can go and look at.
"""

from pathlib import Path

import pytest

from zeroshot.pipeline.messages import (
    DrawingSource,
    FeedbackManifest,
    InputManifest,
    View,
    unread_sheet,
)


def _write(path: Path, content: bytes = b"data") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _drawing(*files: Path) -> DrawingSource:
    return DrawingSource(
        sheets=[
            unread_sheet(f"sheet_{index}", View.FULL_PAGE, file)
            for index, file in enumerate(files)
        ]
    )


def _sample_manifest(tmp_path: Path, **overrides: object) -> InputManifest:
    values: dict[str, object] = {
        "sample_id": "sample-1",
        "drawing": _drawing(
            _write(tmp_path / "input.dxf", b"DXF"),
            _write(tmp_path / "input.png", b"PNG"),
        ),
    }
    values.update(overrides)
    return InputManifest(**values)  # type: ignore[arg-type]


def _feedback_manifest(tmp_path: Path, **overrides: object) -> FeedbackManifest:
    values: dict[str, object] = {"verification_id": "verification-1"}
    values.update(overrides)
    return FeedbackManifest(**values)  # type: ignore[arg-type]


def test_a_sample_keeps_every_sheet_it_was_given(tmp_path: Path) -> None:
    manifest = _sample_manifest(tmp_path, sample_id="  sample-1  ")

    assert manifest.sample_id == "sample-1"
    assert [path.name for path in manifest.drawing.paths()] == [
        "input.dxf",
        "input.png",
    ]


@pytest.mark.parametrize("sample_id", ["", "   ", ".", "..", "a/b", r"a\b"])
def test_a_sample_refuses_an_empty_or_unsafe_id(tmp_path: Path, sample_id: str) -> None:
    with pytest.raises(ValueError):
        _sample_manifest(tmp_path, sample_id=sample_id)


def test_a_sample_refuses_a_sheet_whose_file_is_not_there(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _sample_manifest(tmp_path, drawing=_drawing(tmp_path / "missing.dxf"))


def test_a_verification_that_drew_nothing_is_a_manifest_too(tmp_path: Path) -> None:
    manifest = _feedback_manifest(tmp_path, verification_id="  verification-1  ")

    assert manifest.verification_id == "verification-1"
    assert manifest.drawing is None
    assert manifest.errors == {}


@pytest.mark.parametrize("verification_id", ["", "   ", ".", "..", "a/b", r"a\b"])
def test_a_verification_refuses_an_empty_or_unsafe_id(
    tmp_path: Path, verification_id: str
) -> None:
    with pytest.raises(ValueError):
        _feedback_manifest(tmp_path, verification_id=verification_id)


def test_a_verification_refuses_a_sheet_whose_file_is_not_there(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        _feedback_manifest(tmp_path, drawing=_drawing(tmp_path / "missing.dxf"))


def test_a_sheet_is_either_drawn_or_explained_but_never_both(tmp_path: Path) -> None:
    """A file and a reason are alternatives; holding both means a wiring bug."""
    drawn = _drawing(_write(tmp_path / "feedback.dxf", b"DXF"))
    (name,) = (sheet.name for sheet in drawn.sheets)

    # Either alone is a legitimate outcome.
    assert _feedback_manifest(tmp_path, drawing=drawn).errors == {}
    assert (
        _feedback_manifest(tmp_path, errors={name: "renderer failed"}).drawing is None
    )

    with pytest.raises(ValueError, match="both drawn and failed"):
        _feedback_manifest(tmp_path, drawing=drawn, errors={name: "renderer failed"})
