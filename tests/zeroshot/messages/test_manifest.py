"""What a sample is handed in as, and what a verification hands back.

Both are a `DrawingSource`, so the cases below are the same cases twice: files
that have to exist, an id that has to be safe to make a directory from, and --
on the way back -- an artifact that is either drawn or explained, never both.
"""

from pathlib import Path

import pytest

from zeroshot.pipeline.messages import (
    DrawingSheet,
    DrawingSource,
    FeedbackManifest,
    InputManifest,
    View,
)


def _write(path: Path, content: bytes = b"data") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _sample_manifest(tmp_path: Path, **overrides: object) -> InputManifest:
    values: dict[str, object] = {
        "sample_id": "sample-1",
        "drawing": DrawingSource(
            sheets=[
                DrawingSheet(
                    role=View.UNKNOWN,
                    label="drawing",
                    file=_write(tmp_path / "input.dxf", b"DXF"),
                )
            ]
        ),
    }
    values.update(overrides)
    return InputManifest(**values)


def _feedback_manifest(tmp_path: Path, **overrides: object) -> FeedbackManifest:
    values: dict[str, object] = {"verification_id": "verification-1"}
    values.update(overrides)
    return FeedbackManifest(**values)


def test_sample_normalizes_its_id(tmp_path: Path) -> None:
    dxf = _write(tmp_path / "input.dxf")

    manifest = InputManifest(
        sample_id="  sample-1  ",
        drawing=DrawingSource(
            sheets=[
                DrawingSheet(
                    role=View.UNKNOWN,
                    label="drawing",
                    file=str(dxf),
                )
            ]
        ),
    )

    assert manifest.sample_id == "sample-1"
    assert manifest.drawing.paths() == [dxf]


@pytest.mark.parametrize("sample_id", ["", "   ", ".", "..", "a/b", r"a\b"])
def test_sample_rejects_empty_or_unsafe_id(
    tmp_path: Path,
    sample_id: str,
) -> None:
    with pytest.raises(ValueError):
        _sample_manifest(tmp_path, sample_id=sample_id)


def test_sample_rejects_a_missing_drawing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _sample_manifest(
            tmp_path,
            drawing=DrawingSource(
                sheets=[
                    DrawingSheet(
                        role=View.UNKNOWN,
                        label="drawing",
                        file=tmp_path / "missing.dxf",
                    )
                ]
            ),
        )


def test_sample_rejects_a_missing_sheet(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _sample_manifest(
            tmp_path,
            drawing=DrawingSource(
                sheets=[DrawingSheet(role=View.FRONT, file=tmp_path / "gone.png")]
            ),
        )


def test_sample_takes_a_drawing_split_into_sheets_of_either_format(
    tmp_path: Path,
) -> None:
    front = _write(tmp_path / "front.dxf", b"DXF")
    top = _write(tmp_path / "top.png", b"PNG")

    manifest = _sample_manifest(
        tmp_path,
        drawing=DrawingSource(
            sheets=[
                DrawingSheet(role=View.FRONT, file=front),
                DrawingSheet(role=View.TOP, file=top),
            ]
        ),
    )

    assert manifest.drawing.paths() == [front, top]
    assert manifest.drawing.views() == (View.FRONT, View.TOP)


def test_a_render_offered_with_the_drawing_is_a_sheet_like_any_other(
    tmp_path: Path,
) -> None:
    """A perspective render has a role the vocabulary already names, so it
    needs no container of its own; `label` says which rendering it is."""
    render = _write(tmp_path / "hlg.png", b"PNG")

    manifest = _sample_manifest(
        tmp_path,
        drawing=DrawingSource(
            sheets=[
                DrawingSheet(
                    role=View.UNKNOWN,
                    label="drawing",
                    file=_write(tmp_path / "input.dxf", b"DXF"),
                ),
                DrawingSheet(
                    role=View.PERSPECTIVE, label="hlg_perspective", file=render
                ),
            ],
        ),
    )

    # A pictorial has no frame, so nothing is lifted from it.
    assert manifest.drawing.orthographic() == []
    assert render in manifest.drawing.paths()


def test_feedback_accepts_a_verification_that_drew_nothing(tmp_path: Path) -> None:
    manifest = _feedback_manifest(tmp_path)

    assert manifest.verification_id == "verification-1"
    assert manifest.drawing is None
    assert manifest.errors == {}


def test_feedback_normalizes_its_id(tmp_path: Path) -> None:
    dxf = _write(tmp_path / "feedback.dxf")

    manifest = FeedbackManifest(
        verification_id="  verification-1  ",
        drawing=DrawingSource(
            sheets=[
                DrawingSheet(
                    role=View.UNKNOWN,
                    label="drawing",
                    file=dxf,
                )
            ]
        ),
    )

    assert manifest.verification_id == "verification-1"


@pytest.mark.parametrize(
    "verification_id",
    ["", "   ", ".", "..", "a/b", r"a\b"],
)
def test_feedback_rejects_empty_or_unsafe_verification_id(
    tmp_path: Path,
    verification_id: str,
) -> None:
    with pytest.raises(ValueError):
        _feedback_manifest(tmp_path, verification_id=verification_id)


def test_feedback_rejects_a_drawing_it_cannot_open(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _feedback_manifest(
            tmp_path,
            drawing=DrawingSource(
                sheets=[
                    DrawingSheet(
                        role=View.UNKNOWN,
                        label="drawing",
                        file=tmp_path / "missing.dxf",
                    )
                ]
            ),
        )


def test_feedback_errors_are_immutable(tmp_path: Path) -> None:
    manifest = _feedback_manifest(tmp_path, errors={"drawing": "renderer failed"})

    with pytest.raises(TypeError):
        manifest.errors["another"] = "boom"  # type: ignore[index]


def test_a_sheet_is_either_drawn_or_explained_never_both(tmp_path: Path) -> None:
    """A path and a reason are alternatives; holding both means a wiring bug."""
    render = _write(tmp_path / "hlg.png", b"PNG")
    drawn = DrawingSource(
        sheets=[
            DrawingSheet(role=View.PERSPECTIVE, label="hlg_perspective", file=render)
        ]
    )

    # Either alone is a legitimate outcome.
    assert _feedback_manifest(tmp_path, drawing=drawn).errors == {}
    assert (
        _feedback_manifest(tmp_path, errors={"hlg_perspective": "boom"}).drawing is None
    )

    with pytest.raises(ValueError, match="both drawn and failed"):
        _feedback_manifest(
            tmp_path,
            drawing=drawn,
            errors={"hlg_perspective": "renderer failed"},
        )
