"""What a manifest holds, and what it refuses to hold.

The file checks are here rather than in the drawing contract because a
sheet a stage answers with names a path in the sandbox, not one the host
can go and look at.
"""

from pathlib import Path

import pytest
from PIL import Image

from zeroshot.pipeline.messages.manifest import (
    FeedbackManifest,
    InputManifest,
    register_view,
)
from zeroshot.pipeline.stages.interpretation.contracts import (
    DrawingView,
    Region,
    View,
)


def _write(path: Path, content: bytes = b"data") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".png":
        Image.new("RGB", (10, 10)).save(path)
    else:
        path.write_bytes(content)
    return path


def _drawing(*files: Path) -> list[DrawingView]:
    return [
        DrawingView(
            name=f"view_{index}",
            role=View.FULL_PAGE,
            file=str(file),
            region=Region(view=f"view_{index}", box_uv=(0.0, 0.0, 10.0, 10.0)),
            dimensions=[],
        )
        for index, file in enumerate(files)
    ]


def _rendered(*files: Path) -> list[DrawingView]:
    return [
        DrawingView(
            name=f"view_projected_{index}",
            role=View.FRONT,
            file=str(file),
            region=Region(
                view=f"view_projected_{index}", box_uv=(0.0, 0.0, 10.0, 10.0)
            ),
            dimensions=[],
            u_axis="+x",
            v_axis="+z",
        )
        for index, file in enumerate(files)
    ]


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
    assert [Path(view.file).name for view in manifest.drawing] == [
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


def test_a_sample_rejects_duplicate_names_before_inputs_can_be_copied(tmp_path):
    first = _drawing(_write(tmp_path / "first.png"))[0]
    second = _drawing(_write(tmp_path / "second.png"))[0]
    with pytest.raises(ValueError, match="duplicate names in input views: view_0"):
        InputManifest(sample_id="sample", drawing=[first, second])


def test_registration_rejects_images_the_presenter_cannot_attach(tmp_path):
    path = tmp_path / "input.webp"
    Image.new("RGB", (20, 10)).save(path)
    with pytest.raises(ValueError, match="unsupported drawing file"):
        register_view("view_input", View.FULL_PAGE, path)
    with pytest.raises(ValueError, match="unsupported drawing file"):
        InputManifest(sample_id="sample", drawing=_drawing(path))


@pytest.mark.parametrize(
    "size, limit, rejected",
    [
        ((2000, 10), 2000, False),
        ((10, 2000), 2000, False),
        ((2001, 10), 2000, True),
        ((10, 2001), 2000, True),
        ((2001, 10), 3000, False),
        ((11, 10), 10, True),
    ],
)
def test_input_image_size_limit_preserves_pixels(tmp_path, size, limit, rejected):
    image = tmp_path / "input.png"
    Image.new("RGB", size).save(image)
    original = image.read_bytes()
    # Vector input does not acquire a pixel limit.
    dxf = _write(tmp_path / "drawing.dxf", b"DXF")
    if rejected:
        with pytest.raises(ValueError, match=f"sample.max_input_image_side={limit}"):
            InputManifest(
                sample_id="sample",
                drawing=_drawing(dxf, image),
                max_input_image_side=limit,
            )
    else:
        manifest = InputManifest(
            sample_id="sample", drawing=_drawing(dxf, image), max_input_image_side=limit
        )
        assert manifest.max_input_image_side == limit
    assert image.read_bytes() == original


def test_a_verification_that_drew_nothing_is_a_manifest_too(tmp_path: Path) -> None:
    manifest = _feedback_manifest(tmp_path, verification_id="  verification-1  ")

    assert manifest.verification_id == "verification-1"
    assert manifest.drawing == ()
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
        _feedback_manifest(tmp_path, drawing=_rendered(tmp_path / "missing.dxf"))


def test_a_sheet_is_either_drawn_or_explained_but_never_both(tmp_path: Path) -> None:
    """A file and a reason are alternatives; holding both means a wiring bug."""
    drawn = _rendered(_write(tmp_path / "feedback.dxf", b"DXF"))
    (name,) = (sheet.name for sheet in drawn)

    # Either alone is a legitimate outcome.
    assert _feedback_manifest(tmp_path, drawing=drawn).errors == {}
    assert _feedback_manifest(tmp_path, errors={name: "renderer failed"}).drawing == ()

    with pytest.raises(ValueError, match="both drawn and failed"):
        _feedback_manifest(tmp_path, drawing=drawn, errors={name: "renderer failed"})
