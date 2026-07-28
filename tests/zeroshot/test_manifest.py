from pathlib import Path

import pytest

from zeroshot.pipeline.manifest import SampleManifest


def _write(path: Path, content: bytes = b"data") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _manifest(tmp_path: Path, **overrides: object) -> SampleManifest:
    values: dict[str, object] = {
        "sample_id": "sample-1",
        "input_dxf_path": _write(tmp_path / "input.dxf", b"DXF"),
        "input_render3d_paths": {
            "style-a": _write(tmp_path / "input-a.png", b"input-a"),
            "future-style": _write(tmp_path / "input-future.png", b"input-future"),
        },
    }
    values.update(overrides)
    return SampleManifest(**values)


def test_normalizes_id_and_string_paths(tmp_path: Path) -> None:
    dxf = _write(tmp_path / "input.dxf")
    render = _write(tmp_path / "input.png")

    manifest = SampleManifest(
        sample_id="  sample-1  ",
        input_dxf_path=str(dxf),
        input_render3d_paths={"arbitrary-future-style": str(render)},
    )

    assert manifest.sample_id == "sample-1"
    assert manifest.input_dxf_path == dxf
    assert manifest.input_render3d_paths == {
        "arbitrary-future-style": render,
    }
    assert manifest.feedback_dxf_path is None
    assert manifest.feedback_render3d_paths == {}


@pytest.mark.parametrize("sample_id", ["", "   ", ".", "..", "a/b", r"a\b"])
def test_rejects_empty_or_unsafe_sample_id(
    tmp_path: Path,
    sample_id: str,
) -> None:
    with pytest.raises(ValueError):
        _manifest(tmp_path, sample_id=sample_id)


def test_rejects_missing_input_dxf(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _manifest(tmp_path, input_dxf_path=tmp_path / "missing.dxf")


def test_rejects_non_dxf_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        _manifest(tmp_path, input_dxf_path=_write(tmp_path / "input.txt"))


def test_rejects_missing_input_render(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _manifest(
            tmp_path,
            input_render3d_paths={"style-a": tmp_path / "missing.png"},
        )


def test_accepts_optional_feedback_artifacts(tmp_path: Path) -> None:
    feedback_dxf = _write(tmp_path / "feedback.dxf", b"feedback-dxf")
    feedback_render = _write(tmp_path / "feedback.png", b"feedback-render")

    manifest = _manifest(
        tmp_path,
        feedback_dxf_path=feedback_dxf,
        feedback_render3d_paths={"style-a": feedback_render},
    )

    assert manifest.feedback_dxf_path == feedback_dxf
    assert manifest.feedback_render3d_paths == {"style-a": feedback_render}


def test_feedback_styles_must_come_from_input_styles(tmp_path: Path) -> None:
    feedback_render = _write(tmp_path / "feedback.png", b"feedback-render")

    with pytest.raises(ValueError, match="subset of input styles"):
        _manifest(
            tmp_path,
            feedback_render3d_paths={"feedback-only-style": feedback_render},
        )


def test_rejects_missing_feedback_artifacts(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _manifest(tmp_path, feedback_dxf_path=tmp_path / "missing.dxf")

    with pytest.raises(FileNotFoundError):
        _manifest(
            tmp_path,
            feedback_render3d_paths={"style-a": tmp_path / "missing.png"},
        )


def test_rejects_none_as_feedback_render_value(tmp_path: Path) -> None:
    with pytest.raises((TypeError, ValueError)):
        _manifest(
            tmp_path,
            feedback_render3d_paths={"style-a": None},
        )


def test_reads_selected_render_bytes_in_requested_order(tmp_path: Path) -> None:
    input_a = _write(tmp_path / "input-a.png", b"input-a")
    input_b = _write(tmp_path / "input-b.png", b"input-b")
    feedback_a = _write(tmp_path / "feedback-a.png", b"feedback-a")
    feedback_b = _write(tmp_path / "feedback-b.png", b"feedback-b")
    manifest = _manifest(
        tmp_path,
        input_render3d_paths={"a": input_a, "b": input_b},
        feedback_render3d_paths={"a": feedback_a, "b": feedback_b},
    )

    assert list(manifest.load_input_render3d(["b", "a"]).items()) == [
        ("b", b"input-b"),
        ("a", b"input-a"),
    ]
    assert list(manifest.load_feedback_render3d(["b", "a"]).items()) == [
        ("b", b"feedback-b"),
        ("a", b"feedback-a"),
    ]
