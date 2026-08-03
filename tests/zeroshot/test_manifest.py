from pathlib import Path

import pytest

from zeroshot.pipeline.manifest import FeedbackManifest, InputManifest


def _write(path: Path, content: bytes = b"data") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _sample_manifest(tmp_path: Path, **overrides: object) -> InputManifest:
    values: dict[str, object] = {
        "sample_id": "sample-1",
        "dxf_path": _write(tmp_path / "input.dxf", b"DXF"),
        "render3d_paths": {
            "style-a": _write(tmp_path / "input-a.png", b"input-a"),
            "future-style": _write(
                tmp_path / "input-future.png",
                b"input-future",
            ),
        },
    }
    values.update(overrides)
    return InputManifest(**values)


def _feedback_manifest(tmp_path: Path, **overrides: object) -> FeedbackManifest:
    values: dict[str, object] = {
        "verification_id": "verification-1",
        "execution_feedback": "execution completed",
    }
    values.update(overrides)
    return FeedbackManifest(**values)


def test_sample_normalizes_id_and_string_paths(tmp_path: Path) -> None:
    dxf = _write(tmp_path / "input.dxf")
    render = _write(tmp_path / "input.png")

    manifest = InputManifest(
        sample_id="  sample-1  ",
        dxf_path=str(dxf),
        render3d_paths={"arbitrary-future-style": str(render)},
    )

    assert manifest.sample_id == "sample-1"
    assert manifest.dxf_path == dxf
    assert manifest.render3d_paths == {
        "arbitrary-future-style": render,
    }


@pytest.mark.parametrize("sample_id", ["", "   ", ".", "..", "a/b", r"a\b"])
def test_sample_rejects_empty_or_unsafe_id(
    tmp_path: Path,
    sample_id: str,
) -> None:
    with pytest.raises(ValueError):
        _sample_manifest(tmp_path, sample_id=sample_id)


def test_sample_rejects_missing_dxf(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _sample_manifest(tmp_path, dxf_path=tmp_path / "missing.dxf")


def test_sample_rejects_non_dxf_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        _sample_manifest(
            tmp_path,
            dxf_path=_write(tmp_path / "input.txt"),
        )


def test_sample_rejects_missing_render(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _sample_manifest(
            tmp_path,
            render3d_paths={"style-a": tmp_path / "missing.png"},
        )


def test_sample_rejects_none_as_render_value(tmp_path: Path) -> None:
    with pytest.raises((TypeError, ValueError)):
        _sample_manifest(
            tmp_path,
            render3d_paths={"style-a": None},
        )


def test_sample_render_mapping_is_immutable(tmp_path: Path) -> None:
    manifest = _sample_manifest(tmp_path)

    with pytest.raises(TypeError):
        manifest.render3d_paths["new-style"] = tmp_path / "new.png"  # type: ignore[index]


def test_sample_reads_selected_render_bytes_in_requested_order(
    tmp_path: Path,
) -> None:
    render_a = _write(tmp_path / "input-a.png", b"input-a")
    render_b = _write(tmp_path / "input-b.png", b"input-b")
    manifest = _sample_manifest(
        tmp_path,
        render3d_paths={"a": render_a, "b": render_b},
    )

    assert list(manifest.load_render3d(["b", "a"]).items()) == [
        ("b", b"input-b"),
        ("a", b"input-a"),
    ]


def test_feedback_accepts_execution_feedback_without_artifacts(
    tmp_path: Path,
) -> None:
    manifest = _feedback_manifest(tmp_path)

    assert manifest.verification_id == "verification-1"
    assert manifest.execution_feedback == "execution completed"
    assert manifest.dxf_path is None
    assert manifest.render3d_paths == {}


def test_feedback_normalizes_id_and_string_paths(tmp_path: Path) -> None:
    dxf = _write(tmp_path / "feedback.dxf")
    render = _write(tmp_path / "feedback.png")

    manifest = FeedbackManifest(
        verification_id="  verification-1  ",
        execution_feedback="verified",
        dxf_path=str(dxf),
        render3d_paths={"style-a": str(render)},
    )

    assert manifest.verification_id == "verification-1"
    assert manifest.execution_feedback == "verified"
    assert manifest.dxf_path == dxf
    assert manifest.render3d_paths == {"style-a": render}


@pytest.mark.parametrize(
    "verification_id",
    ["", "   ", ".", "..", "a/b", r"a\b"],
)
def test_feedback_rejects_empty_or_unsafe_verification_id(
    tmp_path: Path,
    verification_id: str,
) -> None:
    with pytest.raises(ValueError):
        _feedback_manifest(
            tmp_path,
            verification_id=verification_id,
        )


def test_feedback_rejects_missing_dxf(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _feedback_manifest(
            tmp_path,
            dxf_path=tmp_path / "missing.dxf",
        )


def test_feedback_rejects_non_dxf_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        _feedback_manifest(
            tmp_path,
            dxf_path=_write(tmp_path / "feedback.txt"),
        )


def test_feedback_rejects_missing_render(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _feedback_manifest(
            tmp_path,
            render3d_paths={"style-a": tmp_path / "missing.png"},
        )


def test_feedback_rejects_none_as_render_value(tmp_path: Path) -> None:
    with pytest.raises((TypeError, ValueError)):
        _feedback_manifest(
            tmp_path,
            render3d_paths={"style-a": None},
        )


def test_feedback_render_mapping_is_immutable(tmp_path: Path) -> None:
    manifest = _feedback_manifest(
        tmp_path,
        render3d_paths={
            "style-a": _write(tmp_path / "feedback-a.png"),
        },
    )

    with pytest.raises(TypeError):
        manifest.render3d_paths["new-style"] = tmp_path / "new.png"  # type: ignore[index]


def test_feedback_reads_selected_render_bytes_in_requested_order(
    tmp_path: Path,
) -> None:
    render_a = _write(tmp_path / "feedback-a.png", b"feedback-a")
    render_b = _write(tmp_path / "feedback-b.png", b"feedback-b")
    manifest = _feedback_manifest(
        tmp_path,
        render3d_paths={"a": render_a, "b": render_b},
    )

    assert list(manifest.load_render3d(["b", "a"]).items()) == [
        ("b", b"feedback-b"),
        ("a", b"feedback-a"),
    ]


def test_feedback_manifest_rejects_an_artifact_that_is_both_present_and_failed(
    tmp_path: Path,
) -> None:
    """A path and a reason are alternatives; holding both means a wiring bug."""
    dxf = _write(tmp_path / "feedback.dxf", b"DXF")
    png = _write(tmp_path / "feedback-a.png", b"PNG")

    # Either alone is a legitimate outcome.
    assert _feedback_manifest(tmp_path, dxf_path=dxf).dxf_error is None
    assert _feedback_manifest(tmp_path, dxf_error="renderer failed").dxf_path is None

    with pytest.raises(ValueError, match="dxf_path and dxf_error are both set"):
        _feedback_manifest(tmp_path, dxf_path=dxf, dxf_error="renderer failed")

    with pytest.raises(ValueError, match="both rendered and failed"):
        _feedback_manifest(
            tmp_path,
            render3d_paths={"style-a": png},
            render3d_errors={"style-a": "renderer failed"},
        )
