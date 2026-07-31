import base64
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from zeroshot.pipeline.manifest import FeedbackManifest, InputManifest
from zeroshot.pipeline.messages import DEFAULT_SYSTEM_PROMPT, MessageBuilder
from zeroshot.pipeline.sandbox import SandboxWorkdir

STYLES = ("style-a", "style-b", "style-c")


def _write(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _input_manifest(tmp_path: Path) -> InputManifest:
    return InputManifest(
        sample_id="sample-1",
        dxf_path=_write(
            tmp_path / "input.dxf",
            b"RAW_DXF_MUST_NOT_BE_IN_PROMPT",
        ),
        render3d_paths={
            style: _write(
                tmp_path / f"input-{index}.png",
                f"input-{style}".encode(),
            )
            for index, style in enumerate(STYLES)
        },
    )


def _feedback_manifest(
    tmp_path: Path,
    *,
    execution_feedback: str = "execution result",
    render_count: int = 0,
    include_dxf: bool = False,
) -> FeedbackManifest:
    return FeedbackManifest(
        verification_id="verification-1",
        execution_feedback=execution_feedback,
        dxf_path=(
            _write(tmp_path / "feedback.dxf", b"FEEDBACK_DXF") if include_dxf else None
        ),
        render3d_paths={
            style: _write(
                tmp_path / f"feedback-{index}.png",
                f"feedback-{style}".encode(),
            )
            for index, style in enumerate(STYLES[:render_count])
        },
    )


def _builder(
    *,
    access_mode: str = "none",
    access_styles: tuple[str, ...] = (),
    feedback_mode: str = "none",
    feedback_styles: tuple[str, ...] = (),
) -> MessageBuilder:
    return MessageBuilder(
        access_render3d=access_mode,
        access_render3d_styles=access_styles,
        feedback_render3d=feedback_mode,
        feedback_render3d_styles=feedback_styles,
    )


@pytest.fixture
def workdir(tmp_path: Path) -> SandboxWorkdir:
    return SandboxWorkdir(host_bind_dir=tmp_path)


def _blocks(message: HumanMessage) -> list[dict[str, Any]]:
    assert isinstance(message.content, list)
    return message.content


def _text(blocks: list[dict[str, Any]]) -> str:
    return "\n".join(block["text"] for block in blocks if block.get("type") == "text")


def test_default_system_prompt_does_not_name_tools() -> None:
    assert "perspective renders" in DEFAULT_SYSTEM_PROMPT
    assert "isometric views" not in DEFAULT_SYSTEM_PROMPT
    for tool_name in (
        "run_python",
        "execute_cad_candidate",
        "submit_final_candidate",
    ):
        assert tool_name not in DEFAULT_SYSTEM_PROMPT


@pytest.mark.parametrize("field", ["access", "feedback"])
def test_rejects_unknown_mode(field: str) -> None:
    kwargs = {f"{field}_mode": "unknown", f"{field}_styles": ("style-a",)}
    with pytest.raises(ValueError):
        _builder(**kwargs)


@pytest.mark.parametrize("field", ["access", "feedback"])
def test_rejects_duplicate_styles(field: str) -> None:
    kwargs = {
        f"{field}_mode": "image",
        f"{field}_styles": ("style-a", "style-a"),
    }
    with pytest.raises(ValueError):
        _builder(**kwargs)


@pytest.mark.parametrize("field", ["access", "feedback"])
@pytest.mark.parametrize("mode", ["path", "image"])
def test_non_none_mode_requires_at_least_one_style(
    field: str,
    mode: str,
) -> None:
    with pytest.raises(ValueError):
        _builder(**{f"{field}_mode": mode, f"{field}_styles": ()})


def test_initial_message_always_contains_dxf_path_but_not_raw_dxf(
    tmp_path: Path,
    workdir: SandboxWorkdir,
) -> None:
    manifest = _input_manifest(tmp_path)
    messages = _builder().build_initial(manifest, workdir)

    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    text = _text(_blocks(messages[1]))
    assert str(workdir.host_to_sandbox_path(manifest.dxf_path)) in text
    assert str(manifest.dxf_path) not in text
    assert "RAW_DXF_MUST_NOT_BE_IN_PROMPT" not in text


def test_initial_none_has_no_sample_specific_render_information(
    tmp_path: Path,
    workdir: SandboxWorkdir,
) -> None:
    manifest = _input_manifest(tmp_path)
    human = _builder().build_initial(manifest, workdir)[1]
    blocks = _blocks(human)
    text = _text(blocks)

    assert [block["type"] for block in blocks] == ["text"]
    for style, path in manifest.render3d_paths.items():
        assert style not in text
        assert str(path) not in text
        assert str(workdir.host_to_sandbox_path(path)) not in text


def test_initial_path_includes_only_selected_styles_in_order(
    tmp_path: Path,
    workdir: SandboxWorkdir,
) -> None:
    manifest = _input_manifest(tmp_path)
    selected = ("style-c", "style-a")
    human = _builder(
        access_mode="path",
        access_styles=selected,
    ).build_initial(manifest, workdir)[1]
    text = _text(_blocks(human))

    assert text.index("style-c") < text.index("style-a")
    for style in selected:
        host_path = manifest.render3d_paths[style]
        assert str(workdir.host_to_sandbox_path(host_path)) in text
        assert str(host_path) not in text
    assert "style-b" not in text
    assert not any(block["type"] == "image" for block in _blocks(human))


def test_initial_image_interleaves_style_labels_and_images(
    tmp_path: Path,
    workdir: SandboxWorkdir,
) -> None:
    manifest = _input_manifest(tmp_path)
    selected = ("style-b", "style-a")
    human = _builder(
        access_mode="image",
        access_styles=selected,
    ).build_initial(manifest, workdir)[1]
    blocks = _blocks(human)

    assert [block["type"] for block in blocks] == [
        "text",
        "text",
        "text",
        "image",
        "text",
        "image",
    ]
    for style, label_index, image_index in (
        ("style-b", 2, 3),
        ("style-a", 4, 5),
    ):
        assert style in blocks[label_index]["text"]
        assert base64.b64decode(blocks[image_index]["base64"]) == (
            manifest.render3d_paths[style].read_bytes()
        )
        assert blocks[image_index]["mime_type"] == "image/png"


def test_initial_rejects_access_style_missing_from_input_manifest(
    tmp_path: Path,
    workdir: SandboxWorkdir,
) -> None:
    manifest = _input_manifest(tmp_path)

    with pytest.raises(ValueError, match="Unknown styles"):
        _builder(
            access_mode="image",
            access_styles=("missing-style",),
        ).build_initial(manifest, workdir)


def test_initial_rejects_feedback_style_missing_from_input_manifest(
    tmp_path: Path,
    workdir: SandboxWorkdir,
) -> None:
    manifest = _input_manifest(tmp_path)

    with pytest.raises(ValueError, match="Unknown styles"):
        _builder(
            feedback_mode="image",
            feedback_styles=("missing-style",),
        ).build_initial(manifest, workdir)


def test_feedback_always_contains_execution_feedback(
    tmp_path: Path,
    workdir: SandboxWorkdir,
) -> None:
    manifest = _feedback_manifest(
        tmp_path,
        execution_feedback="syntax error on line 3",
    )
    message = _builder().build_feedback(manifest, workdir)

    assert "syntax error on line 3" in _text(_blocks(message))


def test_feedback_includes_projected_dxf_only_when_available(
    tmp_path: Path,
    workdir: SandboxWorkdir,
) -> None:
    without_dxf = _builder().build_feedback(_feedback_manifest(tmp_path), workdir)
    assert "Projected DXF path" not in _text(_blocks(without_dxf))

    with_dxf_manifest = _feedback_manifest(tmp_path, include_dxf=True)
    with_dxf = _builder().build_feedback(with_dxf_manifest, workdir)
    text = _text(_blocks(with_dxf))
    assert "Projected DXF path" in text
    assert with_dxf_manifest.dxf_path is not None
    assert str(workdir.host_to_sandbox_path(with_dxf_manifest.dxf_path)) in text
    assert str(with_dxf_manifest.dxf_path) not in text


@pytest.mark.parametrize("mode", ["path", "image"])
@pytest.mark.parametrize("render_count", [0, 1, 2, 3])
def test_feedback_includes_only_available_renders(
    tmp_path: Path,
    workdir: SandboxWorkdir,
    mode: str,
    render_count: int,
) -> None:
    input_manifest = _input_manifest(tmp_path)
    feedback_manifest = _feedback_manifest(
        tmp_path,
        render_count=render_count,
    )
    builder = _builder(
        feedback_mode=mode,
        feedback_styles=STYLES,
    )
    builder.build_initial(input_manifest, workdir)

    message = builder.build_feedback(feedback_manifest, workdir)
    blocks = _blocks(message)
    text = _text(blocks)

    if render_count == 0:
        assert "Projected perspective renders" not in text
        assert "attached images" not in text
    else:
        assert "Projected perspective renders" in text

    for style in STYLES[:render_count]:
        assert style in text
        host_path = feedback_manifest.render3d_paths[style]
        if mode == "path":
            assert str(workdir.host_to_sandbox_path(host_path)) in text
            assert str(host_path) not in text
    for style in STYLES[render_count:]:
        assert style not in text

    image_blocks = [block for block in blocks if block["type"] == "image"]
    expected_image_count = render_count if mode == "image" else 0
    assert len(image_blocks) == expected_image_count

    if mode == "image":
        assert [base64.b64decode(block["base64"]) for block in image_blocks] == [
            feedback_manifest.render3d_paths[style].read_bytes()
            for style in STYLES[:render_count]
        ]


def test_access_and_feedback_style_selections_do_not_mix(
    tmp_path: Path,
    workdir: SandboxWorkdir,
) -> None:
    input_manifest = _input_manifest(tmp_path)
    feedback_manifest = _feedback_manifest(tmp_path, render_count=3)
    builder = _builder(
        access_mode="path",
        access_styles=("style-a",),
        feedback_mode="path",
        feedback_styles=("style-c",),
    )

    initial_text = _text(_blocks(builder.build_initial(input_manifest, workdir)[1]))
    feedback_text = _text(_blocks(builder.build_feedback(feedback_manifest, workdir)))

    assert "style-a" in initial_text
    assert "style-c" not in initial_text
    assert "style-c" in feedback_text
    assert "style-a" not in feedback_text
