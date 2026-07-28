import base64
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from zeroshot.pipeline.manifest import SampleManifest
from zeroshot.pipeline.messages import DEFAULT_SYSTEM_PROMPT, MessageBuilder

STYLES = ("style-a", "style-b", "style-c")


def _write(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _manifest(
    tmp_path: Path,
    *,
    feedback_count: int = 0,
    feedback_dxf: bool = False,
) -> SampleManifest:
    input_dxf = _write(tmp_path / "input.dxf", b"RAW_DXF_MUST_NOT_BE_IN_PROMPT")
    input_renders = {
        style: _write(tmp_path / f"input-{index}.png", f"input-{style}".encode())
        for index, style in enumerate(STYLES)
    }
    feedback_renders = {
        style: _write(
            tmp_path / f"feedback-{index}.png",
            f"feedback-{style}".encode(),
        )
        for index, style in enumerate(STYLES[:feedback_count])
    }
    feedback_dxf_path = (
        _write(tmp_path / "feedback.dxf", b"FEEDBACK_DXF") if feedback_dxf else None
    )
    return SampleManifest(
        sample_id="sample-1",
        input_dxf_path=input_dxf,
        input_render3d_paths=input_renders,
        feedback_dxf_path=feedback_dxf_path,
        feedback_render3d_paths=feedback_renders,
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
) -> None:
    manifest = _manifest(tmp_path)
    messages = _builder().build_initial(manifest)

    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    text = _text(_blocks(messages[1]))
    assert str(manifest.input_dxf_path) in text
    assert "RAW_DXF_MUST_NOT_BE_IN_PROMPT" not in text


def test_initial_none_has_no_sample_specific_render_information(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    human = _builder().build_initial(manifest)[1]
    blocks = _blocks(human)
    text = _text(blocks)

    assert [block["type"] for block in blocks] == ["text"]
    for style, path in manifest.input_render3d_paths.items():
        assert style not in text
        assert str(path) not in text


def test_initial_path_includes_only_selected_styles_in_order(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    selected = ("style-c", "style-a")
    human = _builder(
        access_mode="path",
        access_styles=selected,
    ).build_initial(manifest)[1]
    text = _text(_blocks(human))

    assert text.index("style-c") < text.index("style-a")
    for style in selected:
        assert str(manifest.input_render3d_paths[style]) in text
    assert "style-b" not in text
    assert not any(block["type"] == "image" for block in _blocks(human))


def test_initial_image_interleaves_style_labels_and_images(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    selected = ("style-b", "style-a")
    human = _builder(
        access_mode="image",
        access_styles=selected,
    ).build_initial(manifest)[1]
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
            manifest.input_render3d_paths[style].read_bytes()
        )
        assert blocks[image_index]["mime_type"] == "image/png"


def test_initial_rejects_style_missing_from_manifest(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    with pytest.raises(ValueError, match="Unknown styles"):
        _builder(
            access_mode="image",
            access_styles=("missing-style",),
        ).build_initial(manifest)


def test_feedback_always_contains_execution_feedback(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    message = _builder().build_feedback("syntax error on line 3", manifest)

    assert "syntax error on line 3" in _text(_blocks(message))


def test_feedback_rejects_style_missing_from_input_manifest(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    with pytest.raises(ValueError, match="Unknown styles"):
        _builder(
            feedback_mode="image",
            feedback_styles=("missing-style",),
        ).build_feedback("execution result", manifest)


def test_feedback_includes_projected_dxf_only_when_available(
    tmp_path: Path,
) -> None:
    without_dxf = _builder().build_feedback("failed", _manifest(tmp_path))
    assert "Projected DXF path" not in _text(_blocks(without_dxf))

    with_dxf_manifest = _manifest(tmp_path, feedback_dxf=True)
    with_dxf = _builder().build_feedback("valid", with_dxf_manifest)
    text = _text(_blocks(with_dxf))
    assert "Projected DXF path" in text
    assert str(with_dxf_manifest.feedback_dxf_path) in text


@pytest.mark.parametrize("mode", ["path", "image"])
@pytest.mark.parametrize("feedback_count", [0, 1, 2, 3])
def test_feedback_includes_only_available_renders(
    tmp_path: Path,
    mode: str,
    feedback_count: int,
) -> None:
    manifest = _manifest(tmp_path, feedback_count=feedback_count)
    message = _builder(
        feedback_mode=mode,
        feedback_styles=STYLES,
    ).build_feedback("execution result", manifest)
    blocks = _blocks(message)
    text = _text(blocks)

    if feedback_count == 0:
        assert "Projected perspective renders" not in text
        assert "attached images" not in text
    else:
        assert "Projected perspective renders" in text

    for style in STYLES[:feedback_count]:
        assert style in text
    for style in STYLES[feedback_count:]:
        assert style not in text

    expected_image_count = feedback_count if mode == "image" else 0
    assert sum(block["type"] == "image" for block in blocks) == expected_image_count


def test_access_and_feedback_style_selections_do_not_mix(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path, feedback_count=3)
    builder = _builder(
        access_mode="path",
        access_styles=("style-a",),
        feedback_mode="path",
        feedback_styles=("style-c",),
    )

    initial_text = _text(_blocks(builder.build_initial(manifest)[1]))
    feedback_text = _text(_blocks(builder.build_feedback("valid", manifest)))

    assert "style-a" in initial_text
    assert "style-c" not in initial_text
    assert "style-c" in feedback_text
    assert "style-a" not in feedback_text
