import base64
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from PIL import Image

from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.tools.errors import ToolFeedbackError
from zeroshot.pipeline.tools.load_image import create_load_image_tool


def _write_image(path: Path, image_format: str) -> bytes:
    Image.new("RGB", (3, 2), color=(10, 20, 30)).save(
        path,
        format=image_format,
    )
    return path.read_bytes()


def test_tool_schema_exposes_only_image_path(tmp_path: Path) -> None:
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    load_image = create_load_image_tool(workdir)

    schema = load_image.get_input_jsonschema()

    assert load_image.name == "load_image"
    assert set(schema["properties"]) == {"image_path"}
    assert schema["required"] == ["image_path"]
    assert schema["properties"]["image_path"]["type"] == "string"


@pytest.mark.parametrize(
    ("image_path", "filename", "image_format", "expected_mime_type"),
    [
        ("/work/view.png", "view.png", "PNG", "image/png"),
        ("view.jpg", "view.jpg", "JPEG", "image/jpeg"),
    ],
    ids=["absolute-sandbox-path", "relative-sandbox-path"],
)
def test_tool_returns_image_content_block(
    tmp_path: Path,
    image_path: str,
    filename: str,
    image_format: str,
    expected_mime_type: str,
) -> None:
    expected_bytes = _write_image(tmp_path / filename, image_format)
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    load_image = create_load_image_tool(workdir)

    result = load_image.invoke(
        {
            "type": "tool_call",
            "name": "load_image",
            "args": {"image_path": image_path},
            "id": "call-load-image",
        }
    )

    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call-load-image"
    assert result.name == "load_image"
    assert isinstance(result.content, list)
    assert len(result.content) == 1

    block = result.content[0]
    assert block["type"] == "image"
    assert block["mime_type"] == expected_mime_type
    assert base64.b64decode(block["base64"]) == expected_bytes
    assert result.content_blocks == result.content


def test_tool_loads_image_through_symlink_within_workdir(tmp_path: Path) -> None:
    expected_bytes = _write_image(tmp_path / "target.png", "PNG")
    (tmp_path / "link.png").symlink_to("target.png")
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    load_image = create_load_image_tool(workdir)

    result = load_image.invoke({"image_path": "/work/link.png"})

    assert len(result) == 1
    assert base64.b64decode(result[0]["base64"]) == expected_bytes


@pytest.mark.parametrize(
    "image_path",
    [
        "../outside.png",
        "/work/../outside.png",
        "/etc/passwd",
    ],
)
def test_tool_rejects_path_outside_sandbox_namespace(
    tmp_path: Path,
    image_path: str,
) -> None:
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    load_image = create_load_image_tool(workdir)

    with pytest.raises(ToolFeedbackError):
        load_image.invoke({"image_path": image_path})


def test_tool_rejects_symlink_escape(tmp_path: Path) -> None:
    workdir_path = tmp_path / "work"
    workdir_path.mkdir()
    outside_path = tmp_path / "outside.png"
    _write_image(outside_path, "PNG")
    (workdir_path / "link.png").symlink_to(outside_path)
    workdir = SandboxWorkdir(host_bind_dir=workdir_path)
    load_image = create_load_image_tool(workdir)

    with pytest.raises(ToolFeedbackError, match="Cannot access image"):
        load_image.invoke({"image_path": "/work/link.png"})


def test_tool_rejects_missing_file(tmp_path: Path) -> None:
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    load_image = create_load_image_tool(workdir)

    with pytest.raises(ToolFeedbackError, match="Cannot access image"):
        load_image.invoke({"image_path": "/work/missing.png"})


def test_tool_rejects_directory(tmp_path: Path) -> None:
    image_dir = tmp_path / "image.png"
    image_dir.mkdir()
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    load_image = create_load_image_tool(workdir)

    with pytest.raises(ToolFeedbackError, match="Image is not a regular file"):
        load_image.invoke({"image_path": "/work/image.png"})


def test_tool_rejects_non_image_file(tmp_path: Path) -> None:
    """A model pointing this at a DXF must get feedback, not end the run.

    The tool node forwards ToolFeedbackError to the model and lets anything
    else propagate, so PIL's own error has to be translated here.
    """
    (tmp_path / "not-image.png").write_text("not an image", encoding="utf-8")
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    load_image = create_load_image_tool(workdir)

    with pytest.raises(ToolFeedbackError, match="Not a readable image"):
        load_image.invoke({"image_path": "/work/not-image.png"})


def test_a_truncated_image_is_reported_rather_than_raised(tmp_path: Path) -> None:
    """`Image.open` is lazy, so a header-only file only fails on verify()."""
    image_bytes = _write_image(tmp_path / "full.png", "PNG")
    (tmp_path / "cut.png").write_bytes(image_bytes[: len(image_bytes) // 2])
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    load_image = create_load_image_tool(workdir)

    with pytest.raises(ToolFeedbackError, match="Not a readable image"):
        load_image.invoke({"image_path": "/work/cut.png"})


def test_the_tool_node_turns_a_bad_path_into_feedback(tmp_path: Path) -> None:
    """The property that actually matters: the graph survives the mistake."""
    (tmp_path / "techdraw.dxf").write_text("0\nSECTION\n", encoding="utf-8")
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    builder = StateGraph(MessagesState)  # type: ignore[type-var]
    builder.add_node(
        "tools",
        ToolNode(
            [create_load_image_tool(workdir)], handle_tool_errors=ToolFeedbackError
        ),
    )
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)

    result = builder.compile().invoke(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "load_image",
                            "args": {"image_path": "/work/techdraw.dxf"},
                            "id": "call-dxf",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        }
    )

    message = result["messages"][-1]
    assert isinstance(message, ToolMessage)
    assert message.status == "error"
    assert "Not a readable image" in str(message.content)
