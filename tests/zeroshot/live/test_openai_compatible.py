import base64
import os
from io import BytesIO
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.content import (
    ImageContentBlock,
    create_image_block,
    create_text_block,
)
from langchain_core.tools import tool
from PIL import Image

MODEL_CONFIG = os.environ.get("ZEROSHOT_LIVE_MODEL_CONFIG")
CONFIG_DIR = Path(__file__).parents[3] / "zeroshot" / "configs"

pytestmark = pytest.mark.skipif(
    MODEL_CONFIG is None,
    reason="set ZEROSHOT_LIVE_MODEL_CONFIG to run a live model backend test",
)


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def _red_png_base64() -> str:
    buffer = BytesIO()
    Image.new("RGB", (16, 16), color=(255, 0, 0)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _assert_response_completed(message: AIMessage) -> None:
    metadata = message.response_metadata
    assert metadata.get("finish_reason") == "stop" or metadata.get("status") == (
        "completed"
    ), metadata


@pytest.fixture(scope="module")
def model() -> BaseChatModel:
    with initialize_config_dir(
        config_dir=str(CONFIG_DIR.resolve()),
        version_base="1.3",
    ):
        config = compose(
            config_name="default",
            overrides=[f"model={MODEL_CONFIG}"],
        )

    return instantiate(config.model)


def test_tool_call_round_trip(model: BaseChatModel) -> None:
    human_message = HumanMessage("Use the add tool to calculate 19 + 23.")

    forced_tool_model = model.bind_tools([add], tool_choice="add")
    tool_call_message = forced_tool_model.invoke([human_message])

    assert len(tool_call_message.tool_calls) == 1
    tool_call = tool_call_message.tool_calls[0]
    assert tool_call["name"] == "add"
    assert tool_call["args"] == {"a": 19, "b": 23}
    assert tool_call["id"]

    tool_message = add.invoke(tool_call)
    assert tool_message.tool_call_id == tool_call["id"]
    assert tool_message.content == "42"

    final_message = model.bind_tools([add]).invoke(
        [human_message, tool_call_message, tool_message]
    )

    assert "42" in final_message.text
    _assert_response_completed(final_message)
    assert final_message.usage_metadata is not None
    assert final_message.usage_metadata["total_tokens"] > 0


def test_initial_user_image(model: BaseChatModel) -> None:
    message = HumanMessage(
        content_blocks=[
            create_text_block(
                "What is the dominant color of this image? Reply with one word."
            ),
            create_image_block(
                base64=_red_png_base64(),
                mime_type="image/png",
            ),
        ]
    )

    response = model.invoke([message])

    assert "red" in response.text.casefold()
    _assert_response_completed(response)


def test_image_tool_result_round_trip(model: BaseChatModel) -> None:
    @tool("load_probe_image")
    def load_probe_image() -> list[ImageContentBlock]:
        """Load the image whose dominant color must be identified."""
        return [
            create_image_block(
                base64=_red_png_base64(),
                mime_type="image/png",
            )
        ]

    human_message = HumanMessage(
        "Call load_probe_image, then report its dominant color in one word."
    )
    forced_tool_model = model.bind_tools(
        [load_probe_image],
        tool_choice="load_probe_image",
    )
    tool_call_message = forced_tool_model.invoke([human_message])

    assert len(tool_call_message.tool_calls) == 1
    tool_call = tool_call_message.tool_calls[0]
    assert tool_call["name"] == "load_probe_image"

    tool_message = load_probe_image.invoke(tool_call)
    assert tool_message.tool_call_id == tool_call["id"]
    assert isinstance(tool_message.content, list)
    assert tool_message.content[0]["type"] == "image"

    final_message = model.bind_tools([load_probe_image]).invoke(
        [human_message, tool_call_message, tool_message]
    )

    assert "red" in final_message.text.casefold()
    _assert_response_completed(final_message)
