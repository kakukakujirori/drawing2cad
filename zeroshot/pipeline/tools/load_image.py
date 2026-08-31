import base64
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any

from langchain_core.tools import BaseTool, tool
from PIL import Image

from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.tools.errors import ToolFeedbackError


def create_load_image_tool(workdir: SandboxWorkdir) -> BaseTool:

    def _resolve_image_path(sandbox_image_path: str | PurePosixPath) -> Path:
        try:
            host_image_path = workdir.sandbox_to_host_path(sandbox_image_path)
            host_image_path = host_image_path.resolve(strict=True)
        except (OSError, RuntimeError, ValueError):
            raise ToolFeedbackError(
                f"Cannot access image: {sandbox_image_path}"
            ) from None

        if not host_image_path.is_relative_to(workdir.host_bind_dir.resolve()):
            raise ToolFeedbackError(f"Cannot access image: {sandbox_image_path}")

        if not host_image_path.is_file():
            raise ToolFeedbackError(
                f"Image is not a regular file: {sandbox_image_path}"
            )

        return host_image_path

    @tool("load_image")
    def load_image(image_path: str) -> list[dict[str, Any]]:
        """
        Load an image from a file path.
        Args:
            image_path: The path to the image file.
        """
        host_image_path = _resolve_image_path(image_path)
        try:
            image_bytes = host_image_path.read_bytes()
            with Image.open(BytesIO(image_bytes)) as image:
                mime_type = image.get_format_mimetype()
                image.verify()
        except (OSError, SyntaxError) as error:
            raise ToolFeedbackError(
                f"Not a readable image: {image_path} ({error})"
            ) from None

        if mime_type is None:
            raise ToolFeedbackError(f"Unknown image format: {host_image_path.name}")

        # Return OpenAI-native image_url blocks directly.  langchain-openrouter
        # does not run ToolMessage content through _format_message_content, so
        # the langchain-internal ImageContentBlock format would reach the API
        # unconverted and the model would not see the image.  The image_url
        # format is the wire format both langchain-openai and
        # langchain-openrouter pass through unchanged.
        data_uri = (
            f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('utf-8')}"
        )
        return [
            {
                "type": "image_url",
                "image_url": {"url": data_uri},
            }
        ]

    return load_image
