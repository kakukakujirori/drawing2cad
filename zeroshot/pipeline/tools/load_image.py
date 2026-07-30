import base64
from io import BytesIO
from pathlib import Path, PurePosixPath

from langchain_core.messages.content import (
    ImageContentBlock,
    create_image_block,
)
from langchain_core.tools import BaseTool, tool
from PIL import Image

from zeroshot.pipeline.sandbox import SandboxWorkdir


def create_load_image_tool(workdir: SandboxWorkdir) -> BaseTool:

    def _resolve_image_path(sandbox_image_path: str | PurePosixPath) -> Path:
        try:
            host_image_path = workdir.sandbox_to_host_path(sandbox_image_path)
            host_image_path = host_image_path.resolve(strict=True)
        except (OSError, RuntimeError, ValueError):
            raise ValueError(f"Cannot access image: {sandbox_image_path}") from None

        if not host_image_path.is_relative_to(workdir.host_bind_dir.resolve()):
            raise ValueError(f"Cannot access image: {sandbox_image_path}")

        if not host_image_path.is_file():
            raise ValueError(f"Image is not a regular file: {sandbox_image_path}")

        return host_image_path

    @tool("load_image")
    def load_image(image_path: str) -> list[ImageContentBlock]:
        """
        Load an image from a file path.
        Args:
            image_path: The path to the image file.
        """
        host_image_path = _resolve_image_path(image_path)
        image_bytes = host_image_path.read_bytes()
        with Image.open(BytesIO(image_bytes)) as image:
            mime_type = image.get_format_mimetype()
            image.verify()

        if mime_type is None:
            raise ValueError(f"Unknown image format: {host_image_path.name}")

        return [
            create_image_block(
                base64=base64.b64encode(image_bytes).decode("utf-8"),
                mime_type=mime_type,
            )
        ]

    return load_image
