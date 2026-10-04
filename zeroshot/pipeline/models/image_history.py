"""Choose images before either backend serializes the conversation.

Protect the source drawing and latest visual verification, then fill the
remaining budget with recent tool/report images. Replace older images with
omission notices in message copies; saved transcripts keep their images.
"""

import logging
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage

logger = logging.getLogger(__name__)


def _is_image(part: Any) -> bool:
    # ArtifactPresenter uses LangChain blocks; load_image uses OpenAI blocks.
    return isinstance(part, dict) and part.get("type") in {"image", "image_url"}


def trim_image_history(
    messages: Sequence[BaseMessage], limit: int | None, *, strict: bool = False
) -> list[BaseMessage]:
    """Limit request images, preserving source/latest-report anchors.

    ``None`` disables pruning. With ``strict=True``, protected images alone
    exceeding the limit are an error (OpenRouter's configured hard limit); otherwise
    anchors take priority over the budget (Codex's context budget).
    """
    outgoing = list(messages)
    if limit is None:
        return outgoing

    # Collect image locations oldest-first and find the last visual report.
    images = []
    latest_report = None
    for message_index, message in enumerate(messages):
        if not isinstance(message.content, list):
            continue
        text = "\n".join(
            part.get("text", "") for part in message.content if isinstance(part, dict)
        )
        report = (
            isinstance(message, HumanMessage)
            and "[Execution result]" in text
            and "[Input artifacts]" not in text
        )
        for part_index, part in enumerate(message.content):
            if _is_image(part):
                images.append(
                    (
                        message_index,
                        part_index,
                        isinstance(message, ToolMessage) or report,
                    )
                )
                if report:
                    latest_report = message_index

    # Only reports with images count: a failed build must not replace the last
    # visual report. Other user images (including the drawing) stay protected.
    removable = [
        (message_index, part_index)
        for message_index, part_index, history_image in images
        if history_image and message_index != latest_report
    ]
    protected = len(images) - len(removable)
    if strict and protected > limit:
        raise ValueError(
            f"Protected image attachments ({protected}) exceed "
            f"image_history_limit ({limit}); none were dropped."
        )

    # Remove oldest images first; clone only content lists we actually change.
    omitted = max(0, len(removable) - max(0, limit - protected))
    copied: dict[int, list[Any]] = {}
    for message_index, part_index in removable[:omitted]:
        original = messages[message_index]
        if message_index not in copied:
            copied[message_index] = list(original.content)
            outgoing[message_index] = original.model_copy(
                update={"content": copied[message_index]}
            )
        copied[message_index][part_index] = {
            "type": "text",
            "text": "Image omitted from this request.",
        }
    logger.info(
        "Image history: %s -> %s images (%s protected)",
        len(images),
        len(images) - omitted,
        protected,
    )
    return outgoing
