"""Choose images before either backend serializes the conversation.

Keep the run's input drawings, then the newest other images. Replace older
images with omission notices in message copies; saved transcripts keep theirs.
"""

import logging
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)

# The block id the prompt gives the input drawings it attaches.
INPUT_IMAGE_ID = "input_drawing"


def _is_image(part: Any) -> bool:
    # ArtifactPresenter uses LangChain blocks; load_image uses OpenAI blocks.
    return isinstance(part, dict) and part.get("type") in {"image", "image_url"}


def trim_image_history(
    messages: Sequence[BaseMessage], limit: int | None, *, strict: bool = False
) -> list[BaseMessage]:
    """Limit request images to the input drawings and the newest others.

    ``None`` disables pruning. With ``strict=True`` (OpenRouter's hard limit),
    input drawings alone exceeding the limit are an error.
    """
    outgoing = list(messages)
    if limit is None:
        return outgoing

    # Image locations oldest-first.
    inputs, others = [], []
    for message_index, message in enumerate(messages):
        if not isinstance(message.content, list):
            continue
        for part_index, part in enumerate(message.content):
            if _is_image(part):
                is_input = part.get("id") == INPUT_IMAGE_ID
                (inputs if is_input else others).append((message_index, part_index))
    if strict and len(inputs) > limit:
        raise ValueError(
            f"Input drawings ({len(inputs)}) exceed image_history_limit ({limit}); "
            "none were dropped."
        )

    # Remove oldest images first; clone only content lists we actually change.
    omitted = max(0, len(others) - max(0, limit - len(inputs)))
    copied: dict[int, list[Any]] = {}
    for message_index, part_index in others[:omitted]:
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
        "Image history: %s -> %s images (%s input drawings)",
        len(inputs) + len(others),
        len(inputs) + len(others) - omitted,
        len(inputs),
    )
    return outgoing
