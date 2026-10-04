"""Codex backend with an optional image-history budget."""

from typing import Any, override

from langchain_core.language_models import LanguageModelInput
from langchain_openai.chat_models.codex import _ChatOpenAICodex
from pydantic import Field

from .image_history import trim_image_history


class ChatCodex(_ChatOpenAICodex):
    """Limit request images while keeping the drawing and latest verification."""

    # None sends every image; anchors remain even if they exceed this budget.
    image_history_limit: int | None = Field(default=None, gt=0, strict=True)

    @override
    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # Codex serializes through this hook (Responses API). Prune copies of
        # the same LangChain messages OpenRouter sees, then leave API/auth
        # conversion to the parent. Stored conversation messages stay intact.
        if self.image_history_limit is not None:
            input_ = trim_image_history(
                self._convert_input(input_).to_messages(), self.image_history_limit
            )
        return super()._get_request_payload(input_, stop=stop, **kwargs)
