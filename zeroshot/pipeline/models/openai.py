"""OpenAI-compatible backend with the shared image-history budget."""

from typing import Any, override

from langchain_core.language_models import LanguageModelInput
from langchain_openai import ChatOpenAI as _ChatOpenAI
from pydantic import Field

from .image_history import trim_image_history


class ChatOpenAI(_ChatOpenAI):
    image_history_limit: int | None = Field(default=None, gt=0, strict=True)

    @override
    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if self.image_history_limit is not None:
            input_ = trim_image_history(
                self._convert_input(input_).to_messages(), self.image_history_limit
            )
        return super()._get_request_payload(input_, stop=stop, **kwargs)
