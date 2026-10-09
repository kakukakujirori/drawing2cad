"""Codex backend with complete tool arguments and an image-history budget."""

from collections.abc import AsyncIterator, Iterator
from typing import Any, override

import openai
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_openai.chat_models.base import (
    _astream_with_chunk_timeout,
    _convert_responses_chunk_to_generation_chunk,
    _handle_openai_api_error,
    _handle_openai_bad_request,
)
from langchain_openai.chat_models.codex import (
    _CODEX_HEADERS_KWARG,
    _ChatOpenAICodex,
)
from pydantic import Field

from .image_history import trim_image_history


class _ResponseChunks:
    """Keep LangChain's text/metadata conversion; emit completed tools once."""

    def __init__(self, model: "ChatCodex", response: Any, schema: Any):
        self.cursor = (-1, -1, -1)
        self.schema = schema
        self.output_version = model.output_version
        self.has_reasoning = False
        self.metadata = (
            {"headers": dict(response.response.headers)}
            if model.include_response_headers
            else {}
        )

    def convert(self, event: Any) -> ChatGenerationChunk | None:
        # Codex can omit argument deltas. Emit the final item with its call_id
        # instead of assembling fragments, which also avoids double appending.
        if event.type in ("response.output_item.added", "response.output_item.done"):
            if event.item.type == "function_call":
                if event.type == "response.output_item.added":
                    return None
                # shortcut: LangChain ignores function_call done items; remove
                # this translation when its converter supports complete calls.
                event = event.model_copy(update={"type": "response.output_item.added"})
        elif event.type.startswith("response.function_call_arguments."):
            return None
        index, output_index, sub_index, generation = (
            _convert_responses_chunk_to_generation_chunk(
                event,
                *self.cursor,
                schema=self.schema,
                metadata=self.metadata,
                has_reasoning=self.has_reasoning,
                output_version=self.output_version,
            )
        )
        self.cursor = (index, output_index, sub_index)
        if generation is not None:
            self.metadata = {}
            if "reasoning" in generation.message.additional_kwargs:
                self.has_reasoning = True
        return generation


class ChatCodex(_ChatOpenAICodex):
    """Preserve tool arguments and limit request images."""

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

    @override
    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        self._ensure_sync_client_available()
        payload = self._get_request_payload(
            messages, stop=stop, **{**kwargs, "stream": True}
        )
        try:
            with self.root_client.responses.create(**payload) as response:
                chunks = _ResponseChunks(self, response, kwargs.get("response_format"))
                for event in response:
                    generation = chunks.convert(event)
                    if generation is not None:
                        if run_manager:
                            run_manager.on_llm_new_token(
                                generation.text, chunk=generation
                            )
                        yield generation
        except openai.BadRequestError as error:
            _handle_openai_bad_request(error)
        except openai.APIError as error:
            _handle_openai_api_error(error)

    @override
    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        token = await self.token_provider.aget_token()
        kwargs[_CODEX_HEADERS_KWARG] = self._build_headers(token.account_id)
        payload = self._get_request_payload(
            messages, stop=stop, **{**kwargs, "stream": True}
        )
        try:
            async with await self.root_async_client.responses.create(
                **payload
            ) as response:
                chunks = _ResponseChunks(self, response, kwargs.get("response_format"))
                async for event in _astream_with_chunk_timeout(
                    response, self.stream_chunk_timeout, model_name=self.model_name
                ):
                    generation = chunks.convert(event)
                    if generation is not None:
                        if run_manager:
                            await run_manager.on_llm_new_token(
                                generation.text, chunk=generation
                            )
                        yield generation
        except openai.BadRequestError as error:
            _handle_openai_bad_request(error)
        except openai.APIError as error:
            _handle_openai_api_error(error)
