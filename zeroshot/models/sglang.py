from typing import Any

from langchain_core.messages import AIMessageChunk, BaseMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langchain_openai import ChatOpenAI


class SGLangChatOpenAI(ChatOpenAI):
    """Preserve SGLang reasoning deltas that ChatOpenAI otherwise discards."""

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict[str, Any],
        default_chunk_class: type[BaseMessageChunk],
        base_generation_info: dict[str, Any] | None,
    ) -> ChatGenerationChunk | None:
        generation = super()._convert_chunk_to_generation_chunk(
            chunk,
            default_chunk_class,
            base_generation_info,
        )
        if generation is None or not isinstance(generation.message, AIMessageChunk):
            return generation

        # Select LangChain's provider-neutral content-block fallback for every
        # chunk so merged response metadata remains consistent.
        generation.message.response_metadata["model_provider"] = "sglang"

        choices = chunk.get("choices") or chunk.get("chunk", {}).get("choices", [])
        if choices and isinstance(delta := choices[0].get("delta"), dict):
            reasoning = delta.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning:
                generation.message.additional_kwargs["reasoning_content"] = reasoning

        return generation
