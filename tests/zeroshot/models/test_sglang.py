from langchain_core.messages import AIMessageChunk

from zeroshot.models import SGLangChatOpenAI


def test_sglang_reasoning_delta_becomes_standard_content_block() -> None:
    model = SGLangChatOpenAI(model="test", api_key="EMPTY")

    first = model._convert_chunk_to_generation_chunk(
        {
            "choices": [
                {
                    "delta": {
                        "role": "assistant",
                        "reasoning_content": "inspect ",
                    },
                    "finish_reason": None,
                }
            ]
        },
        AIMessageChunk,
        None,
    )
    second = model._convert_chunk_to_generation_chunk(
        {
            "choices": [
                {
                    "delta": {"reasoning_content": "the drawing"},
                    "finish_reason": None,
                }
            ]
        },
        AIMessageChunk,
        None,
    )

    assert first is not None
    assert second is not None
    generation = first + second
    assert generation.message.response_metadata["model_provider"] == "sglang"
    assert generation.message.content_blocks == [
        {"type": "reasoning", "reasoning": "inspect the drawing"}
    ]
