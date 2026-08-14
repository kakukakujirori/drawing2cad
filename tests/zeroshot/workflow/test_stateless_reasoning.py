"""A stateless backend must never be handed a reasoning id it cannot resolve."""

from typing import Any

import langchain_openai.chat_models.base as openai_base
from langchain.agents.middleware import ModelRequest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    AnyMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_openai.chat_models.base import _construct_responses_api_input

from zeroshot.pipeline.workflow.middleware import StatelessReasoningMiddleware
from zeroshot.pipeline.workflow.middleware import stateless_reasoning as module

PAYLOAD = "gAAAAAB_encrypted_payload"


# --- completing the message while it streams ---------------------------------


class FakeItem:
    def __init__(self, type: str, id: str, encrypted_content: str | None) -> None:
        self.type = type
        self.id = id
        self.encrypted_content = encrypted_content

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return {
            "type": self.type,
            "id": self.id,
            "encrypted_content": self.encrypted_content,
            "summary": [],
        }


class FakeChunk:
    def __init__(
        self,
        type: str,
        item: FakeItem | None = None,
        output_index: int = 0,
        **fields: Any,
    ) -> None:
        self.type = type
        self.item = item
        self.output_index = output_index
        self.__dict__.update(fields)


def stream(chunks: list[FakeChunk]) -> AIMessage | None:
    """Drive chunks through the patched converter the way `_astream` does."""
    module.install_reasoning_repair()
    index, output_index, sub_index = -1, -1, -1
    merged = None
    for chunk in chunks:
        (
            index,
            output_index,
            sub_index,
            generation,
        ) = openai_base._convert_responses_chunk_to_generation_chunk(
            chunk, index, output_index, sub_index
        )
        if generation is None:
            continue
        merged = (
            generation.message if merged is None else merged + generation.message  # type: ignore[operator]
        )
    return merged


def reasoning_of(message: Any) -> list[dict[str, Any]]:
    return [
        block
        for block in (message.content if message is not None else [])
        if isinstance(block, dict) and block.get("type") == "reasoning"
    ]


def test_payload_from_done_lands_in_the_message() -> None:
    merged = stream(
        [
            FakeChunk("response.output_item.added", FakeItem("reasoning", "rs_a", "")),
            FakeChunk(
                "response.output_item.done", FakeItem("reasoning", "rs_a", PAYLOAD)
            ),
        ]
    )
    blocks = reasoning_of(merged)
    assert len(blocks) == 1, "the done event must merge in, not append a second block"
    assert blocks[0]["id"] == "rs_a"
    assert blocks[0]["encrypted_content"] == PAYLOAD


def test_two_reasoning_items_keep_separate_blocks() -> None:
    merged = stream(
        [
            FakeChunk("response.output_item.added", FakeItem("reasoning", "rs_a", "")),
            FakeChunk(
                "response.output_item.done", FakeItem("reasoning", "rs_a", "one")
            ),
            FakeChunk(
                "response.output_item.added",
                FakeItem("reasoning", "rs_b", ""),
                output_index=1,
            ),
            FakeChunk(
                "response.output_item.done",
                FakeItem("reasoning", "rs_b", "two"),
                output_index=1,
            ),
        ]
    )
    blocks = reasoning_of(merged)
    assert [b["id"] for b in blocks] == ["rs_a", "rs_b"]
    assert [b["encrypted_content"] for b in blocks] == ["one", "two"]


def test_streamed_summary_is_not_disturbed() -> None:
    """The full event order a real reasoning item produces.

    The payload has to arrive without displacing the summary the deltas built,
    and without splitting the item into a second block.
    """
    merged = stream(
        [
            FakeChunk("response.output_item.added", FakeItem("reasoning", "rs_a", "")),
            FakeChunk(
                "response.reasoning_summary_part.added", summary_index=0, item_id="rs_a"
            ),
            FakeChunk(
                "response.reasoning_summary_text.delta",
                summary_index=0,
                delta="Listing ezdxf ",
            ),
            FakeChunk(
                "response.reasoning_summary_text.delta",
                summary_index=0,
                delta="entities",
            ),
            FakeChunk(
                "response.output_item.done", FakeItem("reasoning", "rs_a", PAYLOAD)
            ),
        ]
    )
    blocks = reasoning_of(merged)
    assert len(blocks) == 1
    assert blocks[0]["encrypted_content"] == PAYLOAD
    assert blocks[0]["summary"] == [
        {"index": 0, "type": "summary_text", "text": "Listing ezdxf entities"}
    ]


def test_payload_merges_after_per_chunk_v1_translation() -> None:
    """The streaming path translates each chunk to v1 before merging them.

    That happens on `lc_rs_*` indexes rather than the raw integer ones, so a
    block the translator declines to re-index survives as a second, orphaned
    reasoning block carrying the payload away from its summary.

    Translation goes through `content_blocks`, which picks the translator from
    `response_metadata['model_provider']` -- calling a translator directly here
    would skip that dispatch and pass while production still split the block.
    """
    chunks = [
        FakeChunk("response.output_item.added", FakeItem("reasoning", "rs_a", "")),
        FakeChunk(
            "response.reasoning_summary_part.added", summary_index=0, item_id="rs_a"
        ),
        FakeChunk(
            "response.reasoning_summary_text.delta", summary_index=0, delta="Planning"
        ),
        FakeChunk("response.output_item.done", FakeItem("reasoning", "rs_a", PAYLOAD)),
    ]
    module.install_reasoning_repair()
    index, output_index, sub_index = -1, -1, -1
    merged = None
    for chunk in chunks:
        (
            index,
            output_index,
            sub_index,
            generation,
        ) = openai_base._convert_responses_chunk_to_generation_chunk(
            chunk, index, output_index, sub_index
        )
        if generation is None:
            continue
        translated = AIMessageChunk(
            content=generation.message.content_blocks,  # type: ignore[arg-type]
            response_metadata=generation.message.response_metadata,
        )
        merged = translated if merged is None else merged + translated

    blocks = reasoning_of(merged)
    assert len(blocks) == 1, "payload must not orphan itself into a second block"
    assert blocks[0]["extras"]["encrypted_content"] == PAYLOAD
    assert blocks[0]["reasoning"] == "Planning"


def test_done_without_payload_adds_nothing() -> None:
    merged = stream(
        [FakeChunk("response.output_item.done", FakeItem("reasoning", "rs_a", ""))]
    )
    assert merged is None


def test_non_reasoning_items_are_left_to_the_original() -> None:
    merged = stream(
        [
            FakeChunk(
                "response.output_item.done", FakeItem("function_call", "fc_1", None)
            )
        ]
    )
    assert reasoning_of(merged) == []


def test_install_is_idempotent() -> None:
    module.install_reasoning_repair()
    before = openai_base._convert_responses_chunk_to_generation_chunk
    module.install_reasoning_repair()
    assert openai_base._convert_responses_chunk_to_generation_chunk is before


def test_unknown_chunk_types_still_pass_through() -> None:
    assert stream([FakeChunk("response.queued")]) is None


# --- the guard for reasoning that predates the repair ------------------------


def reasoning_block(encrypted_content: str) -> dict[str, Any]:
    return {
        "type": "reasoning",
        "id": "rs_abc",
        "index": "lc_rs_1",
        "reasoning": "thinking",
        "extras": {"content": [], "encrypted_content": encrypted_content},
    }


def conversation(encrypted_content: str) -> list[AnyMessage]:
    return [
        HumanMessage("describe the part"),
        AIMessage(
            content=[
                reasoning_block(encrypted_content),
                {
                    "type": "tool_call",
                    "id": "call_1",
                    "name": "run_shell",
                    "args": {"command": "ls"},
                    "extras": {"item_id": "fc_1"},
                },
            ],
            tool_calls=[
                {"name": "run_shell", "args": {"command": "ls"}, "id": "call_1"}
            ],
        ),
        ToolMessage(content="ok", tool_call_id="call_1"),
    ]


class FakeModel:
    def __init__(self, store: bool | None) -> None:
        self.store = store


def run(messages: list[AnyMessage], store: bool | None) -> list[AnyMessage]:
    """Return the messages the middleware would hand to the model."""
    seen: list[list[AnyMessage]] = []
    request = ModelRequest(
        model=FakeModel(store),  # type: ignore[arg-type]
        messages=messages,
        system_prompt=None,
        tool_choice=None,
        tools=[],
        response_format=None,
        state={"messages": messages},
        runtime=None,  # type: ignore[arg-type]
    )

    def handler(request: ModelRequest[None]) -> Any:
        seen.append(list(request.messages))
        return None

    StatelessReasoningMiddleware().wrap_model_call(request, handler)
    return seen[0]


def reasoning_blocks(messages: list[AnyMessage]) -> list[dict[str, Any]]:
    return [
        block
        for message in messages
        if isinstance(message.content, list)
        for block in message.content
        if isinstance(block, dict) and block.get("type") == "reasoning"
    ]


def multipart_conversation(encrypted_content: str) -> list[AnyMessage]:
    """One reasoning item as v1 explodes it: the payload rides on part 0 only."""
    return [
        HumanMessage("describe the part"),
        AIMessage(
            content=[
                {
                    "type": "reasoning",
                    "id": "rs_abc",
                    "index": "lc_rs_305f30",
                    "reasoning": "first",
                    "extras": {"encrypted_content": encrypted_content},
                },
                {
                    "type": "reasoning",
                    "id": "rs_abc",
                    "index": "lc_rs_305f31",
                    "reasoning": "second",
                },
                {
                    "type": "reasoning",
                    "id": "rs_abc",
                    "index": "lc_rs_305f32",
                    "reasoning": "third",
                },
            ]
        ),
    ]


def test_later_summary_parts_survive_with_the_item() -> None:
    kept = reasoning_blocks(run(multipart_conversation(PAYLOAD), store=False))
    assert [block["reasoning"] for block in kept] == ["first", "second", "third"]


def test_an_item_with_no_payload_anywhere_goes_entirely() -> None:
    assert reasoning_blocks(run(multipart_conversation(""), store=False)) == []


def test_bare_reasoning_is_dropped_when_stateless() -> None:
    assert reasoning_blocks(run(conversation(""), store=False)) == []


def test_reasoning_with_a_payload_is_kept() -> None:
    kept = reasoning_blocks(run(conversation(PAYLOAD), store=False))
    assert [block["id"] for block in kept] == ["rs_abc"]


def test_reasoning_survives_untouched_when_the_backend_stores_it() -> None:
    for store in (True, None):
        kept = reasoning_blocks(run(conversation(""), store=store))
        assert [block["id"] for block in kept] == ["rs_abc"], store


def test_tool_calls_are_left_intact() -> None:
    ai_message = run(conversation(""), store=False)[1]
    assert isinstance(ai_message, AIMessage)
    assert [call["id"] for call in ai_message.tool_calls] == ["call_1"]
    assert [block["type"] for block in ai_message.content] == ["tool_call"]


def test_original_messages_are_not_mutated() -> None:
    messages = conversation("")
    run(messages, store=False)
    assert [block["id"] for block in reasoning_blocks(messages)] == ["rs_abc"]


# --- the wire ----------------------------------------------------------------


def test_no_bare_reasoning_id_reaches_the_wire() -> None:
    """The end this exists for: the 404 the backend would otherwise raise."""
    sent = _construct_responses_api_input(conversation(""), store=False)
    assert [item["id"] for item in sent if item.get("type") == "reasoning"] == [
        "rs_abc"
    ], "precondition: langchain replays the bare id"

    guarded = _construct_responses_api_input(
        run(conversation(""), store=False), store=False
    )
    assert [item for item in guarded if item.get("type") == "reasoning"] == []


def test_recovered_payload_reaches_the_wire() -> None:
    sent = _construct_responses_api_input(
        run(conversation(PAYLOAD), store=False), store=False
    )
    reasoning = [item for item in sent if item.get("type") == "reasoning"]
    assert [item["encrypted_content"] for item in reasoning] == [PAYLOAD]
