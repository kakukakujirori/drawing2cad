"""A stateless backend must never be handed a reasoning id it cannot resolve."""

from typing import Any

from langchain.agents.middleware import ModelRequest
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_openai.chat_models.base import _construct_responses_api_input

from zeroshot.pipeline.workflow.middleware import StatelessReasoningMiddleware

PAYLOAD = "gAAAAAB_encrypted_payload"


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


def test_the_payload_reaches_the_wire_exactly_as_it_arrived() -> None:
    """Byte for byte: this module once wrote the payload itself, and went on
    doing it after `langchain_openai` began supplying one -- appending to a
    payload already there until the backend refused to decrypt the pair."""
    sent = _construct_responses_api_input(
        run(conversation(PAYLOAD), store=False), store=False
    )
    reasoning = [item for item in sent if item.get("type") == "reasoning"]
    assert [item["encrypted_content"] for item in reasoning] == [PAYLOAD]
    assert [item["id"] for item in reasoning] == ["rs_abc"]
