"""Correcting a rejected structured answer without leaving the stage.

Both strategies reach the same correction path: the parse failure raises, and
`ModelCallRetryMiddleware` re-issues the call carrying the validation error.
A `structured_response` written into agent state short-circuits it, because
langchain reads that key as an answer already given.
"""

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel, field_validator

from tests.zeroshot.chat_models import ScriptedChatModel
from zeroshot.pipeline.workflow import create_agent


class Answer(BaseModel):
    """The stage's answer."""

    ticket_id: str

    @field_validator("ticket_id")
    @classmethod
    def require_a_ticket_identifier(cls, value: str) -> str:
        if not value.startswith("ticket_"):
            raise ValueError("ticket_id must be a ticket_... identifier")
        return value


def _answering(ticket_id: str, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "Answer",
                "args": {"ticket_id": ticket_id},
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


def _answering_twice(call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "Answer",
                "args": {"ticket_id": "ticket_initial"},
                "id": f"{call_id}_{index}",
                "type": "tool_call",
            }
            for index in range(2)
        ],
    )


def _agent(model: ScriptedChatModel, *, model_retries: int = 5):
    return create_agent(
        role="coder",
        model=model,
        tools=[],
        output_schema=Answer,
        response_format_strategy="tool",
        max_turns=5,
        announce_turns=False,
        model_retries=model_retries,
        checkpointer=False,
    )


def test_a_rejected_answer_is_returned_to_the_model_to_correct() -> None:
    model = ScriptedChatModel(
        responses=(_answering("T1", "call_1"), _answering("ticket_initial", "call_2"))
    )

    result = _agent(model).invoke({"messages": []})

    assert result["structured_response"] == Answer(ticket_id="ticket_initial")
    assert len(model.received_messages) == 2
    correction = model.received_messages[1][-1]
    assert "ticket_id must be a ticket_... identifier" in str(correction.content)


def test_an_answer_that_stays_malformed_ends_the_run_rather_than_the_stage() -> None:
    """Exhausting the budget has to raise. Ending the stage quietly would hand
    the graph no submission and no reason for one."""
    model = ScriptedChatModel(
        responses=tuple(_answering("T1", "call") for _ in range(3))
    )

    with pytest.raises(Exception, match="ticket_id must be a ticket_"):
        _agent(model, model_retries=2).invoke({"messages": []})

    assert len(model.received_messages) == 3


def test_a_stage_re_entry_does_not_carry_an_answer_into_the_next_invocation() -> None:
    """`create_reconstruction_graph` hands an agent back its own prior state."""
    model = ScriptedChatModel(
        responses=(_answering("T2", "call_1"), _answering("ticket_new", "call_2"))
    )

    result = _agent(model).invoke(
        {
            "messages": [],
            "structured_response": Answer(ticket_id="ticket_previous"),
            "current_turn": 2,
        }
    )

    assert result["structured_response"] == Answer(ticket_id="ticket_new")


def test_two_structured_answers_in_one_turn_are_returned_to_the_model() -> None:
    """A model that calls the answer tool twice is asked for one, not abandoned."""
    model = ScriptedChatModel(
        responses=(_answering_twice("call_1"), _answering("ticket_initial", "call_2"))
    )

    result = _agent(model).invoke({"messages": []})

    assert result["structured_response"] == Answer(ticket_id="ticket_initial")
    assert len(model.received_messages) == 2
    correction = model.received_messages[1][-1]
    assert "2 structured responses" in str(correction.content)
    assert "Answer, Answer" in str(correction.content)


def test_a_rejected_answer_is_not_replayed_with_its_unanswered_tool_calls() -> None:
    """Replaying a tool-calling AIMessage without results invalidates the retry."""
    model = ScriptedChatModel(
        responses=(_answering_twice("call_1"), _answering("ticket_initial", "call_2"))
    )

    _agent(model).invoke({"messages": []})

    assert not any(
        getattr(message, "tool_calls", None) for message in model.received_messages[1]
    )


def test_a_turn_that_only_thought_is_asked_again_rather_than_ending_the_stage() -> None:
    """A model that spends its output budget reasoning returns an empty message,
    which the agent loop would otherwise read as a finished answer."""
    model = ScriptedChatModel(
        responses=(AIMessage(content=""), _answering("ticket_initial", "call_1"))
    )

    result = _agent(model).invoke({"messages": []})

    assert result["structured_response"] == Answer(ticket_id="ticket_initial")
    assert len(model.received_messages) == 2
    nudge = str(model.received_messages[1][-1].content)
    assert "You have been thinking a long time" in nudge
    assert "build on it rather than starting over" in nudge


def test_a_model_that_never_answers_ends_the_stage_without_failing_the_run() -> None:
    """The graph re-asks a stage that submitted nothing, so raising here would
    end a run the pipeline can still recover."""
    model = ScriptedChatModel(responses=tuple(AIMessage(content="") for _ in range(3)))

    result = _agent(model, model_retries=2).invoke({"messages": []})

    assert result.get("structured_response") is None
    assert len(model.received_messages) == 3
