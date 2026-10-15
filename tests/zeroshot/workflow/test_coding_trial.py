"""After a threshold, each coder turn is pushed to write, then to try, not speculate."""

from itertools import pairwise

import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from tests.zeroshot.chat_models import (
    ScriptedChatModel,
    tool_call,
    unanswered_tool_calls,
)
from tests.zeroshot.workflow.test_agent import ExampleProposal, _subgraph, echo
from tests.zeroshot.workflow.test_coding_workspace import _workspace
from zeroshot.pipeline.stages.coding.middleware import CodingReminderMiddleware

_WRITE = (
    "Rather than long speculation, quickly write a provisional {file}: it gets "
    "you useful verification feedback from the pipeline."
)
_TRY = (
    "Rather than long speculation, quickly try any unresolved geometry as a "
    "trial: it gets you useful verification feedback from the pipeline."
)


def _reminders(messages):
    return [
        message.text
        for message in messages
        if isinstance(message, HumanMessage) and message.name == "coding_reminder"
    ]


def test_after_the_threshold_each_turn_gets_one_reminder(tmp_path):
    workspace, _, _ = _workspace(tmp_path)
    model = ScriptedChatModel(
        responses=(
            *[tool_call("echo", {"value": "look"}, f"look{n}") for n in range(5)],
            tool_call(
                "ExampleProposal",
                {"proposal": ["candidate"], "rationale": "checked"},
                "submit",
            ),
        )
    )
    agent = _subgraph(
        model,
        tools=(echo,),
        max_turns=6,
        output_schema=ExampleProposal,
        response_format_strategy="tool",
        extra_middleware=[
            CodingReminderMiddleware(workspace, max_turns=6, remind_after_turns=2)
        ],
    )
    agent.invoke({"messages": [HumanMessage(content="reconstruct")]})
    seen = model.received_messages
    added = [_reminders(now[len(before) :]) for before, now in pairwise(seen)]

    # Turns 2 to 6: from turn 3, but not on the answer-only turn 6.
    write = _WRITE.format(file="interpretation.json")
    assert added == [[], [write], [write], [write], []]
    # The reminder follows the last tool result and the turn's count.
    tool_result, count, reminder = seen[2][-3:]
    assert isinstance(tool_result, ToolMessage)
    assert (count.text, reminder.text) == ("[turn 3/6]", write)
    assert all(not unanswered_tool_calls(messages) for messages in seen)


def test_the_reminder_follows_what_the_round_has_written(tmp_path):
    # Round 0 opens on the seeded interpretation; the helper then writes model.py.
    workspace, _, candidate = _workspace(tmp_path)
    middleware = CodingReminderMiddleware(workspace, max_turns=30, remind_after_turns=5)
    state = {
        "current_turn": 9,
        "messages": [
            tool_call("echo", {"value": "front"}, "front"),
            ToolMessage(content="front", tool_call_id="front"),
        ],
    }

    def reminder():
        update = middleware.before_model(state, None)
        return [] if update is None else _reminders(update["messages"])

    assert reminder() == [_WRITE.format(file="interpretation.json")]
    # Written but invalid: nothing builds, and its own report says what to fix.
    workspace.interpretation.source_path.write_text("{}")
    workspace.feedback()
    assert reminder() == []
    workspace.interpretation.source_path.write_text(candidate.model_dump_json())
    workspace.feedback()
    assert reminder() == [_TRY]
    # Turn 5 is not yet after 5 turns.
    assert middleware.before_model({**state, "current_turn": 4}, None) is None


def test_a_replayed_hook_adds_no_second_reminder(tmp_path):
    workspace, _, _ = _workspace(tmp_path)
    middleware = CodingReminderMiddleware(workspace, max_turns=30, remind_after_turns=5)
    state = {
        "current_turn": 9,
        "messages": [
            tool_call("echo", {"value": "front"}, "front"),
            ToolMessage(content="front", tool_call_id="front"),
        ],
    }

    update = middleware.before_model(state, None)
    replay = {**state, "messages": [*state["messages"], *update["messages"]]}
    assert middleware.before_model(replay, None) is None


def test_the_reminder_cannot_start_on_the_first_turn(tmp_path):
    workspace, _, _ = _workspace(tmp_path)
    with pytest.raises(ValueError, match="remind_after_turns"):
        CodingReminderMiddleware(workspace, max_turns=30, remind_after_turns=0)
