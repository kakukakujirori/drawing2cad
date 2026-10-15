"""After a threshold, each coder turn is pushed to write, then to try, not speculate."""

from itertools import pairwise

import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from tests.zeroshot.chat_models import (
    ScriptedChatModel,
    tool_call,
    unanswered_tool_calls,
)
from tests.zeroshot.workflow.test_agent import (
    ExampleProposal,
    _CountingVerifier,
    _subgraph,
    _writing_tool,
    echo,
)
from zeroshot.pipeline.stages.coding.middleware import CodingReminderMiddleware
from zeroshot.pipeline.workflow.middleware import VerifyOnWriteMiddleware

_WRITE = (
    "Rather than long speculation, quickly write a provisional {file}: it gets "
    "you useful verification feedback from the pipeline."
)
_TRY = (
    "Rather than long speculation, quickly try any unresolved geometry as a "
    "trial: it gets you useful verification feedback from the pipeline."
)


class _Workspace:
    """Stands in for the coding workspace: which file the round has not written."""

    def __init__(self, unwritten):
        self.unwritten = unwritten

    def unwritten_file(self):
        return self.unwritten()


def _reminders(messages):
    return [
        message.text
        for message in messages
        if isinstance(message, HumanMessage) and message.name == "coding_reminder"
    ]


def test_after_the_threshold_each_turn_asks_to_write_then_to_try(tmp_path):
    source = tmp_path / "model.py"
    verifier = _CountingVerifier(source)
    model = ScriptedChatModel(
        responses=(
            tool_call("echo", {"value": "front"}, "front"),
            tool_call("echo", {"value": "top"}, "top"),
            tool_call("echo", {"value": "right"}, "right"),
            tool_call("write", {"text": "candidate"}, "write"),
            tool_call("echo", {"value": "inspect"}, "inspect"),
            tool_call(
                "ExampleProposal",
                {"proposal": ["candidate"], "rationale": "checked"},
                "submit",
            ),
        )
    )
    agent = _subgraph(
        model,
        tools=(echo, _writing_tool(source)),
        max_turns=6,
        output_schema=ExampleProposal,
        response_format_strategy="tool",
        extra_middleware=[
            VerifyOnWriteMiddleware(verifier),
            CodingReminderMiddleware(
                _Workspace(lambda: None if source.is_file() else "model.py"),  # type: ignore[arg-type]
                max_turns=6,
                remind_after_turns=2,
            ),
        ],
    )
    agent.invoke({"messages": [HumanMessage(content="reconstruct")]})
    seen = model.received_messages
    added = [_reminders(now[len(before) :]) for before, now in pairwise(seen)]

    # Turns 2 to 6; turn 4 is asked before its write, turn 6 is answer-only.
    write = _WRITE.format(file="model.py")
    assert added == [[], [write], [write], [_TRY], []]
    # The reminder follows the tool result and the verification report.
    texts = [message.text for message in seen[4]]
    assert texts.index(_TRY) > texts.index("[verified] build 1")
    assert all(not unanswered_tool_calls(messages) for messages in seen)


def test_a_replayed_hook_adds_no_second_reminder():
    middleware = CodingReminderMiddleware(
        _Workspace(lambda: "interpretation.json"),  # type: ignore[arg-type]
        max_turns=30,
        remind_after_turns=5,
    )
    state = {
        "current_turn": 9,
        "messages": [
            tool_call("echo", {"value": "front"}, "front"),
            ToolMessage(content="front", tool_call_id="front"),
        ],
    }

    update = middleware.before_model(state, None)
    assert _reminders(update["messages"]) == [_WRITE.format(file="interpretation.json")]
    replay = {**state, "messages": [*state["messages"], *update["messages"]]}
    assert middleware.before_model(replay, None) is None
    # Turn 5 is not yet after 5 turns.
    assert middleware.before_model({**state, "current_turn": 4}, None) is None


def test_the_reminder_cannot_start_on_the_first_turn():
    with pytest.raises(ValueError, match="remind_after_turns"):
        CodingReminderMiddleware(
            _Workspace(lambda: None),  # type: ignore[arg-type]
            max_turns=30,
            remind_after_turns=0,
        )
