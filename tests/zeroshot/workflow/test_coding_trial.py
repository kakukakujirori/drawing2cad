"""Coder reminders run after tools, even when the program was not edited."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

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
from zeroshot.pipeline.stages.coding.middleware import (
    CodingDraftMiddleware,
    CodingTrialMiddleware,
)
from zeroshot.pipeline.workflow.middleware import VerifyOnWriteMiddleware


def _reminders(messages):
    return [
        message
        for message in messages
        if isinstance(message, HumanMessage) and message.name == "coding_trial_reminder"
    ]


def test_read_only_tools_and_verification_share_one_reminder_per_batch(tmp_path):
    source = tmp_path / "model.py"
    verifier = _CountingVerifier(source)
    model = ScriptedChatModel(
        responses=(
            AIMessage(
                content="Measure first",
                tool_calls=[
                    {"name": "echo", "args": {"value": "front"}, "id": "front"},
                    {"name": "echo", "args": {"value": "top"}, "id": "top"},
                ],
            ),
            tool_call("write", {"text": "candidate"}, "write"),
            tool_call("echo", {"value": "inspect candidate"}, "inspect"),
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
        max_turns=4,
        output_schema=ExampleProposal,
        response_format_strategy="tool",
        extra_middleware=[
            VerifyOnWriteMiddleware(verifier),
            CodingTrialMiddleware(("echo", "write"), max_turns=4),
        ],
    )
    result = agent.invoke({"messages": [HumanMessage(content="reconstruct")]})
    seen = model.received_messages

    assert [len(_reminders(messages)) for messages in seen] == [0, 1, 2, 2]
    assert verifier.seen == ["candidate"]
    assert not any("[verified]" in message.text for message in seen[1])
    assert sum("[verified]" in message.text for message in seen[2]) == 1
    assert result["structured_response"].proposal == ["candidate"]
    assert model.bound_tool_name_history[-1] == ("ExampleProposal",)

    # Both parallel results precede the first reminder; all provider requests
    # have complete call/result pairs, including the final submission.
    reminder_index = seen[1].index(_reminders(seen[1])[0])
    assert {
        message.tool_call_id
        for message in seen[1][:reminder_index]
        if isinstance(message, ToolMessage)
    } == {"front", "top"}
    assert all(not unanswered_tool_calls(messages) for messages in seen)

    # Re-entering on the retained transcript must not replay the old reminder.
    model.responses = (
        *model.responses,
        tool_call(
            "ExampleProposal",
            {"proposal": ["candidate"], "rationale": "still checked"},
            "submit-again",
        ),
    )
    agent.invoke(
        {
            **result,
            "messages": [
                *result["messages"],
                HumanMessage(content="review the answer"),
            ],
        }
    )
    assert len(_reminders(model.received_messages[-1])) == 2


def test_reminder_waits_for_results_and_does_not_repeat_on_reentry():
    middleware = CodingTrialMiddleware(("echo",), max_turns=4)
    call = AIMessage(
        content="",
        tool_calls=[
            {"name": "echo", "args": {}, "id": "a"},
            {"name": "echo", "args": {}, "id": "b"},
        ],
    )
    partial = [call, ToolMessage(content="front", tool_call_id="a")]
    complete = [*partial, ToolMessage(content="top", tool_call_id="b")]

    for turn, messages in (
        (0, complete),  # Old batch at stage entry.
        (1, partial),  # One tool still pending.
        (3, complete),  # Next turn is answer-only.
        (4, complete),  # Budget exhausted.
        (1, [AIMessage(content="finished")]),  # No tools.
        (
            1,
            [
                tool_call("TicketAnswers", {}, "submission"),
                ToolMessage(content="refused", tool_call_id="submission"),
            ],
        ),  # Submission is not a trial tool.
    ):
        assert (
            middleware.before_model({"messages": messages, "current_turn": turn}, None)
            is None
        )

    update = middleware.before_model({"messages": complete, "current_turn": 1}, None)
    assert update is not None
    assert len(_reminders(update["messages"])) == 1
    # A retry/checkpoint replay may revisit this hook without new tools.
    assert (
        middleware.before_model(
            {"messages": [*complete, *update["messages"]], "current_turn": 1}, None
        )
        is None
    )


def test_no_reminder_before_the_interpretation_is_ready():
    """Trial builds mean nothing while interpretation.json still blocks the build."""
    ready = False
    middleware = CodingTrialMiddleware(("echo",), max_turns=4, ready=lambda: ready)
    state = {
        "current_turn": 1,
        "messages": [
            tool_call("echo", {"value": "front"}, "front"),
            ToolMessage(content="front", tool_call_id="front"),
        ],
    }

    assert middleware.before_model(state, None) is None
    ready = True
    assert _reminders(middleware.before_model(state, None)["messages"])


def _draft_reminders(messages):
    return [
        message
        for message in messages
        if isinstance(message, HumanMessage) and message.name == "coding_draft_reminder"
    ]


def _drafting_agent(model, source, max_turns):
    return _subgraph(
        model,
        tools=(echo, _writing_tool(source)),
        max_turns=max_turns,
        output_schema=ExampleProposal,
        response_format_strategy="tool",
        extra_middleware=[
            CodingDraftMiddleware(
                lambda: source.read_text() if source.is_file() else "",
                files=("interpretation.json", "model.py"),
                first_turn=2,
                max_turns=max_turns,
            )
        ],
    )


_SUBMIT = tool_call(
    "ExampleProposal", {"proposal": ["candidate"], "rationale": "checked"}, "submit"
)


def test_draft_reminder_repeats_every_ten_turns_while_nothing_is_written(tmp_path):
    inspections = [tool_call("echo", {"value": "look"}, f"look{n}") for n in range(22)]
    model = ScriptedChatModel(responses=(*inspections, _SUBMIT))

    _drafting_agent(model, tmp_path / "model.py", max_turns=23).invoke(
        {"messages": [HumanMessage(content="reconstruct")]}
    )
    counts = [len(_draft_reminders(messages)) for messages in model.received_messages]

    # Turns 2, 12 and 22; the answer-only turn 23 gets none.
    assert [turn for turn in range(2, 24) if counts[turn - 1] > counts[turn - 2]] == [
        2,
        12,
        22,
    ]
    assert counts[-1] == 3
    [reminder, *_] = _draft_reminders(model.received_messages[-1])
    assert "Neither interpretation.json nor model.py has changed" in reminder.text


def test_draft_reminder_stops_once_a_file_changes(tmp_path):
    source = tmp_path / "model.py"
    source.write_text("seeded", encoding="utf-8")
    model = ScriptedChatModel(
        responses=(
            tool_call("echo", {"value": "look"}, "look"),
            tool_call("write", {"text": "draft"}, "write"),
            *[tool_call("echo", {"value": "look"}, f"look{n}") for n in range(11)],
            _SUBMIT,
        )
    )

    _drafting_agent(model, source, max_turns=14).invoke(
        {"messages": [HumanMessage(content="reconstruct")]}
    )

    # Only turn 2's, sent before the write: the seeded file is the start.
    assert len(_draft_reminders(model.received_messages[-1])) == 1


def test_draft_reminder_needs_a_turn_to_start_on():
    with pytest.raises(ValueError, match="first_turn"):
        CodingDraftMiddleware(lambda: "", ("a", "b"), first_turn=0, max_turns=60)
