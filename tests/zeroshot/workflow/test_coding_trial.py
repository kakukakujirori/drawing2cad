"""Coder reminders run after tools, even when the program was not edited."""

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
from zeroshot.pipeline.workflow.middleware import (
    CodingTrialMiddleware,
    VerifyOnWriteMiddleware,
)


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
            VerifyOnWriteMiddleware(verifier, require_feedback_before_submit=True),
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
