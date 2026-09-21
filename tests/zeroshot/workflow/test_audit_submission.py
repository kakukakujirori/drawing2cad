"""An audit that contradicts its snapshot goes back to the model before it is the answer."""

import json
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, tool

from tests.zeroshot.chat_models import ScriptedChatModel, unanswered_tool_calls
from tests.zeroshot.contracts import bootstrap_review
from tests.zeroshot.workflow.test_reconstruction_workflow import (
    _completed_run,
    _ref,
    _report,
    _snapshot,
)
from zeroshot.pipeline.stages.audit.contracts import AuditReport, TicketReview
from zeroshot.pipeline.stages.audit.validate import validate_audit_report
from zeroshot.pipeline.workflow import create_agent
from zeroshot.pipeline.workflow.lifecycle import open_next_round
from zeroshot.pipeline.workflow.middleware import VerifyOnSubmitMiddleware

_TICKET = "ticket_001_shape_mismatch"
_REVIEWED = AuditReport(
    concern_reviews={},
    ticket_reviews={
        _TICKET: TicketReview(summary="The hole is restored.", solved=True)
    },
    findings=[],
)
_UNREVIEWED = AuditReport(
    concern_reviews={}, ticket_reviews=bootstrap_review(), findings=[]
)


def _revision_snapshot():
    first = open_next_round(
        _completed_run(), _report(target=_ref("coding", "ret_hole"))
    )
    return _completed_run(first).snapshots[-1]


def _agent(
    model: ScriptedChatModel,
    strategy: str,
    *,
    tools: list[BaseTool] | None = None,
    model_retries: int = 1,
):
    return create_agent(
        role="output_auditor",
        model=model,
        tools=tools or [],
        output_schema=AuditReport,
        response_format_strategy=strategy,  # type: ignore[arg-type]
        max_turns=5,
        announce_turns=False,
        model_retries=model_retries,
        checkpointer=False,
        extra_middleware=[VerifyOnSubmitMiddleware(validate_audit_report)],
    )


def _answer(payload: dict[str, Any], strategy: str, call_id: str) -> AIMessage:
    if strategy == "provider":
        return AIMessage(content=json.dumps(payload))
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "AuditReport", "args": payload, "id": call_id, "type": "tool_call"}
        ],
    )


@pytest.mark.parametrize("strategy", ["tool", "provider"])
@pytest.mark.parametrize(
    ("rejected", "problem"),
    [
        ({"accepted": True, "findings": []}, "$: field 'ticket_reviews' is required"),
        (_UNREVIEWED.model_dump(mode="json"), f"missing=['{_TICKET}']"),
        (
            _report(target=_ref("coding", "ret_hole")).model_dump(mode="json"),
            f"missing=['{_TICKET}']",
        ),
    ],
    ids=["omitted", "accepted_with_empty", "rejected_with_empty"],
)
def test_an_unreviewed_ticket_is_corrected_before_the_report_is_committed(
    strategy: str, rejected: dict[str, Any], problem: str
) -> None:
    model = ScriptedChatModel(
        responses=(
            _answer(rejected, strategy, "answer_1"),
            _answer(_REVIEWED.model_dump(mode="json"), strategy, "answer_2"),
        )
    )

    result = _agent(model, strategy).invoke(
        {"messages": [HumanMessage("Audit round 1.")]},
        context=_revision_snapshot(),
    )

    assert result["structured_response"] == _REVIEWED
    assert problem in model.received_messages[1][-1].text
    assert unanswered_tool_calls(model.received_messages[1]) == []
    # Neither the rejected answer nor its acknowledgement enters the transcript.
    assert [type(m) for m in result["messages"]] == [
        HumanMessage,
        AIMessage,
        *([ToolMessage] if strategy == "tool" else []),
    ]


def test_a_rejected_report_discards_the_tool_calls_made_beside_it() -> None:
    probed: list[str] = []

    @tool
    def probe() -> str:
        """Record that the tool ran."""
        probed.append("probe")
        return "probed"

    def turn(report: AuditReport, index: int) -> AIMessage:
        return AIMessage(
            content="",
            tool_calls=[
                {"name": "probe", "args": {}, "id": f"probe_{index}"},
                {
                    "name": "AuditReport",
                    "args": report.model_dump(mode="json"),
                    "id": f"answer_{index}",
                },
            ],
        )

    model = ScriptedChatModel(responses=(turn(_UNREVIEWED, 1), turn(_REVIEWED, 2)))

    result = _agent(model, "tool", tools=[probe]).invoke(
        {"messages": [HumanMessage("Audit round 1.")]},
        context=_revision_snapshot(),
    )

    assert result["structured_response"] == _REVIEWED
    assert probed == ["probe"]
    sent = json.dumps(_UNREVIEWED.model_dump(mode="json"))
    assert f"You sent: {sent}." in model.received_messages[1][-1].text
    assert unanswered_tool_calls(model.received_messages[1]) == []
    assert unanswered_tool_calls(result["messages"]) == []
    assert {
        m.tool_call_id for m in result["messages"] if isinstance(m, ToolMessage)
    } == {
        "probe_2",
        "answer_2",
    }


def test_each_invocation_checks_its_own_snapshot_and_may_end_unanswered() -> None:
    model = ScriptedChatModel(
        responses=tuple(
            _answer(_UNREVIEWED.model_dump(mode="json"), "tool", f"answer_{index}")
            for index in range(3)
        )
    )
    agent = _agent(model, "tool")

    bootstrap = agent.invoke(
        {"messages": [HumanMessage("Audit round 0.")]},
        context=_snapshot(),
    )
    revision = agent.invoke(
        {**bootstrap, "messages": [*bootstrap["messages"], HumanMessage("Round 1.")]},
        context=_revision_snapshot(),
    )

    assert bootstrap["structured_response"] == _UNREVIEWED
    # The report round 0 accepted is not carried over as round 1's answer.
    assert revision.get("structured_response") is None
    assert f"missing=['{_TICKET}']" in revision["messages"][-1].text


def test_an_audit_invoked_without_its_snapshot_is_a_pipeline_defect() -> None:
    model = ScriptedChatModel(
        responses=(_answer(_REVIEWED.model_dump(mode="json"), "tool", "answer"),)
    )

    with pytest.raises(RuntimeError, match="with the context"):
        _agent(model, "tool").invoke({"messages": [HumanMessage("Audit.")]})
    assert model.received_messages == []
