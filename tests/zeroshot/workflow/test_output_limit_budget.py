import asyncio
import json
import sqlite3
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from PIL import Image

from tests.zeroshot.chat_models import ScriptedChatModel, tool_call
from tests.zeroshot.pipeline_single import test_single_graph as single_tests
from tests.zeroshot.workflow import test_graph as staged_tests
from tests.zeroshot.workflow.test_agent import _FlakyChatModel, _length_failure
from tests.zeroshot.workflow.test_structured_output_retry import Answer, _answering
from zeroshot.pipeline.messages.artifact import ArtifactPresenter
from zeroshot.pipeline.messages.manifest import InputManifest, register_view
from zeroshot.pipeline.runner import PipelineRunner
from zeroshot.pipeline.sandbox import SandboxRunner, SandboxWorkdir
from zeroshot.pipeline.stages.interpretation.contracts import View
from zeroshot.pipeline.workflow import create_agent
from zeroshot.pipeline.workflow._config import _child_graph_config
from zeroshot.pipeline.workflow.lifecycle import load_reconstruction
from zeroshot.pipeline.workflow.middleware.output_limit_budget import (
    OutputLimitBudget,
    OutputLimitBudgetExceeded,
)


def limited(metadata=None):
    return AIMessage(
        content="", response_metadata=metadata or {"finish_reason": "length"}
    )


def agent(model, **kwargs):
    return create_agent(
        role="coder",
        model=model,
        tools=kwargs.pop("tools", []),
        max_turns=5,
        announce_turns=False,
        model_retries=5,
        **kwargs,
    )


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "metadata",
    [
        {"finish_reason": "length"},
        {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
    ],
)
def test_third_limit_stops_before_a_fourth_call(asynchronous, metadata):
    model = ScriptedChatModel(responses=tuple(limited(metadata) for _ in range(4)))
    budget = OutputLimitBudget(3)
    subject = agent(model, output_limit_budget=budget)
    with pytest.raises(OutputLimitBudgetExceeded, match="3/3"):
        if asynchronous:
            asyncio.run(subject.ainvoke({"messages": []}))
        else:
            subject.invoke({"messages": []})
    assert len(model.received_messages) == budget.failures == 3


def test_agents_share_budget_across_success_and_stage_reentry():
    budget = OutputLimitBudget(3)
    first_model = ScriptedChatModel(responses=(limited(), AIMessage(content="done")))
    coder_model = ScriptedChatModel(
        responses=(
            limited(),
            AIMessage(content="done"),
            limited(),
            AIMessage(content="unused"),
        )
    )
    first = agent(first_model, output_limit_budget=budget)
    coder = agent(coder_model, output_limit_budget=budget)
    graph = StateGraph(dict)
    graph.add_node("interpretation", lambda state: first.invoke({"messages": []}))
    graph.add_node("coding", lambda state: coder.invoke({"messages": []}))
    graph.add_node("refinement", lambda state: coder.invoke(state))
    graph.add_edge(START, "interpretation")
    graph.add_edge("interpretation", "coding")
    graph.add_edge("coding", "refinement")
    graph.add_edge("refinement", END)
    with pytest.raises(OutputLimitBudgetExceeded):
        graph.compile().invoke({})
    assert len(first_model.received_messages) == 2
    assert len(coder_model.received_messages) == 3
    assert budget.failures == 3


def test_length_with_a_valid_tool_call_counts_before_tools_run():
    calls = []

    @tool
    def record(value: int) -> str:
        """Record an executed tool call."""
        calls.append(value)
        return "ok"

    responses = []
    for value in range(3):
        message = tool_call("record", {"value": value}, f"call-{value}")
        message.response_metadata = {"finish_reason": "length"}
        responses.append(message)
    model = ScriptedChatModel(responses=tuple(responses))
    with pytest.raises(OutputLimitBudgetExceeded):
        agent(model, tools=[record], output_limit_budget=OutputLimitBudget(3)).invoke(
            {"messages": []}
        )
    assert calls == [0, 1]
    assert len(model.received_messages) == 3


def test_structured_error_with_length_and_sdk_length_exception_count():
    invalid = _answering("not-a-ticket", "bad")
    invalid.response_metadata = {"finish_reason": "length"}
    model = ScriptedChatModel(responses=(invalid,))
    with pytest.raises(OutputLimitBudgetExceeded):
        agent(
            model,
            output_schema=Answer,
            response_format_strategy="tool",
            output_limit_budget=OutputLimitBudget(1),
        ).invoke({"messages": []})
    sdk_model = _FlakyChatModel(
        responses=(), errors=tuple(_length_failure() for _ in range(4))
    )
    with pytest.raises(OutputLimitBudgetExceeded):
        agent(sdk_model, output_limit_budget=OutputLimitBudget(3)).invoke(
            {"messages": []}
        )
    assert sdk_model.attempts == 3


@pytest.mark.parametrize("asynchronous", [False, True])
def test_valid_structured_answer_counts_ai_length_before_submission(asynchronous):
    message = _answering("ticket_ok", "good")
    message.response_metadata = {"finish_reason": "length"}
    model = ScriptedChatModel(responses=(message,))
    budget = OutputLimitBudget(1)
    subject = agent(
        model,
        output_schema=Answer,
        response_format_strategy="tool",
        output_limit_budget=budget,
    )
    with pytest.raises(OutputLimitBudgetExceeded):
        if asynchronous:
            asyncio.run(subject.ainvoke({"messages": []}))
        else:
            subject.invoke({"messages": []})
    assert len(model.received_messages) == budget.failures == 1


@pytest.mark.parametrize(
    "metadata",
    [
        {"finish_reason": "tool_calls"},
        {"status": "incomplete", "incomplete_details": {"reason": "content_filter"}},
    ],
)
def test_token_count_and_other_incomplete_reason_do_not_count(metadata):
    message = AIMessage(
        content="done",
        response_metadata=metadata,
        usage_metadata={
            "input_tokens": 1,
            "output_tokens": 60000,
            "total_tokens": 60001,
        },
    )
    budget = OutputLimitBudget(3)
    agent(ScriptedChatModel(responses=(message,)), output_limit_budget=budget).invoke(
        {"messages": []}
    )
    assert budget.failures == 0


def test_no_budget_preserves_existing_retry_behavior():
    model = ScriptedChatModel(
        responses=(limited(), limited(), limited(), AIMessage(content="done"))
    )
    result = agent(model).invoke({"messages": []})
    assert result["messages"][-1].text == "done"
    assert len(model.received_messages) == 4


@pytest.mark.parametrize("cap", [None, 0, -1, True, 1.5, "3"])
def test_output_limit_cap_requires_a_positive_integer(cap):
    with pytest.raises(ValueError, match="max_output_limit_failures"):
        OutputLimitBudget(cap)


@pytest.mark.parametrize("cross_round", [False, True])
def test_staged_workflow_shares_the_cap_across_stages_and_rounds(
    monkeypatch, cross_round
):
    staged_tests._stub_verification(
        monkeypatch, staged_tests._verified("000"), staged_tests._verified("001")
    )
    interpretation = staged_tests._interpretation_script()
    operations = staged_tests._operations_script()
    coding = (
        staged_tests._coding_submission(),
        staged_tests._coding_submission(staged_tests._ROUND_ONE_TICKET),
    )
    audit = staged_tests._audit_script(staged_tests._rejected_audit())
    interpretation[0].response_metadata = {"finish_reason": "length"}
    if cross_round:
        audit[-1].response_metadata = {"finish_reason": "length"}
        coding[1].response_metadata = {"finish_reason": "length"}
    else:
        operations[0].response_metadata = {"finish_reason": "length"}
        coding[0].response_metadata = {"finish_reason": "length"}
    interpreter = ScriptedChatModel(responses=interpretation)
    planner = ScriptedChatModel(responses=operations)
    coder = ScriptedChatModel(responses=coding)
    auditor = ScriptedChatModel(responses=audit)
    with SandboxWorkdir() as workdir:
        graph = staged_tests._graph(
            workdir,
            interpreter=interpreter,
            planner=planner,
            coder=coder,
            auditor=auditor,
            max_output_limit_failures=3,
            max_audit_reject_count=1,
        )
        with pytest.raises(OutputLimitBudgetExceeded, match="3/3"):
            graph.invoke({})
        history = load_reconstruction(workdir.host_bind_dir / "reconstruction.json")
        assert history.snapshots[-1].round == int(cross_round)
    assert len(interpreter.received_messages) == len(planner.received_messages) == 2
    assert len(coder.received_messages) == (2 if cross_round else 1)
    assert len(auditor.received_messages) == (2 if cross_round else 0)


@pytest.mark.parametrize("cap", [2, 3])
def test_single_workflow_shares_the_configured_cap(monkeypatch, cap):
    monkeypatch.setattr(
        single_tests.graph_module, "OutputVerifier", single_tests._StubVerifier
    )
    monkeypatch.setattr(
        single_tests.graph_module, "compact_transcript", lambda thread, **_: thread[-1:]
    )
    coding = (
        single_tests._answer(single_tests.CodingReport(summary="plate")),
        single_tests._answer(single_tests.CodingReport(summary="revised plate")),
    )
    finding = single_tests.Finding(
        observation="hole missing", evidence=["front"], revision_request="add it"
    )
    audit = single_tests._answer(
        single_tests.SingleAuditReport(accepted=False, findings=[finding])
    )
    for message in (*coding, audit):
        message.response_metadata = {"finish_reason": "length"}
    coder = ScriptedChatModel(responses=coding)
    auditor = ScriptedChatModel(responses=(audit,))
    with SandboxWorkdir() as workdir:
        graph = single_tests._graph(
            workdir, coder, auditor, max_output_limit_failures=cap
        )
        with pytest.raises(OutputLimitBudgetExceeded, match=f"{cap}/{cap}"):
            graph.invoke({})
    assert len(coder.received_messages) == cap - 1
    assert len(auditor.received_messages) == 1


def test_runner_preserves_artifacts_and_new_runs_start_with_a_fresh_cap(tmp_path):
    models = []
    budgets = []

    def factory(**kwargs):
        workdir = kwargs["sandbox_workdir"].host_bind_dir
        model = ScriptedChatModel(responses=tuple(limited() for _ in range(4)))
        models.append(model)
        budget = OutputLimitBudget(3)
        budgets.append(budget)
        subject = agent(model, output_limit_budget=budget)

        def coding(state, config: RunnableConfig):
            (workdir / "model.py").write_text("saved program")
            (workdir / "output.step").write_bytes(b"saved STEP")
            return subject.invoke({"messages": []}, config=_child_graph_config(config))

        graph = StateGraph(dict)
        graph.add_node("coding", coding)
        graph.add_edge(START, "coding")
        graph.add_edge("coding", END)
        return graph.compile(checkpointer=kwargs["checkpointer"])

    input_path = tmp_path / "drawing.png"
    Image.new("RGB", (8, 8)).save(input_path)
    runner = PipelineRunner(
        sandbox_runner=SandboxRunner(
            python_executable=Path(sys.executable), default_timeout_s=30
        ),
        graph_factory=factory,
        artifact_presenter=ArtifactPresenter(input="path"),
        artifact_root=tmp_path / "runs",
        on_existing="retry",
    )

    def manifest(name):
        return InputManifest(
            sample_id=name,
            drawing=[register_view("view_drawing", View.FULL_PAGE, input_path)],
        )

    for name in ["sample-1", "sample-2", "sample-1"]:
        with pytest.raises(OutputLimitBudgetExceeded):
            runner.run_sample(manifest(name))
        root = tmp_path / "runs" / name
        assert not (root / "output_limit_budget.json").exists()
        events = [
            json.loads(line)
            for line in (root / "events.jsonl").read_text().splitlines()
        ]
        assert events[-1]["event"] == "run_failed"
        assert events[-1]["data"]["error_type"] == "OutputLimitBudgetExceeded"
        assert (root / "workspace/model.py").read_text() == "saved program"
        assert (root / "workspace/output.step").read_bytes() == b"saved STEP"
        with sqlite3.connect(root / "checkpoints.sqlite") as connection:
            connection.execute("BEGIN EXCLUSIVE")
            connection.rollback()
    assert [len(m.received_messages) for m in models] == [3, 3, 3]
    assert [budget.failures for budget in budgets] == [3, 3, 3]
    assert len({id(budget) for budget in budgets}) == 3


class _CappedScriptedModel(ScriptedChatModel):
    max_tokens: int = 60000


def test_an_answer_that_fills_max_tokens_counts_whatever_finish_it_reports():
    message = AIMessage(
        content="done",
        response_metadata={"finish_reason": "tool_calls"},
        usage_metadata={
            "input_tokens": 1,
            "output_tokens": 60000,
            "total_tokens": 60001,
        },
    )
    budget = OutputLimitBudget(3)
    model = _CappedScriptedModel(responses=(message,))
    agent(model, output_limit_budget=budget).invoke({"messages": []})
    assert budget.failures == 1
