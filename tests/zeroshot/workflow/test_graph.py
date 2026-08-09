import json
import sys
from functools import partial
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool, tool

from tests.zeroshot.chat_models import ScriptedChatModel, tool_call
from zeroshot.pipeline.messages import MessageBuilder
from zeroshot.pipeline.sandbox import SandboxRunner, SandboxWorkdir
from zeroshot.pipeline.tools import VerifyOutputResult
from zeroshot.pipeline.verification import StepRenderer
from zeroshot.pipeline.workflow import SemanticHypothesis, StopReason, create_agent
from zeroshot.pipeline.workflow import graph as graph_module
from zeroshot.pipeline.workflow.graph import AgentFactory, create_reconstruction_graph
from zeroshot.pipeline.workflow.proposer_critic import create_proposer_critic_loop
from zeroshot.pipeline.workflow.state import OperationPlan


def _renderer() -> StepRenderer:
    return StepRenderer(timeout_s=60.0)


def _message_builder() -> MessageBuilder:
    return MessageBuilder(
        access_render3d="none",
        access_render3d_styles=(),
        feedback_render3d="none",
        feedback_render3d_styles=(),
    )


def _agent(role: str, model: BaseChatModel, **overrides: Any) -> AgentFactory:
    return partial(create_agent, role=role, model=model, **overrides)


def _hypothesis(*semantics: str) -> AIMessage:
    return AIMessage(content=json.dumps({"semantics": list(semantics)}))


def _plan(*operations: str) -> AIMessage:
    return AIMessage(content=json.dumps({"operations": list(operations)}))


def _review(decision: str, feedback: str = "") -> AIMessage:
    return AIMessage(content=json.dumps({"decision": decision, "feedback": feedback}))


def _audit(decision: str, feedback: str = "") -> AIMessage:
    return AIMessage(content=json.dumps({"decision": decision, "feedback": feedback}))


def _stage(
    proposer_role: str,
    proposer: ScriptedChatModel,
    critic_role: str,
    critic: ScriptedChatModel,
    max_revisions: int,
    **options: Any,
):
    return partial(
        create_proposer_critic_loop,
        proposer=_agent(proposer_role, proposer, **options),
        critic=_agent(critic_role, critic, **options),
        max_revisions=max_revisions,
    )


def _graph(
    workdir: SandboxWorkdir,
    hypothesizer: ScriptedChatModel,
    reviewer: ScriptedChatModel,
    coder: ScriptedChatModel,
    input_message: HumanMessage | None = None,
    agent_options: dict[str, Any] | None = None,
    *,
    planner: ScriptedChatModel | None = None,
    plan_reviewer: ScriptedChatModel | None = None,
    auditor: ScriptedChatModel | None = None,
    **overrides: Any,
):
    """The staged workflow with one scripted model behind each role.

    The operations stage agrees at once unless a test says otherwise: most of
    these are about what the workflow does with an artifact, not how the stage
    that produced it argued about it.
    """
    options = {"announce_turn_budget": False, **(agent_options or {})}
    input_message = input_message or HumanMessage(content="Reconstruct this drawing")
    max_revisions = overrides.pop("max_revisions", 2)
    return create_reconstruction_graph(
        semantic_stage=_stage(
            "semantic_hypothesizer",
            hypothesizer,
            "semantic_reviewer",
            reviewer,
            max_revisions,
            **options,
        ),
        operations_stage=_stage(
            "operation_planner",
            planner or ScriptedChatModel(responses=(_plan("extrude the outline"),)),
            "operation_reviewer",
            plan_reviewer or ScriptedChatModel(responses=(_review("accept", "fine"),)),
            max_revisions,
            **options,
        ),
        coder=_agent("coder", coder, **options),
        auditor=_agent(
            "output_auditor",
            auditor or ScriptedChatModel(responses=(_audit("accept", "matches"),)),
            **options,
        ),
        sandbox_runner=SandboxRunner(
            python_executable=Path(sys.executable), default_timeout_s=10
        ),
        sandbox_workdir=workdir,
        renderer=_renderer(),
        message_builder=_message_builder(),
        input_message=input_message,
        **overrides,
    )


def _texts(messages: list[BaseMessage]) -> list[str]:
    return [message.text for message in messages]


def test_an_accepted_hypothesis_reaches_the_coder_and_the_final_verification() -> None:
    hypothesizer = ScriptedChatModel(responses=(_hypothesis("a flanged boss"),))
    reviewer = ScriptedChatModel(responses=(_review("accept", "matches all views"),))
    coder = ScriptedChatModel(responses=(AIMessage(content="written"),))
    input_message = HumanMessage(content="Reconstruct this drawing")

    with SandboxWorkdir() as workdir:
        graph = _graph(workdir, hypothesizer, reviewer, coder, input_message)
        result = graph.invoke({"messages": []})

    assert result["semantic_hypothesis"] == SemanticHypothesis(
        semantics=["a flanged boss"]
    )
    # Every role that spoke reported for itself, under its own name.
    assert result["stop_reasons"] == {
        role: StopReason.COMPLETED
        for role in (
            "semantic_hypothesizer",
            "semantic_reviewer",
            "operation_planner",
            "operation_reviewer",
            "coder",
            "output_auditor",
        )
    }
    assert result["last_verification"] == VerifyOutputResult(
        status="REJECTED",
        executor_error="model.py was not found",
    )

    # Each role read what came before it, and the drawing opened all three.
    for model in (hypothesizer, reviewer, coder):
        assert model.received_messages[0][1].text == input_message.text
    assert "a flanged boss" in reviewer.received_messages[0][-1].text
    assert "a flanged boss" in coder.received_messages[0][-1].text
    assert len(coder.received_messages[0]) > len(hypothesizer.received_messages[0])


def test_the_workflow_transcript_records_each_message_once() -> None:
    hypothesizer = ScriptedChatModel(responses=(_hypothesis("a plate"),))
    reviewer = ScriptedChatModel(responses=(_review("accept", "fine"),))
    coder = ScriptedChatModel(responses=(AIMessage(content="written"),))

    with SandboxWorkdir() as workdir:
        graph = _graph(workdir, hypothesizer, reviewer, coder)
        result = graph.invoke({"messages": []})

    messages = result["messages"]
    ids = [message.id for message in messages]
    assert len(ids) == len(set(ids))
    # The run input, then an instruction and an answer for each of the six
    # roles the workflow asked something of.
    assert [type(message) for message in messages] == [
        HumanMessage,
        *[HumanMessage, AIMessage] * 6,
    ]
    assert not [message for message in messages if isinstance(message, SystemMessage)]


def test_the_revision_loop_stops_at_its_limit_and_codes_what_it_has() -> None:
    hypothesizer = ScriptedChatModel(
        responses=tuple(_hypothesis("a plate") for _ in range(6))
    )
    reviewer = ScriptedChatModel(
        responses=tuple(_review("revise", "still incomplete") for _ in range(6))
    )
    coder = ScriptedChatModel(responses=(AIMessage(content="written"),))

    with SandboxWorkdir() as workdir:
        graph = _graph(workdir, hypothesizer, reviewer, coder, max_revisions=2)
        result = graph.invoke({"messages": []})

    # The workflow codes the latest hypothesis rather than abandoning the run.
    assert result["semantic_hypothesis"] == SemanticHypothesis(semantics=["a plate"])
    assert len(coder.received_messages) == 1


def test_the_coder_is_given_both_settled_artifacts() -> None:
    hypothesizer = ScriptedChatModel(responses=(_hypothesis("a flanged boss"),))
    reviewer = ScriptedChatModel(responses=(_review("accept", "fine"),))
    planner = ScriptedChatModel(responses=(_plan("extrude the outline 25mm"),))
    plan_reviewer = ScriptedChatModel(responses=(_review("accept", "builds it"),))
    coder = ScriptedChatModel(responses=(AIMessage(content="written"),))

    with SandboxWorkdir() as workdir:
        graph = _graph(
            workdir,
            hypothesizer,
            reviewer,
            coder,
            planner=planner,
            plan_reviewer=plan_reviewer,
        )
        result = graph.invoke({"messages": []})

    assert result["operation_plan"] == OperationPlan(
        operations=["extrude the outline 25mm"]
    )
    # The plan was written against the hypothesis, and the coder was handed
    # both rather than being left to find them in the transcript.
    assert "a flanged boss" in planner.received_messages[0][-1].text
    instruction = coder.received_messages[0][-1].text
    assert "a flanged boss" in instruction
    assert "extrude the outline 25mm" in instruction


def test_a_plan_that_was_never_settled_leaves_the_coder_nothing_to_follow() -> None:
    hypothesizer = ScriptedChatModel(responses=(_hypothesis("a plate"),))
    reviewer = ScriptedChatModel(responses=(_review("accept", "fine"),))
    planner = ScriptedChatModel(
        responses=tuple(
            tool_call("run_shell", {"command": "true"}, f"call-{turn}")
            for turn in range(4)
        )
    )
    coder = ScriptedChatModel(responses=())

    with SandboxWorkdir() as workdir:
        graph = _graph(
            workdir,
            hypothesizer,
            reviewer,
            coder,
            agent_options={"max_turns": 2},
            planner=planner,
            plan_reviewer=ScriptedChatModel(responses=()),
        )
        result = graph.invoke({"messages": []})

    assert result["semantic_hypothesis"] == SemanticHypothesis(semantics=["a plate"])
    assert "operation_plan" not in result
    # The planner is the one that ran out; the semantic stage before it did not.
    assert result["stop_reasons"]["operation_planner"] is StopReason.BUDGET_EXHAUSTED
    assert result["stop_reasons"]["semantic_hypothesizer"] is StopReason.COMPLETED
    assert coder.received_messages == []


def test_a_stage_that_never_answered_leaves_the_coder_nothing_to_write() -> None:
    """Its budget went on investigating, so there is no hypothesis to code."""
    hypothesizer = ScriptedChatModel(
        responses=tuple(
            tool_call("run_shell", {"command": "true"}, f"call-{turn}")
            for turn in range(4)
        )
    )
    reviewer = ScriptedChatModel(responses=())
    coder = ScriptedChatModel(responses=())

    with SandboxWorkdir() as workdir:
        graph = _graph(
            workdir,
            hypothesizer,
            reviewer,
            coder,
            agent_options={"max_turns": 2},
        )
        result = graph.invoke({"messages": []})

    assert "semantic_hypothesis" not in result
    assert result["stop_reasons"] == {
        "semantic_hypothesizer": StopReason.BUDGET_EXHAUSTED
    }
    assert reviewer.received_messages == []
    assert coder.received_messages == []
    # The run still reports what the workspace holds, so every sample has a row.
    assert result["last_verification"] == VerifyOutputResult(
        status="REJECTED",
        executor_error="model.py was not found",
    )


def test_an_agent_that_finished_without_its_typed_answer_stops_the_run() -> None:
    hypothesizer = ScriptedChatModel(
        responses=(AIMessage(content='{"semantics": "a boss"}'),)
    )
    reviewer = ScriptedChatModel(responses=())
    coder = ScriptedChatModel(responses=())

    with SandboxWorkdir() as workdir:
        graph = _graph(workdir, hypothesizer, reviewer, coder)
        with pytest.raises(Exception, match="SemanticHypothesis"):
            graph.invoke({"messages": []})

    assert coder.received_messages == []


def test_a_later_agent_does_not_inherit_an_earlier_agents_turn_notices() -> None:
    hypothesizer = ScriptedChatModel(
        responses=(
            tool_call("run_shell", {"command": "true"}, "call-look"),
            _hypothesis("a plate"),
        )
    )
    reviewer = ScriptedChatModel(responses=(_review("accept", "fine"),))
    coder = ScriptedChatModel(responses=(AIMessage(content="written"),))

    with SandboxWorkdir() as workdir:
        graph = _graph(
            workdir,
            hypothesizer,
            reviewer,
            coder,
            agent_options={"announce_turn_budget": True},
        )
        result = graph.invoke({"messages": []})

    def notices(messages: list[BaseMessage]) -> list[str]:
        return [
            message.text
            for message in messages
            if isinstance(message, HumanMessage) and message.text.startswith("[turn ")
        ]

    # An agent keeps its own ladder, counting only its own turns, so a stage
    # that inherits four AI turns still opens at turn 1 of its budget.
    assert notices(hypothesizer.received_messages[-1]) == [
        "[turn 1/30]",
        "[turn 2/30]",
    ]
    assert notices(reviewer.received_messages[0]) == ["[turn 1/30]"]
    assert notices(coder.received_messages[0]) == ["[turn 1/30]"]
    # They are a true record of what each agent was shown, so the workflow
    # transcript keeps every one; only the handoff to the next agent drops them.
    assert len(notices(result["messages"])) == 7


def test_the_coder_still_verifies_its_own_work_before_the_workflow_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hypothesizer = ScriptedChatModel(responses=(_hypothesis("a plate"),))
    reviewer = ScriptedChatModel(responses=(_review("accept", "fine"),))
    coder = ScriptedChatModel(
        responses=(
            tool_call("verify_output", {}, "call-intermediate-verification"),
            AIMessage(content="done"),
        )
    )
    invocations: list[str] = []
    final_report = VerifyOutputResult(
        verification_id="001",
        status="VERIFIED",
        source="result = object()",
        returncode=0,
    )

    @tool("verify_output")
    def model_verify_output() -> dict[str, str]:
        """Stub the model-facing intermediate verification."""
        invocations.append("model")
        return {"status": "REJECTED"}

    @tool("verify_output")
    def final_verify_output() -> VerifyOutputResult:
        """Stub the workflow-owned final verification."""
        invocations.append("workflow")
        return final_report

    def create_stub_verify_output_tool(
        *args: object,
        serialize_output: bool = True,
        **kwargs: object,
    ) -> BaseTool:
        del args, kwargs
        return model_verify_output if serialize_output else final_verify_output

    monkeypatch.setattr(
        graph_module,
        "create_verify_output_tool",
        create_stub_verify_output_tool,
    )

    with SandboxWorkdir() as workdir:
        graph = _graph(workdir, hypothesizer, reviewer, coder)
        result = graph.invoke({"messages": []})

    assert invocations == ["model", "workflow"]
    assert result["last_verification"] == final_report
    # The coder's own verification is a tool result inside its own turns, two
    # messages before the audit the workflow asked for afterwards.
    assert isinstance(result["messages"][-4], ToolMessage)


def test_the_workflow_runs_the_final_verification_after_a_coder_runs_out() -> None:
    hypothesizer = ScriptedChatModel(responses=(_hypothesis("a plate"),))
    reviewer = ScriptedChatModel(responses=(_review("accept", "fine"),))
    coder = ScriptedChatModel(
        responses=tuple(
            tool_call("run_shell", {"command": "true"}, f"call-{turn}")
            for turn in range(4)
        )
    )

    with SandboxWorkdir() as workdir:
        graph = _graph(
            workdir,
            hypothesizer,
            reviewer,
            coder,
            agent_options={"max_turns": 3, "announce_turn_budget": False},
        )
        result = graph.invoke({"messages": []})

    assert result["agent_turns"] == {
        "semantic_hypothesizer": 1,
        "semantic_reviewer": 1,
        "operation_planner": 1,
        "operation_reviewer": 1,
        "coder": 3,
        "output_auditor": 1,
    }
    assert result["last_verification"] == VerifyOutputResult(
        status="REJECTED",
        executor_error="model.py was not found",
    )


@pytest.mark.parametrize(
    ("decision", "rerun"),
    [
        ("redo_code", "coder"),
        ("redo_operations", "operation_planner"),
        ("redo_semantics", "semantic_hypothesizer"),
    ],
)
def test_the_audit_sends_the_run_back_to_the_stage_it_names(
    decision: str, rerun: str
) -> None:
    """Each verdict re-enters a different stage, and everything after it."""
    models = {
        "semantic_hypothesizer": ScriptedChatModel(
            responses=tuple(_hypothesis("a plate") for _ in range(4))
        ),
        "semantic_reviewer": ScriptedChatModel(
            responses=tuple(_review("accept", "fine") for _ in range(4))
        ),
        "operation_planner": ScriptedChatModel(
            responses=tuple(_plan("extrude it") for _ in range(4))
        ),
        "operation_reviewer": ScriptedChatModel(
            responses=tuple(_review("accept", "builds it") for _ in range(4))
        ),
        "coder": ScriptedChatModel(
            responses=tuple(AIMessage(content=f"written {n}") for n in range(4))
        ),
    }
    auditor = ScriptedChatModel(
        responses=(
            _audit(decision, "the boss is missing"),
            _audit("accept", "now it matches"),
        )
    )

    with SandboxWorkdir() as workdir:
        graph = _graph(
            workdir,
            models["semantic_hypothesizer"],
            models["semantic_reviewer"],
            models["coder"],
            planner=models["operation_planner"],
            plan_reviewer=models["operation_reviewer"],
            auditor=auditor,
        )
        result = graph.invoke({"messages": []})

    assert result["audit"].decision == "accept"
    assert result["audit_reject_count"] == 1
    # The named stage ran twice, and so did everything downstream of it.
    assert len(models[rerun].received_messages) == 2
    assert len(models["coder"].received_messages) == 2
    # Nothing upstream of it was asked again.
    if rerun == "coder":
        assert len(models["operation_planner"].received_messages) == 1
    if rerun != "semantic_hypothesizer":
        assert len(models["semantic_hypothesizer"].received_messages) == 1


def test_a_stage_the_audit_sends_back_opens_on_its_revise_instruction() -> None:
    planner = ScriptedChatModel(responses=tuple(_plan("extrude it") for _ in range(2)))
    auditor = ScriptedChatModel(
        responses=(
            _audit("redo_operations", "the 10mm hole is never cut"),
            _audit("accept", "now it matches"),
        )
    )

    with SandboxWorkdir() as workdir:
        graph = _graph(
            workdir,
            ScriptedChatModel(responses=(_hypothesis("a plate"),)),
            ScriptedChatModel(responses=(_review("accept", "fine"),)),
            ScriptedChatModel(
                responses=tuple(AIMessage(content="written") for _ in range(2))
            ),
            planner=planner,
            plan_reviewer=ScriptedChatModel(
                responses=tuple(_review("accept", "builds it") for _ in range(2))
            ),
            auditor=auditor,
        )
        graph.invoke({"messages": []})

    first, second = (messages[-1].text for messages in planner.received_messages)
    assert "the 10mm hole is never cut" in second
    assert "the 10mm hole is never cut" not in first


def test_a_stage_whose_premise_was_redone_is_not_asked_the_first_time_question() -> (
    None
):
    """The planner was not at fault, but it planned from a hypothesis that has
    since been replaced, and its own accepted plan is still in the transcript."""
    planner = ScriptedChatModel(responses=tuple(_plan("extrude it") for _ in range(2)))
    auditor = ScriptedChatModel(
        responses=(
            _audit("redo_semantics", "it was read as a plate but it is a bracket"),
            _audit("accept", "now it matches"),
        )
    )

    with SandboxWorkdir() as workdir:
        graph = _graph(
            workdir,
            ScriptedChatModel(responses=tuple(_hypothesis(f"h{n}") for n in range(2))),
            ScriptedChatModel(
                responses=tuple(_review("accept", "fine") for _ in range(2))
            ),
            ScriptedChatModel(
                responses=tuple(AIMessage(content="written") for _ in range(2))
            ),
            planner=planner,
            plan_reviewer=ScriptedChatModel(
                responses=tuple(_review("accept", "builds it") for _ in range(2))
            ),
            auditor=auditor,
        )
        graph.invoke({"messages": []})

    first, second = (messages[-1].text for messages in planner.received_messages)
    assert "has been revised" in second
    assert "has been revised" not in first
    # It is being asked to restate, not to answer for a rejection that was not
    # its own.
    assert "rejected" not in second
    # And the hypothesis it restates from is the new one.
    assert '"h1"' in second


def test_a_coder_the_audit_sends_back_is_asked_for_a_correction() -> None:
    """It already has the hypothesis and the plan; what it lacks is the fault."""
    coder = ScriptedChatModel(
        responses=tuple(AIMessage(content=f"written {n}") for n in range(2))
    )
    auditor = ScriptedChatModel(
        responses=(
            _audit("redo_code", "the pocket is cut on the wrong face"),
            _audit("accept", "now it matches"),
        )
    )

    with SandboxWorkdir() as workdir:
        graph = _graph(
            workdir,
            ScriptedChatModel(responses=(_hypothesis("a plate"),)),
            ScriptedChatModel(responses=(_review("accept", "fine"),)),
            coder,
            auditor=auditor,
        )
        graph.invoke({"messages": []})

    first, second = (messages[-1].text for messages in coder.received_messages)
    assert "reviewed and accepted" in first
    assert "the pocket is cut on the wrong face" in second
    assert "reviewed and accepted" not in second


def test_a_coder_whose_plan_was_redone_is_told_the_plan_changed() -> None:
    """The coder was not at fault, but the program it wrote builds a plan that
    no longer exists."""
    coder = ScriptedChatModel(
        responses=tuple(AIMessage(content=f"written {n}") for n in range(2))
    )
    auditor = ScriptedChatModel(
        responses=(
            _audit("redo_operations", "the plan never cuts the 10mm hole"),
            _audit("accept", "now it matches"),
        )
    )

    with SandboxWorkdir() as workdir:
        graph = _graph(
            workdir,
            ScriptedChatModel(
                responses=tuple(_hypothesis("a plate") for _ in range(2))
            ),
            ScriptedChatModel(
                responses=tuple(_review("accept", "fine") for _ in range(2))
            ),
            coder,
            planner=ScriptedChatModel(
                responses=tuple(_plan(f"plan-v{n}") for n in range(2))
            ),
            plan_reviewer=ScriptedChatModel(
                responses=tuple(_review("accept", "builds it") for _ in range(2))
            ),
            auditor=auditor,
        )
        graph.invoke({"messages": []})

    first, second = (messages[-1].text for messages in coder.received_messages)
    assert "has been revised" in second
    assert "has been revised" not in first
    # It is rebuilding against the new plan, not answering for a fault of its own.
    assert "rejected" not in second
    assert "plan-v1" in second


def test_a_stage_sent_back_does_not_read_the_work_it_invalidated() -> None:
    """The plan and the program that followed were built on a hypothesis the
    audit rejected, so they are an abandoned branch, not the run so far."""
    planner = ScriptedChatModel(responses=tuple(_plan(f"plan-v{n}") for n in range(2)))
    coder = ScriptedChatModel(
        responses=tuple(AIMessage(content=f"written {n}") for n in range(2))
    )

    with SandboxWorkdir() as workdir:
        graph = _graph(
            workdir,
            ScriptedChatModel(responses=tuple(_hypothesis(f"h{n}") for n in range(2))),
            ScriptedChatModel(
                responses=tuple(_review("accept", "fine") for _ in range(2))
            ),
            coder,
            planner=planner,
            plan_reviewer=ScriptedChatModel(
                responses=tuple(_review("accept", "builds it") for _ in range(2))
            ),
            auditor=ScriptedChatModel(
                responses=(
                    _audit("redo_semantics", "it is a bracket, not a plate"),
                    _audit("accept", "now it matches"),
                )
            ),
        )
        graph.invoke({"messages": []})

    speakers = [message.name for message in planner.received_messages[1]]
    assert "coder" not in speakers
    assert "output_auditor" not in speakers
    # Its own earlier plan stays: the stage is upstream of nothing it wrote.
    assert speakers.count("operation_planner") == 1


def test_the_coder_is_not_shown_the_audits_own_turns() -> None:
    """It is told the verdict in its instruction; the audit's reasoning about a
    program it is now replacing is not the run so far."""
    coder = ScriptedChatModel(
        responses=tuple(AIMessage(content=f"written {n}") for n in range(2))
    )

    with SandboxWorkdir() as workdir:
        graph = _graph(
            workdir,
            ScriptedChatModel(responses=(_hypothesis("a plate"),)),
            ScriptedChatModel(responses=(_review("accept", "fine"),)),
            coder,
            auditor=ScriptedChatModel(
                responses=(
                    _audit("redo_code", "the pocket is cut on the wrong face"),
                    _audit("accept", "now it matches"),
                )
            ),
        )
        graph.invoke({"messages": []})

    speakers = [message.name for message in coder.received_messages[1]]
    assert "output_auditor" not in speakers
    assert "the pocket is cut on the wrong face" in coder.received_messages[1][-1].text


def test_the_run_ends_when_the_audit_has_sent_it_back_too_often() -> None:
    auditor = ScriptedChatModel(
        responses=tuple(_audit("redo_code", "still wrong") for _ in range(5))
    )
    coder = ScriptedChatModel(
        responses=tuple(AIMessage(content=f"written {n}") for n in range(5))
    )

    with SandboxWorkdir() as workdir:
        graph = _graph(
            workdir,
            ScriptedChatModel(responses=(_hypothesis("a plate"),)),
            ScriptedChatModel(responses=(_review("accept", "fine"),)),
            coder,
            auditor=auditor,
            max_audit_reject_count=2,
        )
        result = graph.invoke({"messages": []})

    assert result["audit_reject_count"] == 3
    # The first run of the coder plus one for each send-back it was granted.
    assert len(coder.received_messages) == 3
    assert result["audit"].decision == "redo_code"


def test_a_run_that_never_reached_the_coder_is_not_audited() -> None:
    """There is no built model to judge, so the audit would have no evidence."""
    auditor = ScriptedChatModel(responses=(_audit("accept", "unused"),))

    with SandboxWorkdir() as workdir:
        graph = _graph(
            workdir,
            ScriptedChatModel(
                responses=(
                    tool_call("run_shell", {"command": "true"}, "call-loop"),
                    tool_call("run_shell", {"command": "true"}, "call-loop-2"),
                )
            ),
            ScriptedChatModel(responses=(_review("accept", "fine"),)),
            ScriptedChatModel(responses=()),
            agent_options={"max_turns": 2, "announce_turn_budget": False},
            auditor=auditor,
        )
        result = graph.invoke({"messages": []})

    assert "semantic_hypothesis" not in result
    assert auditor.received_messages == []
    assert result["stop_reasons"] == {
        "semantic_hypothesizer": StopReason.BUDGET_EXHAUSTED
    }
