import json
import sys
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any

import pytest
from langchain.agents.structured_output import StructuredOutputValidationError
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from tests.zeroshot.chat_models import (
    ScriptedChatModel,
    tool_call,
    unanswered_tool_calls,
)
from zeroshot.pipeline.messages import ArtifactPresenter, InputManifest
from zeroshot.pipeline.sandbox import SandboxRunner, SandboxWorkdir
from zeroshot.pipeline.tools import VerifyOutputResult
from zeroshot.pipeline.verification import StepRenderer
from zeroshot.pipeline.workflow import (
    FanoutReduceProposal,
    Proposal,
    StopReason,
    create_agent,
    create_fanout_reduce_graph,
)
from zeroshot.pipeline.workflow import graph as graph_module
from zeroshot.pipeline.workflow.components.proposer_reviewer import (
    create_proposer_reviewer_loop,
)
from zeroshot.pipeline.workflow.graph import AgentBuilder, create_reconstruction_graph


def _renderer() -> StepRenderer:
    return StepRenderer(timeout_s=60.0)


def _artifact_presenter() -> ArtifactPresenter:
    return ArtifactPresenter(
        input_render3d_mode="none",
        input_render3d_styles=(),
        feedback_render3d_mode="none",
        feedback_render3d_styles=(),
    )


def _agent(role: str, model: BaseChatModel, **overrides: Any) -> AgentBuilder:
    return partial(create_agent, role=role, model=model, **overrides)


def _proposed(*items: str) -> Proposal:
    return Proposal(proposal=list(items), rationale="the views agree")


def _semantic_proposed(*items: str) -> FanoutReduceProposal:
    return FanoutReduceProposal(proposal=list(items), rationale="the views agree")


def _hypothesis(*semantics: str) -> AIMessage:
    return AIMessage(content=_semantic_proposed(*semantics).model_dump_json())


def _hypothesizer(*semantics: str) -> ScriptedChatModel:
    """A head proposer that answers, then merges the fan-out to that answer.

    The stage needs at least two proposers, so the head is asked twice: once on
    its own branch, and once more as the reducer.  A test that is not about the
    fan-out says what the stage settles on and leaves the merge alone.
    """
    return ScriptedChatModel(
        responses=(_hypothesis(*semantics), _hypothesis(*semantics))
    )


def _peer_hypothesizer() -> ScriptedChatModel:
    """The branch the head has to reconcile, for tests not about its content."""
    return ScriptedChatModel(responses=(_hypothesis("a peer reading"),))


def _silent_hypothesizer(turns: int) -> ScriptedChatModel:
    """A proposer that spends its budget investigating and never answers."""
    return ScriptedChatModel(
        responses=tuple(
            tool_call("run_shell", {"command": "true"}, f"call-{turn}")
            for turn in range(turns)
        )
    )


def _plan(*operations: str) -> AIMessage:
    return AIMessage(content=_proposed(*operations).model_dump_json())


def _review(accept: bool, rationale: str = "") -> AIMessage:
    return AIMessage(content=json.dumps({"accept": accept, "rationale": rationale}))


def _audit(revise: str | None, rationale: str = "") -> AIMessage:
    return AIMessage(content=json.dumps({"revise": revise, "rationale": rationale}))


def _stage(
    proposer_role: str,
    proposer: ScriptedChatModel,
    reviewer_role: str,
    reviewer: ScriptedChatModel,
    max_revisions: int,
    **options: Any,
):
    return partial(
        create_proposer_reviewer_loop,
        proposer_role=proposer_role,
        proposer_model=proposer,
        reviewer_role=reviewer_role,
        reviewer_model=reviewer,
        max_revisions=max_revisions,
        **options,
    )


def _graph(
    workdir: SandboxWorkdir,
    hypothesizer: ScriptedChatModel,
    reviewer: ScriptedChatModel,
    coder: ScriptedChatModel,
    agent_options: dict[str, Any] | None = None,
    *,
    semantic_models: Sequence[ScriptedChatModel] | None = None,
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
    # Nothing here compiles the graph with a checkpointer, so there is none for
    # `checkpointer=True` -- what a run builds its agents with -- to inherit.
    options = {"announce_turns": False, **(agent_options or {})}
    max_turns = options.pop("max_turns", 30)
    stage_options = {
        "max_proposer_turns_per_revision": max_turns,
        "max_reviewer_turns_per_revision": max_turns,
        "announce_turns": options.get("announce_turns", False),
        "model_retries": options.get("model_retries", 5),
        "checkpointer": False,
    }
    agent_options = {
        "max_turns": max_turns,
        "announce_turns": options.get("announce_turns", False),
        "model_retries": options.get("model_retries", 5),
        "checkpointer": False,
    }
    fanout_options = {
        "max_proposer_turns": max_turns,
        "max_reducer_turns": max_turns,
        "announce_turns": options.get("announce_turns", False),
        "model_retries": options.get("model_retries", 5),
        "checkpointer": False,
    }
    max_revisions = overrides.pop("max_revisions", 2)
    dxf_path = workdir.host_bind_dir / "drawing.dxf"
    dxf_path.write_text("0\nSECTION\n0\nEOF\n", encoding="utf-8")
    return create_reconstruction_graph(
        semantics_agent_builder=partial(
            create_fanout_reduce_graph,
            proposer_role="semantic_hypothesizer",
            proposer_models=list(
                semantic_models or [hypothesizer, _peer_hypothesizer()]
            ),
            **fanout_options,
        ),
        operations_agent_builder=_stage(
            "operation_planner",
            planner or ScriptedChatModel(responses=(_plan("extrude the outline"),)),
            "operation_reviewer",
            plan_reviewer or ScriptedChatModel(responses=(_review(True, "fine"),)),
            max_revisions,
            **stage_options,
        ),
        coding_agent_builder=_agent("coder", coder, **agent_options),
        audit_agent_builder=_agent(
            "output_auditor",
            auditor or ScriptedChatModel(responses=(_audit(None, "matches"),)),
            **agent_options,
        ),
        sandbox_runner=SandboxRunner(
            python_executable=Path(sys.executable), default_timeout_s=10
        ),
        sandbox_workdir=workdir,
        renderer=_renderer(),
        artifact_presenter=_artifact_presenter(),
        input_manifest=InputManifest(
            sample_id="test",
            dxf_path=dxf_path,
            render3d_paths={},
        ),
        **overrides,
    )


def _texts(messages: list[BaseMessage]) -> list[str]:
    return [message.text for message in messages]


def _stop_reasons(result: dict[str, Any]) -> dict[str, StopReason]:
    semantics = result.get("semantics_state") or {}
    operations = result.get("operations_state") or {}
    states = {
        "semantic_hypothesizer": semantics.get("reducer_state"),
        "operation_planner": operations.get("proposer_state"),
        "operation_reviewer": operations.get("reviewer_state"),
        "coder": result.get("coding_state"),
        "output_auditor": result.get("audit_state"),
    }
    return {
        role: reason
        for role, state in states.items()
        if state is not None and (reason := state.get("stop_reason")) is not None
    }


def _last_instruction(messages: list[BaseMessage]) -> str:
    return next(
        message.text
        for message in reversed(messages)
        if isinstance(message, HumanMessage) and not message.text.startswith("[turn ")
    )


def test_an_accepted_hypothesis_reaches_the_coder_and_the_final_verification() -> None:
    hypothesizer = _hypothesizer("a flanged boss")
    reviewer = ScriptedChatModel(responses=(_review(True, "matches all views"),))
    coder = ScriptedChatModel(responses=(AIMessage(content="written"),))

    with SandboxWorkdir() as workdir:
        graph = _graph(workdir, hypothesizer, reviewer, coder)
        result = graph.invoke({})

    assert result["semantic_hypothesis"] == _semantic_proposed("a flanged boss")
    # Every role that spoke reported for itself, under its own name.
    assert _stop_reasons(result) == {
        role: StopReason.COMPLETED
        for role in (
            "semantic_hypothesizer",
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

    # Each active stage read what came before it, including the drawing.
    for model in (hypothesizer, coder):
        assert "[Input DXF path:" in model.received_messages[0][1].text
    assert reviewer.received_messages == []
    assert "a flanged boss" in coder.received_messages[0][-1].text


def test_semantics_fans_out_and_reduces_before_operations() -> None:
    head = ScriptedChatModel(
        responses=(
            _hypothesis("a plate"),
            _hypothesis("a plate", "a through hole"),
        )
    )
    peer = ScriptedChatModel(responses=(_hypothesis("a through hole"),))
    coder = ScriptedChatModel(responses=(AIMessage(content="written"),))

    with SandboxWorkdir() as workdir:
        graph = _graph(
            workdir,
            head,
            ScriptedChatModel(responses=()),
            coder,
            semantic_models=[head, peer],
        )
        result = graph.invoke({})

        assert (workdir.host_bind_dir / "semantic_hypothesis_0").is_dir()
        assert (workdir.host_bind_dir / "semantic_hypothesis_1").is_dir()

    assert result["semantic_hypothesis"] == _semantic_proposed(
        "a plate", "a through hole"
    )
    assert len(head.received_messages) == 2
    assert len(peer.received_messages) == 1
    assert "a through hole" in head.received_messages[1][-1].text
    assert "semantic_hypothesis_0" in head.received_messages[0][-1].text
    assert "semantic_hypothesis_1" in peer.received_messages[0][-1].text


def test_each_agent_keeps_a_private_transcript() -> None:
    hypothesizer = _hypothesizer("a plate")
    reviewer = ScriptedChatModel(responses=(_review(True, "fine"),))
    coder = ScriptedChatModel(responses=(AIMessage(content="written"),))

    with SandboxWorkdir() as workdir:
        graph = _graph(workdir, hypothesizer, reviewer, coder)
        result = graph.invoke({})

    states = [
        result["semantics_state"]["reducer_state"],
        result["operations_state"]["proposer_state"],
        result["operations_state"]["reviewer_state"],
        result["coding_state"],
        result["audit_state"],
    ]
    for state in states:
        messages = state["messages"]
        ids = [message.id for message in messages]
        assert len(ids) == len(set(ids))
        assert not [
            message for message in messages if isinstance(message, SystemMessage)
        ]

    assert "semantic_hypothesizer" not in [
        message.name for message in result["coding_state"]["messages"]
    ]


def test_semantics_no_longer_invokes_the_legacy_reviewer() -> None:
    hypothesizer = ScriptedChatModel(
        responses=tuple(_hypothesis("a plate") for _ in range(6))
    )
    reviewer = ScriptedChatModel(
        responses=tuple(_review(False, "still incomplete") for _ in range(6))
    )
    coder = ScriptedChatModel(responses=(AIMessage(content="written"),))

    with SandboxWorkdir() as workdir:
        graph = _graph(workdir, hypothesizer, reviewer, coder, max_revisions=2)
        result = graph.invoke({})

    assert result["semantic_hypothesis"] == _semantic_proposed("a plate")
    assert reviewer.received_messages == []
    assert len(coder.received_messages) == 1


def test_the_coder_is_given_both_settled_artifacts() -> None:
    hypothesizer = _hypothesizer("a flanged boss")
    reviewer = ScriptedChatModel(responses=(_review(True, "fine"),))
    planner = ScriptedChatModel(responses=(_plan("extrude the outline 25mm"),))
    plan_reviewer = ScriptedChatModel(responses=(_review(True, "builds it"),))
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
        result = graph.invoke({})

    assert result["operation_plan"] == _proposed("extrude the outline 25mm")
    # The plan was written against the hypothesis, and the coder was handed
    # both rather than being left to find them in the transcript.
    assert "a flanged boss" in planner.received_messages[0][-1].text
    instruction = coder.received_messages[0][-1].text
    assert "a flanged boss" in instruction
    assert "extrude the outline 25mm" in instruction


def test_a_plan_that_was_never_settled_leaves_the_coder_nothing_to_follow() -> None:
    hypothesizer = _hypothesizer("a plate")
    reviewer = ScriptedChatModel(responses=(_review(True, "fine"),))
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
        result = graph.invoke({})

    assert result["semantic_hypothesis"] == _semantic_proposed("a plate")
    assert result["operation_plan"] is None
    # The planner is the one that ran out; the semantic stage before it did not.
    assert _stop_reasons(result)["operation_planner"] is StopReason.BUDGET_EXHAUSTED
    assert _stop_reasons(result)["semantic_hypothesizer"] is StopReason.COMPLETED
    assert coder.received_messages == []


def test_a_stage_that_never_answered_leaves_the_coder_nothing_to_write() -> None:
    """Its budget went on investigating, so there is no hypothesis to code."""
    hypothesizer = _silent_hypothesizer(4)
    reviewer = ScriptedChatModel(responses=())
    coder = ScriptedChatModel(responses=())

    with SandboxWorkdir() as workdir:
        graph = _graph(
            workdir,
            hypothesizer,
            reviewer,
            coder,
            # Every branch runs out, so the reduction has nothing to fall back
            # on and the stage settles on no hypothesis at all.
            semantic_models=[hypothesizer, _silent_hypothesizer(4)],
            agent_options={"max_turns": 2},
        )
        result = graph.invoke({})

    assert result["semantic_hypothesis"] is None
    assert _stop_reasons(result) == {
        "semantic_hypothesizer": StopReason.BUDGET_EXHAUSTED
    }
    assert reviewer.received_messages == []
    assert coder.received_messages == []
    assert "last_verification" not in result


def test_an_agent_that_finished_without_its_typed_answer_stops_the_run() -> None:
    hypothesizer = ScriptedChatModel(
        responses=(AIMessage(content='{"proposal": "a boss"}'),)
    )
    reviewer = ScriptedChatModel(responses=())
    coder = ScriptedChatModel(responses=())

    with SandboxWorkdir() as workdir:
        graph = _graph(
            workdir,
            hypothesizer,
            reviewer,
            coder,
            agent_options={"model_retries": 0},
        )
        with pytest.raises(StructuredOutputValidationError, match="Proposal"):
            graph.invoke({})

    assert coder.received_messages == []


def test_a_later_agent_does_not_inherit_an_earlier_agents_turn_notices() -> None:
    hypothesizer = ScriptedChatModel(
        responses=(
            tool_call("run_shell", {"command": "true"}, "call-look"),
            _hypothesis("a plate"),
            _hypothesis("a plate"),
        )
    )
    reviewer = ScriptedChatModel(responses=(_review(True, "fine"),))
    coder = ScriptedChatModel(responses=(AIMessage(content="written"),))

    with SandboxWorkdir() as workdir:
        graph = _graph(
            workdir,
            hypothesizer,
            reviewer,
            coder,
            agent_options={"announce_turns": True},
        )
        result = graph.invoke({})

    def notices(messages: list[BaseMessage]) -> list[str]:
        return [
            message.text
            for message in messages
            if isinstance(message, HumanMessage) and message.text.startswith("[turn ")
        ]

    # An agent keeps its own ladder, counting only its own turns, so a stage
    # that inherits four AI turns still opens at turn 1 of its budget.
    # Two on its own branch, and one more when the reduction reopens it: the
    # ladder is per ask, so the merge starts over at turn 1 of the same budget.
    assert notices(hypothesizer.received_messages[-1]) == [
        "[turn 1/30]",
        "[turn 2/30]",
        "[turn 1/30]",
    ]
    assert reviewer.received_messages == []
    assert notices(coder.received_messages[0]) == ["[turn 1/30]"]
    assert len(notices(result["semantics_state"]["reducer_state"]["messages"])) == 3
    assert len(notices(result["coding_state"]["messages"])) == 1


def test_the_coder_is_told_what_its_edit_built_without_asking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The coder has no verification tool: writing the program is the request.

    Asking cost a turn, and the turns went on edits, so the answer arrived on
    the last turn with nothing left to fix it with. Reported here instead, from
    the path back to the model, where it costs no turn at all.
    """
    hypothesizer = _hypothesizer("a plate")
    reviewer = ScriptedChatModel(responses=(_review(True, "fine"),))
    final_report = VerifyOutputResult(
        verification_id="001",
        status="VERIFIED",
        source="result = object()",
        returncode=0,
    )
    invocations: list[str] = []

    class StubVerifier:
        """One verifier serves both callers, as the graph builds only one."""

        def __init__(self, source_path: Path) -> None:
            self.source_path = source_path

        def verify(self) -> tuple[VerifyOutputResult, None]:
            invocations.append("workflow")
            return final_report, None

        def feedback(self) -> list[object]:
            invocations.append("model")
            return [{"type": "text", "text": "VERIFICATION_REPORT"}]

    with SandboxWorkdir() as workdir:
        source_path = workdir.host_bind_dir / "model.py"
        monkeypatch.setattr(
            graph_module, "OutputVerifier", lambda **kwargs: StubVerifier(source_path)
        )
        coder = ScriptedChatModel(
            responses=(
                tool_call(
                    "run_shell",
                    {"command": f"echo result=1 > {workdir.sandbox_bind_dir}/model.py"},
                    "call-write",
                ),
                AIMessage(content="done"),
            )
        )
        graph = _graph(workdir, hypothesizer, reviewer, coder)
        result = graph.invoke({})

    assert invocations == ["model", "workflow"]
    assert result["last_verification"] == final_report
    assert any(
        "VERIFICATION_REPORT" in text
        for text in _texts(result["coding_state"]["messages"])
    )
    assert not any(
        call["name"] == "verify_output"
        for message in result["coding_state"]["messages"]
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    )


def test_the_workflow_runs_the_final_verification_after_a_coder_runs_out() -> None:
    hypothesizer = _hypothesizer("a plate")
    reviewer = ScriptedChatModel(responses=(_review(True, "fine"),))
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
            agent_options={"max_turns": 3, "announce_turns": False},
        )
        result = graph.invoke({})

    assert {
        "semantic_hypothesizer": result["semantics_state"]["reducer_state"][
            "total_turns"
        ],
        "operation_planner": result["operations_state"]["proposer_state"][
            "total_turns"
        ],
        "operation_reviewer": result["operations_state"]["reviewer_state"][
            "total_turns"
        ],
        "coder": result["coding_state"]["total_turns"],
        "output_auditor": result["audit_state"]["total_turns"],
    } == {
        # One turn on its own branch, one more to reduce the fan-out.
        "semantic_hypothesizer": 2,
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
    ("revise", "rerun"),
    [
        ("coding", "coder"),
        ("operations", "operation_planner"),
        ("semantics", "semantic_hypothesizer"),
    ],
)
def test_the_audit_sends_the_run_back_to_the_stage_it_names(
    revise: str, rerun: str
) -> None:
    """Each verdict re-enters a different stage, and everything after it."""
    models = {
        "semantic_hypothesizer": ScriptedChatModel(
            responses=tuple(_hypothesis("a plate") for _ in range(4))
        ),
        "semantic_reviewer": ScriptedChatModel(
            responses=tuple(_review(True, "fine") for _ in range(4))
        ),
        "operation_planner": ScriptedChatModel(
            responses=tuple(_plan("extrude it") for _ in range(4))
        ),
        "operation_reviewer": ScriptedChatModel(
            responses=tuple(_review(True, "builds it") for _ in range(4))
        ),
        "coder": ScriptedChatModel(
            responses=tuple(AIMessage(content=f"written {n}") for n in range(4))
        ),
    }
    auditor = ScriptedChatModel(
        responses=(
            _audit(revise, "the boss is missing"),
            _audit(None, "now it matches"),
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
        result = graph.invoke({})

    assert result["audit"].revise is None
    assert result["audit_reject_count"] == 1
    # The named stage ran twice, and so did everything downstream of it.  The
    # hypothesizer answers one extra ask throughout, because its stage fans out
    # and its head both proposes and reduces; a send-back re-enters at the
    # reduction without fanning out again.
    assert len(models[rerun].received_messages) == (
        3 if rerun == "semantic_hypothesizer" else 2
    )
    assert len(models["coder"].received_messages) == 2
    # Nothing upstream of it was asked again.
    if rerun == "coder":
        assert len(models["operation_planner"].received_messages) == 1
    if rerun != "semantic_hypothesizer":
        assert len(models["semantic_hypothesizer"].received_messages) == 2


def test_a_stage_the_audit_sends_back_opens_on_its_revise_instruction() -> None:
    planner = ScriptedChatModel(responses=tuple(_plan("extrude it") for _ in range(2)))
    auditor = ScriptedChatModel(
        responses=(
            _audit("operations", "the 10mm hole is never cut"),
            _audit(None, "now it matches"),
        )
    )

    with SandboxWorkdir() as workdir:
        graph = _graph(
            workdir,
            _hypothesizer("a plate"),
            ScriptedChatModel(responses=(_review(True, "fine"),)),
            ScriptedChatModel(
                responses=tuple(AIMessage(content="written") for _ in range(2))
            ),
            planner=planner,
            plan_reviewer=ScriptedChatModel(
                responses=tuple(_review(True, "builds it") for _ in range(2))
            ),
            auditor=auditor,
        )
        graph.invoke({})

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
            _audit("semantics", "it was read as a plate but it is a bracket"),
            _audit(None, "now it matches"),
        )
    )

    with SandboxWorkdir() as workdir:
        graph = _graph(
            workdir,
            ScriptedChatModel(
                # Its branch, the reduction that settles on it, and the
                # revision the audit sends it back for.
                responses=(_hypothesis("h0"), _hypothesis("h0"), _hypothesis("h1"))
            ),
            ScriptedChatModel(responses=tuple(_review(True, "fine") for _ in range(2))),
            ScriptedChatModel(
                responses=tuple(AIMessage(content="written") for _ in range(2))
            ),
            planner=planner,
            plan_reviewer=ScriptedChatModel(
                responses=tuple(_review(True, "builds it") for _ in range(2))
            ),
            auditor=auditor,
        )
        graph.invoke({})

    first, second = map(_last_instruction, planner.received_messages)
    assert "has changed" in second
    assert "has changed" not in first
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
            _audit("coding", "the pocket is cut on the wrong face"),
            _audit(None, "now it matches"),
        )
    )

    with SandboxWorkdir() as workdir:
        graph = _graph(
            workdir,
            _hypothesizer("a plate"),
            ScriptedChatModel(responses=(_review(True, "fine"),)),
            coder,
            auditor=auditor,
        )
        graph.invoke({})

    first, second = map(_last_instruction, coder.received_messages)
    assert "current semantic hypothesis" in first
    assert "the pocket is cut on the wrong face" in second
    assert "audit of the program" in second


def test_a_coder_sent_back_after_running_out_carries_a_usable_transcript() -> None:
    """The coder spent its last turn on a tool call and the audit sent it back,
    so the second ask hands the provider the first ask's transcript."""
    coder = ScriptedChatModel(
        responses=(
            tool_call("run_shell", {"command": "true"}, "call-0"),
            tool_call("run_shell", {"command": "true"}, "call-1"),
            AIMessage(content="corrected"),
        )
    )
    auditor = ScriptedChatModel(
        responses=(
            _audit("coding", "the planned fillets are missing"),
            _audit(None, "now it matches"),
        )
    )

    with SandboxWorkdir() as workdir:
        graph = _graph(
            workdir,
            _hypothesizer("a plate"),
            ScriptedChatModel(responses=(_review(True, "fine"),)),
            coder,
            agent_options={"max_turns": 2, "announce_turns": False},
            auditor=auditor,
        )
        result = graph.invoke({})

    assert result["audit"].revise is None
    coding_messages = result["coding_state"]["messages"]
    # Both of the budget's turns reached the tools, the one it ended on included.
    assert sum(isinstance(m, ToolMessage) for m in coding_messages) == 2
    for ask in coder.received_messages:
        assert unanswered_tool_calls(ask) == []
    assert unanswered_tool_calls(coding_messages) == []


def test_a_coder_whose_plan_was_redone_is_told_the_plan_changed() -> None:
    """The coder was not at fault, but the program it wrote builds a plan that
    no longer exists."""
    coder = ScriptedChatModel(
        responses=tuple(AIMessage(content=f"written {n}") for n in range(2))
    )
    auditor = ScriptedChatModel(
        responses=(
            _audit("operations", "the plan never cuts the 10mm hole"),
            _audit(None, "now it matches"),
        )
    )

    with SandboxWorkdir() as workdir:
        graph = _graph(
            workdir,
            ScriptedChatModel(
                responses=tuple(_hypothesis("a plate") for _ in range(2))
            ),
            ScriptedChatModel(responses=tuple(_review(True, "fine") for _ in range(2))),
            coder,
            planner=ScriptedChatModel(
                responses=tuple(_plan(f"plan-v{n}") for n in range(2))
            ),
            plan_reviewer=ScriptedChatModel(
                responses=tuple(_review(True, "builds it") for _ in range(2))
            ),
            auditor=auditor,
        )
        graph.invoke({})

    first, second = map(_last_instruction, coder.received_messages)
    assert "upstream stages have produced" in second
    assert "upstream stages have produced" not in first
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
            ScriptedChatModel(
                # Its branch, the reduction that settles on it, and the
                # revision the audit sends it back for.
                responses=(_hypothesis("h0"), _hypothesis("h0"), _hypothesis("h1"))
            ),
            ScriptedChatModel(responses=tuple(_review(True, "fine") for _ in range(2))),
            coder,
            planner=planner,
            plan_reviewer=ScriptedChatModel(
                responses=tuple(_review(True, "builds it") for _ in range(2))
            ),
            auditor=ScriptedChatModel(
                responses=(
                    _audit("semantics", "it is a bracket, not a plate"),
                    _audit(None, "now it matches"),
                )
            ),
        )
        graph.invoke({})

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
            _hypothesizer("a plate"),
            ScriptedChatModel(responses=(_review(True, "fine"),)),
            coder,
            auditor=ScriptedChatModel(
                responses=(
                    _audit("coding", "the pocket is cut on the wrong face"),
                    _audit(None, "now it matches"),
                )
            ),
        )
        graph.invoke({})

    speakers = [message.name for message in coder.received_messages[1]]
    assert "output_auditor" not in speakers
    assert "the pocket is cut on the wrong face" in _last_instruction(
        coder.received_messages[1]
    )


def test_the_run_ends_when_the_audit_has_sent_it_back_too_often() -> None:
    auditor = ScriptedChatModel(
        responses=tuple(_audit("coding", "still wrong") for _ in range(5))
    )
    coder = ScriptedChatModel(
        responses=tuple(AIMessage(content=f"written {n}") for n in range(5))
    )

    with SandboxWorkdir() as workdir:
        graph = _graph(
            workdir,
            _hypothesizer("a plate"),
            ScriptedChatModel(responses=(_review(True, "fine"),)),
            coder,
            auditor=auditor,
            max_audit_reject_count=2,
        )
        result = graph.invoke({})

    assert result["audit_reject_count"] == 2
    # The first run of the coder plus one for each send-back it was granted.
    assert len(coder.received_messages) == 3
    assert result["audit"].revise == "coding"
    # The budget is spent before the audit runs: the run does not pay for a
    # verdict that no send-back is left to act on.
    assert len(auditor.received_messages) == 2


def test_a_run_that_never_reached_the_coder_is_not_audited() -> None:
    """There is no built model to judge, so the audit would have no evidence."""
    auditor = ScriptedChatModel(responses=(_audit(None, "unused"),))
    hypothesizer = _silent_hypothesizer(2)

    with SandboxWorkdir() as workdir:
        graph = _graph(
            workdir,
            hypothesizer,
            ScriptedChatModel(responses=(_review(True, "fine"),)),
            ScriptedChatModel(responses=()),
            semantic_models=[hypothesizer, _silent_hypothesizer(2)],
            agent_options={"max_turns": 2, "announce_turns": False},
            auditor=auditor,
        )
        result = graph.invoke({})

    assert result["semantic_hypothesis"] is None
    assert auditor.received_messages == []
    assert _stop_reasons(result) == {
        "semantic_hypothesizer": StopReason.BUDGET_EXHAUSTED
    }
