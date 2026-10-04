import sys
from functools import partial
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.content import create_image_block
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from PIL import Image

from tests.zeroshot.chat_models import (
    ScriptedChatModel,
    tool_call,
    unanswered_tool_calls,
)
from tests.zeroshot.contracts import interpretation, view
from zeroshot.pipeline.event_logging import RunEventTransformer
from zeroshot.pipeline.messages.artifact import ArtifactPresenter
from zeroshot.pipeline.messages.manifest import InputManifest, register_view
from zeroshot.pipeline.sandbox import SandboxRunner, SandboxWorkdir
from zeroshot.pipeline.stages._base.prompt import StageInstructions
from zeroshot.pipeline.stages.audit.contracts import (
    AuditFinding,
    AuditRegion,
    RevisionRequest,
    StageOutputRef,
)
from zeroshot.pipeline.stages.coding.progress import ProgressOutputVerifier
from zeroshot.pipeline.stages.coding.stage import create_coding_stage
from zeroshot.pipeline.stages.coding.verify import VerifyOutputResult
from zeroshot.pipeline.stages.contracts import ReconstructionHistory
from zeroshot.pipeline.stages.interpretation.contracts import View
from zeroshot.pipeline.stages.operations.contracts import Operation, OperationPlan
from zeroshot.pipeline.stages.tickets.contracts import (
    StageReport,
    Ticket,
    TicketAnswers,
)
from zeroshot.pipeline.stages.types import PipelineStage
from zeroshot.pipeline.verification.attempts import AttemptStore
from zeroshot.pipeline.verification.drawing_diff.align import AlignmentResult
from zeroshot.pipeline.verification.run_cadquery import (
    CadQueryExecutionReport,
    ExecutionStatus,
)
from zeroshot.pipeline.verification.run_drawing_diff import DrawingDiffReport
from zeroshot.pipeline.workflow import create_agent
from zeroshot.pipeline.workflow.graph import create_reconstruction_graph
from zeroshot.pipeline.workflow.lifecycle import (
    advance_reconstruction,
    start_reconstruction,
)
from zeroshot.pipeline.workflow.middleware.fresh_coding import (
    CHECKPOINT_NAME,
    INSTRUCTIONS_NAME,
    FreshCodingMiddleware,
)
from zeroshot.pipeline.workflow.middleware.output_limit_budget import (
    OutputLimitBudget,
    OutputLimitBudgetExceeded,
)

SOURCE = "ret_base = object()\nresult = ret_base\n"


def answers(ticket="ticket_initial", coding=True):
    return TicketAnswers(
        responses={ticket: "Reviewed ret_base"},
        stage_report=StageReport(
            concerns={"concern_scale": "Scale needs checking"},
            dimension_checks={} if coding else None,
            unticketed_changes={},
        ),
    )


@pytest.fixture
def setup(tmp_path, monkeypatch):
    workdir = SandboxWorkdir(tmp_path)
    (tmp_path / "inputs").mkdir()
    drawing = tmp_path / "inputs/front.png"
    Image.new("RGB", (20, 20), "white").save(drawing)
    ir = interpretation(views=[view("front", scale=0.1, file="/work/inputs/front.png")])
    plan = OperationPlan(
        rationale="Base",
        proposal=[
            Operation(name="op_base", verb="extrude", detail="base", semantics=[])
        ],
    )
    history = start_reconstruction("run_memory", "Build the drawing", [ir.views[0]])
    for artifact in (ir, plan):
        history = advance_reconstruction(
            history, answers(coding=False), workspace_output=artifact
        )
    state = {"reconstruction": history}
    store = AttemptStore(
        workdir, round_source=lambda: state["reconstruction"].snapshots[-1].round
    )
    presenter = ArtifactPresenter(
        input="image", output_renders="path", unmatched="path", intermediates="none"
    )
    instructions = StageInstructions(
        input_artifact=[register_view("view_input", View.FULL_PAGE, drawing)],
        input_presentation_mode="image",
        prompt_context={
            "coding_output_path": "/work/model.py",
            "verification_dir": "/work/attempts",
            "reconstruction_path": "/work/reconstruction.json",
        },
        workdir=workdir,
    )

    def build(verifier):
        identifier, host, sandbox = verifier.attempt_store.issue("coding")
        source = verifier.source_path.read_text()
        (host / "model.py").write_text(source)
        step = host / "output.step"
        step.write_text("mock STEP")
        failed = "# fail" in source
        return VerifyOutputResult(
            verification_id=identifier,
            host_verification_dir=host,
            sandbox_verification_dir=str(sandbox),
            exec_report=CadQueryExecutionReport(
                status=ExecutionStatus.FAILED if failed else ExecutionStatus.VERIFIED,
                returncode=1 if failed else 0,
                source=source,
                step_path=step,
                stderr="kernel failed" if failed else "",
            ),
            drawing_diff_report={
                "view_front": DrawingDiffReport(
                    drawing_path=drawing,
                    projection_path=host / "front.png",
                    alignment=AlignmentResult(
                        backend="directional_chamfer",
                        model="similarity",
                        status="ok",
                        H_drawing_to_projection=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        diagnostics={"scale_calibration": {"status": "applied"}},
                    ),
                    stats={
                        "match_score": 0.5 if "v2" in source else 1.0,
                        "comparison_status": "ok",
                        "provisional": False,
                        "outside_count": 0,
                        "material_error": {"status": "ok"},
                    },
                )
            },
        )

    monkeypatch.setattr(ProgressOutputVerifier, "_build", build)
    original = ProgressOutputVerifier.feedback

    def feedback(verifier):
        return [
            *original(verifier),
            create_image_block(
                source_type="url",
                url="https://example.test/"
                + verifier._last_feedback_report.verification_id
                + ".png",
            ),
        ]

    monkeypatch.setattr(ProgressOutputVerifier, "feedback", feedback)
    verifier = ProgressOutputVerifier(None, workdir, None, None, presenter, store)
    verifier.interpretation, verifier.operations = ir, plan
    middleware = FreshCodingMiddleware(verifier, fingerprint=verifier.source_digest)
    middleware.set_context(state, instructions)

    @tool
    def write(source: str) -> str:
        """Write a trial program."""
        verifier.source_path.write_text(source)
        return "Written"

    @tool
    def measure() -> str:
        """Save an auxiliary measurement."""
        (tmp_path / "scan.json").write_text('{"measured":17}')
        (tmp_path / "tmp/scan.json").write_text('{"measured":17}')
        return "measured 17"

    return state, instructions, verifier, middleware, (write, measure)


def test_valid_writes_replace_only_inference_and_keep_between_write_work(setup):
    state, instructions, verifier, middleware, tools = setup
    model = ScriptedChatModel(
        responses=(
            tool_call("measure", {}, "measure-before"),
            tool_call("write", {"source": SOURCE + "# v1\n"}, "write-v1"),
            tool_call("measure", {}, "measure-after"),
            tool_call("write", {"source": SOURCE + "# fail\n"}, "write-fail"),
            tool_call("write", {"source": SOURCE + "# v2\n"}, "write-v2"),
            AIMessage(content="done"),
        )
    )
    saver = InMemorySaver()
    agent = create_agent(
        "coder",
        model,
        tools,
        max_turns=6,
        checkpointer=saver,
        extra_middleware=[middleware],
    )
    events = []
    config = {"configurable": {"thread_id": "memory-test"}}
    initial = [
        AIMessage(content="stale judgment"),
        instructions.build(
            state, PipelineStage.CODING, append_inputs=True, dimension_inventory="{}"
        ).model_copy(update={"name": INSTRUCTIONS_NAME}),
    ]
    result = agent.stream_events(
        {"messages": initial},
        config=config,
        version="v3",
        transformers=[partial(RunEventTransformer, sink=events.append)],
    ).output
    seen = model.received_messages
    assert any(m.text == "stale judgment" for m in seen[0])  # No initial program.
    assert not any(isinstance(m, (AIMessage, ToolMessage)) for m in seen[2])
    assert any(
        isinstance(m, ToolMessage) and m.tool_call_id == "measure-after"
        for m in seen[3]
    )
    assert any("kernel failed" in m.text for m in seen[4])
    assert any(
        isinstance(m, ToolMessage) and m.tool_call_id == "write-fail" for m in seen[4]
    )
    assert not any(isinstance(m, (AIMessage, ToolMessage)) for m in seen[5])
    latest = next(m for m in seen[5] if m.name == CHECKPOINT_NAME)
    assert "A previous coding pass produced" in latest.text
    assert "Take over from this checkpoint" in latest.text
    assert SOURCE + "# v2" in latest.text
    assert (
        '"verification_id":"002"' in latest.text
        and '"verification_id":"000"' in latest.text
    )
    assert "/work/scan.json" in latest.text and "Scale needs checking" in latest.text
    assert "/work/tmp/" in latest.text
    assert any("[turn 6/6] Final turn" in m.text for m in seen[5])
    assert all(not unanswered_tool_calls(messages) for messages in seen)
    assert result["current_turn"] == result["total_turns"] == 6
    saved = agent.get_state(config).values["messages"]
    assert any(m.text == "stale judgment" for m in saved)
    assert sum(isinstance(m, ToolMessage) for m in saved) == 5
    assert [m.id for m in saved] == [m.id for m in result["messages"]]
    assert middleware._reported == verifier.source_digest()
    metadata = [
        e["data"]
        for e in events
        if e["event"] == "prompt" and e["data"].get("memory_mode")
    ]
    assert len(metadata) == 4 and metadata[-1]["retained_ai_messages"] == 0
    assert metadata[-1]["ordinary_tool_count"] == 0
    assert metadata[-1]["candidate"]["source_sha256"] == verifier.source_digest()


@pytest.mark.parametrize("fresh", [False, True])
def test_stage_flag_and_new_round_baseline_and_validation_instruction(setup, fresh):
    state, instructions, verifier, _, tools = setup
    verifier.source_path.write_text(SOURCE + "# previous round\n")
    completed = advance_reconstruction(
        state["reconstruction"], answers(), workspace_output=verifier.verify()
    )
    finding = AuditFinding(
        name="find_base",
        observation="Check the base",
        evidence=[AuditRegion(file="/work/inputs/front.png", box=(0, 0, 10, 10))],
        backtrace=[],
        revision_request=RevisionRequest(
            action="modify",
            targets=[StageOutputRef(stage=PipelineStage.CODING, name=None)],
            instruction="Check base",
            proposed_names=[],
        ),
        related_ticket_ids=[],
    )
    previous = completed.snapshots[0]
    current = previous.model_copy(
        update={
            "round": 1,
            "last_completed_stage": PipelineStage.OPERATIONS,
            "program_source": None,
            "verification": None,
            "open_tickets": [
                Ticket(
                    ticket_id="ticket_fix",
                    subject=finding,
                    assigned_stages=[PipelineStage.CODING],
                    responses=[],
                )
            ],
            "stage_reports": {
                stage: report
                for stage, report in previous.stage_reports.items()
                if stage != PipelineStage.CODING
            },
        }
    )
    state["reconstruction"] = ReconstructionHistory.model_validate(
        {**dict(completed), "snapshots": [previous, current]}
    )
    model = ScriptedChatModel(
        responses=(
            AIMessage(content=answers("ticket_fix").model_dump_json()),
            tool_call("write", {"source": SOURCE + "# v2\n"}, "write-validation"),
            AIMessage(content=answers("ticket_fix").model_dump_json()),
        )
    )
    stage = create_coding_stage(
        partial(
            create_agent,
            role="coder",
            model=model,
            max_turns=4,
            reset_turns_when_reentrant=False,
            model_retries=0,
        ),
        tools,
        None,
        instructions,
        instructions.prompt_context,
        verifier.attempt_store,
        SandboxRunner(Path(sys.executable), default_timeout_s=30),
        None,
        verifier.artifact_presenter,
        fresh_memory=fresh,
    )
    state["coding_state"] = {
        "messages": [AIMessage(content="stale judgment")],
        "current_turn": 1,
        "total_turns": 17,
    }
    first = stage.run(state, {})["coding_state"]
    assert isinstance(stage.middleware, FreshCodingMiddleware) is fresh
    assert (
        any(m.text == "stale judgment" for m in model.received_messages[0]) is not fresh
    )
    if fresh:
        checkpoint = next(
            m for m in model.received_messages[0] if m.name == CHECKPOINT_NAME
        )
        assert "# previous round" in checkpoint.text and '"round":1' in checkpoint.text
        assert (
            sum(
                b.get("type") == "image"
                for m in model.received_messages[0]
                for b in m.content_blocks
            )
            == 2
        )
    assert (first["current_turn"], first["total_turns"]) == (2, 18)
    state.update(
        coding_state=first, stage_validation_error="current validation instruction"
    )
    second = stage.run(state, {})["coding_state"]
    assert any(
        "current validation instruction" in m.text for m in model.received_messages[-1]
    )
    assert any("reconstruction round 1" in m.text for m in model.received_messages[-1])
    assert (second["current_turn"], second["total_turns"]) == (4, 20)
    if fresh:
        checkpoint = next(
            m for m in model.received_messages[-1] if m.name == CHECKPOINT_NAME
        )
        assert "# v2" in checkpoint.text and "# previous round" not in checkpoint.text


def test_structured_answer_with_tools_is_still_refused_and_pairs_survive(setup):
    _, _, verifier, middleware, tools = setup
    verifier.source_path.write_text(SOURCE)
    checkpoint = middleware.checkpoint_message(middleware.baseline_feedback())
    model = ScriptedChatModel(
        responses=(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "TicketAnswers",
                        "args": answers().model_dump(mode="json"),
                        "id": "answer-mixed",
                        "type": "tool_call",
                    },
                    {
                        "name": "measure",
                        "args": {},
                        "id": "measure-mixed",
                        "type": "tool_call",
                    },
                ],
            ),
            tool_call(
                "TicketAnswers", answers().model_dump(mode="json"), "answer-final"
            ),
        )
    )
    result = create_agent(
        "coder",
        model,
        tools,
        output_schema=TicketAnswers,
        response_format_strategy="tool",
        max_turns=2,
        extra_middleware=[middleware],
    ).invoke({"messages": [checkpoint]})
    assert isinstance(result["structured_response"], TicketAnswers)
    assert any(
        isinstance(m, ToolMessage)
        and m.tool_call_id == "answer-mixed"
        and m.status == "error"
        for m in model.received_messages[-1]
    )
    assert not unanswered_tool_calls(model.received_messages[-1])
    assert model.bound_tool_name_history[-1] == ("TicketAnswers",)
    assert (
        result["current_turn"] == 2 and middleware._reported == verifier.source_digest()
    )


@pytest.mark.parametrize("failure", ["schema", "length"])
def test_retry_feedback_and_shared_budget_are_retained(setup, failure):
    _, _, verifier, middleware, tools = setup
    verifier.source_path.write_text(SOURCE)
    checkpoint = middleware.checkpoint_message(middleware.baseline_feedback())
    valid = AIMessage(content=answers().model_dump_json())
    rejected = AIMessage(
        content="invalid JSON",
        response_metadata={"finish_reason": "length"} if failure == "length" else {},
    )
    model = ScriptedChatModel(responses=(rejected, valid))
    budget = OutputLimitBudget(3, failures=1)
    result = create_agent(
        "coder",
        model,
        tools,
        output_schema=TicketAnswers,
        response_format_strategy="provider",
        max_turns=1,
        model_retries=1,
        extra_middleware=[middleware],
        output_limit_budget=budget,
    ).invoke({"messages": [AIMessage(content="stale judgment"), checkpoint]})
    seen = model.received_messages[-1]
    assert not any(m.text == "stale judgment" for m in seen)
    assert len(seen) > len(model.received_messages[0]) and isinstance(
        seen[-1], HumanMessage
    )
    if failure == "schema":
        assert any(isinstance(m, AIMessage) and m.text == "invalid JSON" for m in seen)
    assert budget.failures == (2 if failure == "length" else 1)
    assert result["current_turn"] == result["total_turns"] == 1
    assert model.bound_tool_name_history[-1] == ()
    assert middleware._reported == verifier.source_digest()


def test_fresh_coder_stops_at_the_existing_output_limit_cap(setup):
    _, _, verifier, middleware, tools = setup
    verifier.source_path.write_text(SOURCE)
    checkpoint = middleware.checkpoint_message(middleware.baseline_feedback())
    model = ScriptedChatModel(
        responses=(
            AIMessage(content="", response_metadata={"finish_reason": "length"}),
            AIMessage(content=answers().model_dump_json()),
        )
    )
    budget = OutputLimitBudget(3, failures=2)
    agent = create_agent(
        "coder",
        model,
        tools,
        output_schema=TicketAnswers,
        max_turns=1,
        model_retries=5,
        extra_middleware=[middleware],
        output_limit_budget=budget,
    )
    with pytest.raises(OutputLimitBudgetExceeded, match="3/3"):
        agent.invoke({"messages": [checkpoint]})
    assert budget.failures == 3
    assert len(model.received_messages) == 1


@pytest.mark.parametrize("fresh", [False, True])
def test_graph_applies_fresh_memory_only_to_coder_and_shares_the_budget(setup, fresh):
    _, instructions, verifier, _, _ = setup
    captured = {}

    def builder(role, **kwargs):
        captured[role] = kwargs
        return create_agent(role, ScriptedChatModel(responses=()), **kwargs)

    create_reconstruction_graph(
        *[
            partial(builder, role=role, max_turns=1)
            for role in (
                "drawing_interpreter",
                "operation_planner",
                "coder",
                "output_auditor",
            )
        ],
        sandbox_runner=SandboxRunner(Path(sys.executable), default_timeout_s=30),
        sandbox_workdir=verifier.workdir,
        artifact_presenter=verifier.artifact_presenter,
        input_manifest=InputManifest(
            sample_id="memory", drawing=instructions.input_artifact
        ),
        fresh_coder=fresh,
    )
    for role, kwargs in captured.items():
        uses_fresh = any(
            isinstance(middleware, FreshCodingMiddleware)
            for middleware in kwargs["extra_middleware"]
        )
        assert uses_fresh is (fresh and role == "coder")
    assert set(captured) == {
        "drawing_interpreter",
        "operation_planner",
        "coder",
        "output_auditor",
    }
    assert len({id(kwargs["output_limit_budget"]) for kwargs in captured.values()}) == 1
