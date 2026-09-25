import base64
import hashlib
import json
import sys
from itertools import pairwise
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from omegaconf import OmegaConf
from PIL import Image

from tests.zeroshot.chat_models import (
    ScriptedChatModel,
    tool_call,
    unanswered_tool_calls,
)
from tests.zeroshot.contracts import interpretation, view
from zeroshot import interactive_debug as debug
from zeroshot.pipeline.stages.audit.contracts import AuditReport, AuditSubmission
from zeroshot.pipeline.stages.coding.verify import VerifyOutputResult
from zeroshot.pipeline.stages.operations.contracts import Operation, OperationPlan
from zeroshot.pipeline.stages.tickets.contracts import (
    StageReport,
    TicketAnswers,
    TicketResponse,
)
from zeroshot.pipeline.stages.types import PipelineStage
from zeroshot.pipeline.verification import ExecutionStatus
from zeroshot.pipeline.verification.run_cadquery import CadQueryExecutionReport
from zeroshot.pipeline.verification.run_render import (
    ProjectionPaths,
    Render3dPaths,
    RenderReport,
    RenderStatus,
)
from zeroshot.pipeline.workflow.lifecycle import start_reconstruction
from zeroshot.pipeline.workflow.state import ReconstructionState


def make_run(tmp_path: Path) -> tuple[Path, list]:
    """A real checkpoint DB with a submission, compaction, and a later audit."""
    run = tmp_path / "run"
    work = run / "workspace"
    (work / "inputs").mkdir(parents=True)
    picture = work / "inputs/page.png"
    Image.new("RGB", (12, 8), "white").save(picture)
    attempt = work / "attempts/round_000/coding/000"
    (attempt / "projection").mkdir(parents=True)
    (attempt / "output.step").write_text("saved STEP")
    (attempt / "projection/front.dxf").write_text("saved DXF")
    Image.new("RGB", (12, 8), "black").save(attempt / "projection/front.png")

    history = start_reconstruction(
        "run_test", "Build the part", [view("full_page", file="/work/inputs/page.png")]
    )
    snapshot = history.snapshots[-1]
    snapshot.last_completed_stage = PipelineStage.OPERATIONS
    snapshot.interpretation = interpretation("block")
    snapshot.interpretation.views[0].file = "/work/inputs/page.png"
    snapshot.operations = OperationPlan(
        proposal=[
            Operation(
                name="op_block",
                verb="extrude",
                detail="Build a block",
                semantics=["sem_feature_1"],
            )
        ],
        rationale="One block",
    )
    snapshot.open_tickets[0].responses = [
        TicketResponse(ticket_id="ticket_initial", stage=stage, summary="Done")
        for stage in (PipelineStage.INTERPRETATION, PipelineStage.OPERATIONS)
    ]
    completed = history.model_copy(deep=True)
    final = completed.snapshots[-1]
    final.last_completed_stage = PipelineStage.CODING
    final.program_source = "result = 'original submission'\n"
    final.open_tickets[0].responses.append(
        TicketResponse(ticket_id="ticket_initial", stage="coding", summary="Done")
    )
    final.verification = VerifyOutputResult(
        verification_id="000",
        host_verification_dir=attempt,
        sandbox_verification_dir="/work/attempts/round_000/coding/000",
        exec_report=CadQueryExecutionReport(
            status=ExecutionStatus.VERIFIED, step_path=attempt / "output.step"
        ),
        render_report={
            "result": RenderReport(
                RenderStatus.OK,
                ProjectionPaths(front=attempt / "projection/front.dxf"),
                Render3dPaths(),
            )
        },
    )
    answer = TicketAnswers(
        responses={"ticket_initial": "Done"},
        stage_report=StageReport(
            concerns={}, dimension_checks={}, unticketed_changes={}
        ),
    )
    image = "data:image/png;base64," + base64.b64encode(picture.read_bytes()).decode()
    messages = [
        HumanMessage(content="Original upstream context"),
        tool_call("load_image", {"image_path": "/work/inputs/page.png"}, "old_image"),
        ToolMessage(
            content=[{"type": "image_url", "image_url": {"url": image}}],
            tool_call_id="old_image",
        ),
        tool_call("TicketAnswers", answer.model_dump(mode="json"), "submission"),
        ToolMessage(content="Submission received.", tool_call_id="submission"),
    ]
    graph = StateGraph(ReconstructionState)
    graph.add_node(
        "coding",
        lambda _: {
            "coding_state": {
                "messages": messages,
                "structured_response": answer,
                "current_turn": 20,
            },
            "stage_submission": answer,
        },
    )
    graph.add_node(
        "integrate_stage_submission",
        lambda _: {
            "reconstruction": completed,
            "stage_submission": None,
            "stage_validation_error": None,
        },
    )
    graph.add_node(
        "coding_handover",
        lambda _: {
            "coding_state": {
                "messages": [HumanMessage(content="Compacted later")],
                "structured_response": answer,
            }
        },
    )
    audit = AuditReport(ticket_reviews={}, findings=[], concern_reviews={})
    graph.add_node(
        "audit",
        lambda _: {
            "audit_state": {
                "messages": [
                    HumanMessage(content="Audit question"),
                    AIMessage(content="Audit submitted"),
                ],
                "structured_response": AuditSubmission(accepted=True),
            },
            "audit_report": audit,
        },
    )
    graph.add_node("integrate_audit_report", lambda _: {"audit_report": None})
    nodes = [
        START,
        "coding",
        "integrate_stage_submission",
        "coding_handover",
        "audit",
        "integrate_audit_report",
        END,
    ]
    for left, right in pairwise(nodes):
        graph.add_edge(left, right)
    database = run / "checkpoints.sqlite"
    with SqliteSaver.from_conn_string(str(database)) as saver:
        saver.serde = debug.serializer()
        graph.compile(checkpointer=saver).invoke(
            {"reconstruction": history}, {"configurable": {"thread_id": "source"}}
        )
    (work / "reconstruction.json").write_text(completed.model_dump_json())
    (work / "model.py").write_text("FUTURE SUBMISSION MUST NOT LEAK")
    (work / "interpretation.json").write_text("future")
    (work / "operations.json").write_text("future")
    (work / "future.png").write_bytes(picture.read_bytes())
    config = {
        "model": {"_target_": "tests.zeroshot.chat_models.ScriptedChatModel"},
        "workflow": {
            "share_thread": False,
            "diff_drawer_config": None,
            "show_intermediate_returns": False,
            "coding_agent_builder": {
                "role": "coder",
                "model": "${model}",
                "model_retries": 0,
            },
        },
        "sandbox_runner": {
            "python_executable": sys.executable,
            "default_timeout_s": 30,
        },
        "artifact_presenter": {"feedback_mode": "path"},
    }
    (run / ".hydra").mkdir()
    OmegaConf.save(OmegaConf.create(config), run / ".hydra/config.yaml")
    (run / "events.jsonl").write_text(
        json.dumps(
            {
                "event": "prompt",
                "namespace": ["coding:abc"],
                "data": {"system": "Original system prompt"},
            }
        )
        + "\n"
    )
    return database, messages


def test_selects_before_compaction_and_preserves_audit_report(tmp_path):
    database, messages = make_run(tmp_path)
    before = hashlib.sha256(database.read_bytes()).digest()
    source = debug.select_checkpoint(database, "coding", 0)
    assert source.values["coding_state"]["messages"] == messages
    assert not unanswered_tool_calls(source.values["coding_state"]["messages"])
    assert source.values.get("audit_report") is None
    assert (
        debug.select_checkpoint(database, "coding", 0, source.checkpoint_id) == source
    )
    audit = debug.select_checkpoint(database, "audit", 0)
    assert audit.values["audit_report"] is not None
    assert audit.values["audit_state"]["messages"][-1].text == "Audit submitted"
    with pytest.raises(ValueError, match="found none"):
        debug.select_checkpoint(database, "coding", 1)
    assert hashlib.sha256(database.read_bytes()).digest() == before


def test_workspace_contains_selected_assets_not_submitted_or_future_files(tmp_path):
    database, _ = make_run(tmp_path)
    source = debug.select_checkpoint(database, "coding", 0)
    work = tmp_path / "new_workspace"
    assert debug.prepare_workspace(source, database.parent / "workspace", work) == []
    assert (work / "inputs/page.png").is_file()
    assert (work / "attempts/round_000/coding/000/projection/front.png").is_file()
    assert (work / "attempts/round_000/coding/000/output.step").is_file()
    assert not any(
        (work / name).exists()
        for name in ["model.py", "interpretation.json", "operations.json", "future.png"]
    )
    record = json.loads((work / "reconstruction.json").read_text())
    assert (
        record["snapshots"][-1]["program_source"] == "result = 'original submission'\n"
    )
    assert record["snapshots"][-1]["verification"]["exec_report"][
        "step_path"
    ].startswith("/work/")
    # A fresh reset branch also excludes files created by the previous dialogue.
    (work / "trial.py").write_text("modified")
    reset = tmp_path / "reset_workspace"
    debug.prepare_workspace(source, database.parent / "workspace", reset)
    assert not (reset / "trial.py").exists()


def test_dialogue_tools_history_and_reset_without_submission_or_turn_limit(
    tmp_path, monkeypatch, capsys
):
    database, messages = make_run(tmp_path)
    original_hash = hashlib.sha256(database.read_bytes()).digest()
    source_work = database.parent / "workspace"
    original_files = {
        p.relative_to(source_work): p.read_bytes()
        for p in source_work.rglob("*")
        if p.is_file()
    }
    # More than the old 20-turn budget, followed by a second human question.
    first = ScriptedChatModel(
        responses=(
            *(
                tool_call("run_shell", {"command": "true"}, f"run_{i}")
                for i in range(22)
            ),
            tool_call(
                "load_image", {"image_path": "/work/inputs/page.png"}, "new_image"
            ),
            tool_call(
                "run_shell",
                {
                    "command": "python -c \"from pathlib import Path; Path('trial.py').write_text('candidate')\""
                },
                "trial",
            ),
            tool_call("run_shell", {"command": "cp trial.py model.py"}, "candidate"),
            AIMessage(content="First answer"),
            AIMessage(content="Follow-up answer"),
        )
    )
    second = ScriptedChatModel(responses=(AIMessage(content="Independent answer"),))
    models = iter([first, second])
    monkeypatch.setattr(debug, "instantiate", lambda _: next(models))
    builds = []

    def feedback(verifier):
        builds.append(verifier.source_path.read_text())
        return [{"type": "text", "text": "Candidate preview from auto verification"}]

    monkeypatch.setattr(debug.OutputVerifier, "feedback", feedback)
    questions = iter(
        [
            "First question",
            "Follow-up question",
            "/reset",
            "Independent question",
            "/quit",
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _: next(questions))
    output = tmp_path / "session"
    debug.main(
        [
            "--run",
            str(database),
            "--stage",
            "coder",
            "--round",
            "000",
            "--output-dir",
            str(output),
        ]
    )
    assert first.received_messages[0][0].text == "Original system prompt"
    restored = first.received_messages[0][1 : 1 + len(messages)]
    assert [m.model_dump(exclude={"id"}) for m in restored] == [
        m.model_dump(exclude={"id"}) for m in messages
    ]
    assert not unanswered_tool_calls(first.received_messages[-1])
    assert all("TicketAnswers" not in names for names in first.bound_tool_name_history)
    assert not any(
        m.text.startswith("[turn ") for call in first.received_messages for m in call
    )
    last = "\n".join(m.text for m in first.received_messages[-1])
    assert "First answer" in last and "Follow-up question" in last
    assert "Candidate preview from auto verification" in last
    assert builds == ["candidate"]
    reset_text = "\n".join(m.text for m in second.received_messages[0])
    assert "First question" not in reset_text and "Independent question" in reset_text
    assert (output / "branch_000/workspace/trial.py").is_file()
    assert not (output / "branch_001/workspace/trial.py").exists()
    assert not (output / "branch_001/workspace/model.py").exists()
    assert not (database.parent / "workspace/trial.py").exists()
    assert hashlib.sha256(database.read_bytes()).digest() == original_hash
    assert original_files == {
        p.relative_to(source_work): p.read_bytes()
        for p in source_work.rglob("*")
        if p.is_file()
    }
    # Dialogue history belongs to its checkpoint, not either reconstruction file.
    assert (output / "branch_000/workspace/reconstruction.json").read_bytes() == (
        output / "branch_001/workspace/reconstruction.json"
    ).read_bytes()
    terminal = capsys.readouterr().err
    assert "[node]" not in terminal
    assert "[model]" in terminal and "[tool]" in terminal
    for branch in ["branch_000", "branch_001"]:
        events = [
            json.loads(line)
            for line in (output / branch / "events.jsonl").read_text().splitlines()
        ]
        assert events[-1]["event"] == "run_completed"
        assert any(e["event"] == "user_question" for e in events)
        assert any(
            e["event"] == "node_started"
            and e["data"]["node"] == "PromptLogMiddleware.before_agent"
            for e in events
        )
        assert (output / branch / "checkpoints.sqlite").is_file()


def test_source_asset_symlink_cannot_copy_files_outside_workspace(tmp_path):
    database, _ = make_run(tmp_path)
    source = debug.select_checkpoint(database, "coding", 0)
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    image = database.parent / "workspace/inputs/page.png"
    image.unlink()
    image.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        debug.prepare_workspace(source, database.parent / "workspace", tmp_path / "new")
