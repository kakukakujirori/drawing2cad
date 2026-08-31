import json
import shlex
import sys
from collections.abc import Mapping
from functools import partial
from inspect import cleandoc
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from rich.console import Console

from tests.zeroshot.chat_models import ScriptedChatModel
from tests.zeroshot.contracts import hypothesis, replacing, unchanged
from zeroshot.evaluation.aggregate_run import read_events
from zeroshot.pipeline.event_logging import ConsoleReporter, has_run_completed
from zeroshot.pipeline.messages import ArtifactPresenter, InputManifest
from zeroshot.pipeline.messages.contracts import (
    Operation,
    OperationPlan,
    OperationVerb,
)
from zeroshot.pipeline.messages.contracts.reconstruction import (
    CodingSubmission,
    OperationSubmission,
    SemanticSubmission,
    TicketResponse,
)
from zeroshot.pipeline.runner import (
    GraphFactory,
    PipelineRunner,
    _latest_program_source,
)
from zeroshot.pipeline.sandbox import SandboxRunner
from zeroshot.pipeline.verification import (
    CadQueryExecutor,
    ExecutionStatus,
    StepRenderer,
    VerifyOutputResult,
)
from zeroshot.pipeline.workflow import (
    StopReason,
    create_agent,
    create_reconstruction_graph,
)
from zeroshot.pipeline.workflow.reconstruction import (
    advance_reconstruction,
    save_reconstruction,
    start_reconstruction,
)


def _agent(role: str, model: BaseChatModel, **overrides: Any):
    return partial(create_agent, role=role, model=model, **overrides)


_ACCEPTED_AUDIT = AIMessage(content='{"accepted": true, "findings": []}')


def _ticket_response(stage: str, summary: str) -> TicketResponse:
    return TicketResponse(
        ticket_id="ticket_initial",
        stage=stage,  # type: ignore[arg-type]
        summary=summary,
    )


_A_BOX = AIMessage(
    content=SemanticSubmission(
        **replacing(hypothesis("a box")),
        responses=[_ticket_response("semantics", "Established sem_feature_1.")],
    ).model_dump_json()
)


def _semantic_stage():
    return partial(
        create_agent,
        role="semantic_hypothesizer",
        model=ScriptedChatModel(responses=(_A_BOX,)),
        announce_turns=False,
    )


def _operations_stage():
    submission = OperationSubmission(
        **replacing(
            OperationPlan(
                proposal=[
                    Operation(
                        name="op_base",
                        verb=OperationVerb.EXTRUDE,
                        detail="extrude it",
                        depends_on=[],
                        semantics=["sem_feature_1"],
                    )
                ],
                rationale="one extrude",
            )
        ),
        responses=[_ticket_response("operations", "Established op_base.")],
    )
    return _agent(
        "operation_planner",
        ScriptedChatModel(responses=(AIMessage(content=submission.model_dump_json()),)),
        announce_turns=False,
    )


_CODING_ANSWER = AIMessage(
    content=CodingSubmission(
        **unchanged(),
        responses=[_ticket_response("coding", "Implemented ret_base and result.")],
    ).model_dump_json()
)


def _graph_factory(
    model: BaseChatModel,
    *,
    max_stage_validation_retries: int = 3,
    **agent_overrides: Any,
) -> GraphFactory:
    """The staged graph with a cast bound. A cast is a graph's own setting,
    so a run's config -- or a test -- binds it before the runner ever sees it."""
    return partial(
        create_reconstruction_graph,
        semantics_agent_builder=_semantic_stage(),
        operations_agent_builder=_operations_stage(),
        coding_agent_builder=_agent("coder", model, **agent_overrides),
        audit_agent_builder=_agent(
            "output_auditor",
            ScriptedChatModel(responses=(_ACCEPTED_AUDIT,)),
            announce_turns=False,
        ),
        max_stage_validation_retries=max_stage_validation_retries,
    )


VALID_BOX_SOURCE = """\
import cadquery as cq

ret_base = cq.Workplane("XY").box(10, 20, 30)
result = ret_base
"""


def _verified_resume_run():
    run = start_reconstruction("run_sample", "Reconstruct the drawing.")
    run = advance_reconstruction(
        run,
        SemanticSubmission(
            **replacing(hypothesis("a box")),
            responses=[_ticket_response("semantics", "Established sem_feature_1.")],
        ),
    )
    run = advance_reconstruction(
        run,
        OperationSubmission(
            **replacing(
                OperationPlan(
                    proposal=[
                        Operation(
                            name="op_base",
                            verb=OperationVerb.EXTRUDE,
                            detail="extrude it",
                            depends_on=[],
                            semantics=["sem_feature_1"],
                        )
                    ],
                    rationale="one extrude",
                )
            ),
            responses=[_ticket_response("operations", "Established op_base.")],
        ),
    )
    run = advance_reconstruction(
        run,
        CodingSubmission(
            **unchanged(),
            responses=[_ticket_response("coding", "Implemented ret_base and result.")],
        ),
        verification=VerifyOutputResult(
            verification_id="007",
            status=ExecutionStatus.VERIFIED,
            source=VALID_BOX_SOURCE,
            returncode=0,
        ),
    )

    return run


def test_resume_copies_an_external_attempt_directly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _verified_resume_run()

    source_workspace = tmp_path / "source" / "workspace"
    attempt = source_workspace / "attempts" / "007"
    attempt.mkdir(parents=True)
    (attempt / "output.step").write_bytes(b"STEP")
    resume_path = source_workspace / "reconstruction.json"
    save_reconstruction(resume_path, run)

    def reject_temporary_directory(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("an external resume source needs no temporary copy")

    monkeypatch.setattr(
        "zeroshot.pipeline.runner.tempfile.TemporaryDirectory",
        reject_temporary_directory,
    )
    artifact_root = tmp_path / "destination"
    runner = _runner_for_rerun(
        artifact_root,
        "retry",
        resume_from=resume_path,
    )
    sample_root = artifact_root / "sample"

    workspace = runner._prepare_workspace(
        sample_root,
        sample_root / "events.jsonl",
        run,
    )

    assert _latest_program_source(run) == VALID_BOX_SOURCE
    assert (workspace / "model.py").read_text(encoding="utf-8") == VALID_BOX_SOURCE
    assert (workspace / "attempts" / "007" / "output.step").read_bytes() == b"STEP"
    assert (attempt / "output.step").is_file()


def test_resume_temporarily_protects_an_attempt_cleared_by_retry(
    tmp_path: Path,
) -> None:
    run = _verified_resume_run()
    artifact_root = tmp_path / "artifacts"
    sample_root = artifact_root / "sample"
    workspace = sample_root / "workspace"
    attempt = workspace / "attempts" / "007"
    attempt.mkdir(parents=True)
    (attempt / "output.step").write_bytes(b"STEP")
    (workspace / "stale.txt").write_text("stale", encoding="utf-8")
    resume_path = workspace / "reconstruction.json"
    save_reconstruction(resume_path, run)
    events_path = sample_root / "events.jsonl"
    events_path.write_text('{"event":"run_started"}\n', encoding="utf-8")
    runner = _runner_for_rerun(
        artifact_root,
        "retry",
        resume_from=resume_path,
    )

    prepared = runner._prepare_workspace(sample_root, events_path, run)

    assert not (prepared / "stale.txt").exists()
    assert (prepared / "model.py").read_text(encoding="utf-8") == VALID_BOX_SOURCE
    assert (prepared / "attempts" / "007" / "output.step").read_bytes() == b"STEP"


def _final_verification(result: Mapping[str, Any]) -> VerifyOutputResult | None:
    return result["reconstruction"].snapshots[-1].verification


def _message_text(message: BaseMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return "\n".join(
        block["text"]
        for block in message.content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _write_text_command(filename: str, content: str) -> str:
    script = (
        "from pathlib import Path; "
        f"Path({filename!r}).write_text({content!r}, encoding='utf-8')"
    )
    return f"python -c {shlex.quote(script)}"


def _manifest_without_renders(tmp_path: Path, sample_id: str) -> InputManifest:
    dxf_path = tmp_path / f"{sample_id}.dxf"
    dxf_path.write_text("DXF_FIXTURE", encoding="utf-8")
    return InputManifest(
        sample_id=sample_id,
        dxf_path=dxf_path,
        render3d_paths={},
    )


def _artifact_presenter_without_renders() -> ArtifactPresenter:
    return ArtifactPresenter(
        input_render3d_mode="none",
        input_render3d_styles=(),
        feedback_render3d_mode="none",
        feedback_render3d_styles=(),
    )


def _sandbox_runner() -> SandboxRunner:
    return SandboxRunner(
        python_executable=Path(sys.executable),
        default_timeout_s=30,
    )


def _renderer() -> StepRenderer:
    return StepRenderer(timeout_s=120.0)


def test_run_sample_stages_only_allowed_inputs_and_preserves_workdir(
    tmp_path: Path,
) -> None:
    dxf_path = tmp_path / "source.dxf"
    selected_render_path = tmp_path / "selected.png"
    hidden_render_path = tmp_path / "hidden.png"
    dxf_path.write_text("ORIGINAL_DXF", encoding="utf-8")
    selected_render_path.write_bytes(b"ALLOWED_RENDER")
    hidden_render_path.write_bytes(b"HIDDEN_RENDER")

    manifest = InputManifest(
        sample_id="sample-1",
        dxf_path=dxf_path,
        render3d_paths={
            "style-a": selected_render_path,
            "style-b": hidden_render_path,
        },
    )
    inspect_inputs = cleandoc(
        """
        from pathlib import Path

        dxf = Path('/work/inputs/techdraw.dxf')
        assert dxf.read_text() == 'ORIGINAL_DXF'
        assert Path('/work/inputs/style-a.png').read_bytes() == b'ALLOWED_RENDER'
        assert not Path('/work/inputs/style-b.png').exists()
        try:
            dxf.write_text('SANDBOX_MUTATION')
        except OSError:
            pass
        else:
            raise AssertionError('sandbox input must be read-only')
        Path('/work/scratch.txt').write_text('persisted')
        Path('/work/events.jsonl').write_text('FORGED')
        print('staged-ok')
        """
    )
    model = ScriptedChatModel(
        responses=(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "run_shell",
                        "args": {"command": f"python -c {shlex.quote(inspect_inputs)}"},
                        "id": "call-inspect-inputs",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "run_shell",
                        "args": {"command": "cat /work/scratch.txt"},
                        "id": "call-read-scratch",
                        "type": "tool_call",
                    }
                ],
            ),
            _CODING_ANSWER,
        )
    )
    artifact_presenter = ArtifactPresenter(
        input_render3d_mode="path",
        input_render3d_styles=("style-a",),
        feedback_render3d_mode="none",
        feedback_render3d_styles=(),
    )
    runner = PipelineRunner(
        # This test is about input staging and transcript contents, not budget
        # announcements, so keep those extra HumanMessages out of its fixture.
        graph_factory=_graph_factory(
            model, announce_turns=False, max_stage_validation_retries=0
        ),
        artifact_presenter=artifact_presenter,
        sandbox_runner=SandboxRunner(
            python_executable=Path(sys.executable),
            default_timeout_s=10,
        ),
        artifact_root=tmp_path / "artifacts",
        renderer=_renderer(),
    )

    result = runner.run_sample(manifest)

    assert result is not None
    assert model.bound_tool_names == (
        "run_shell",
        "load_image",
    )
    assert len(model.received_messages) == 3

    # The coder opens on the workflow transcript: its own prompt, the run's
    # input, then what the semantic stage made of it.
    initial_messages = model.received_messages[0]
    initial_human_message = initial_messages[1]
    assert isinstance(initial_human_message, HumanMessage)
    initial_text = _message_text(initial_human_message)
    assert "/work/inputs/techdraw.dxf" in initial_text
    assert "/work/inputs/style-a.png" in initial_text
    assert "style-b" not in initial_text
    assert str(dxf_path) not in initial_text
    assert str(selected_render_path) not in initial_text
    assert str(hidden_render_path) not in initial_text

    # The transcript belongs to the agent, so what the model was last handed is
    # where the run's conversation is read back from.
    messages = model.received_messages[-1]
    assert [type(message) for message in messages] == [
        *[type(message) for message in initial_messages],
        AIMessage,
        ToolMessage,
        AIMessage,
        ToolMessage,
    ]

    inspect_result = messages[-3]
    assert isinstance(inspect_result, ToolMessage)
    assert inspect_result.tool_call_id == "call-inspect-inputs"
    assert isinstance(inspect_result.content, str)
    assert json.loads(inspect_result.content) == {
        "status": "COMPLETED",
        "returncode": 0,
        "stdout": "staged-ok\n",
        "stderr": "",
    }

    persistence_result = messages[-1]
    assert isinstance(persistence_result, ToolMessage)
    assert persistence_result.tool_call_id == "call-read-scratch"
    assert isinstance(persistence_result.content, str)
    assert json.loads(persistence_result.content) == {
        "status": "COMPLETED",
        "returncode": 0,
        "stdout": "persisted",
        "stderr": "",
    }

    assert result["coding_state"]["stop_reason"] is StopReason.COMPLETED

    assert dxf_path.read_text(encoding="utf-8") == "ORIGINAL_DXF"
    assert selected_render_path.read_bytes() == b"ALLOWED_RENDER"
    assert hidden_render_path.read_bytes() == b"HIDDEN_RENDER"

    sample_artifact_root = tmp_path / "artifacts" / "sample-1"
    saved_workdir = sample_artifact_root / "workspace"
    assert (saved_workdir / "inputs" / "techdraw.dxf").read_text(
        encoding="utf-8"
    ) == "ORIGINAL_DXF"
    assert (saved_workdir / "inputs" / "style-a.png").read_bytes() == b"ALLOWED_RENDER"
    assert not (saved_workdir / "inputs" / "style-b.png").exists()
    assert (saved_workdir / "scratch.txt").read_text(encoding="utf-8") == "persisted"
    assert (saved_workdir / "attempts").is_dir()

    assert (saved_workdir / "events.jsonl").read_text(encoding="utf-8") == "FORGED"

    event_log_path = sample_artifact_root / "events.jsonl"
    events = [
        json.loads(line)
        for line in event_log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event_index"] for event in events] == list(range(len(events)))
    assert events[0]["event"] == "run_started"
    assert events[-1]["event"] == "run_completed"
    assert {event["sample_id"] for event in events} == {"sample-1"}
    assert len({event["run_id"] for event in events}) == 1
    assert sum(event["event"] == "input" for event in events) == 1

    tool_events = [
        event
        for event in events
        if event["event"] in {"tool_started", "tool_finished"}
        and event["data"].get("tool_call_id")
        in {"call-inspect-inputs", "call-read-scratch"}
    ]
    assert [event["event"] for event in tool_events] == [
        "tool_started",
        "tool_finished",
        "tool_started",
        "tool_finished",
    ]

    # The workflow's own final verification is a node, not a tool the coder
    # called; what it produced is reported under its own event.
    final_verification = next(
        event for event in events if event["event"] == "verification"
    )
    assert final_verification["data"]["node"] == "integrate_stage_submission"
    audit = next(
        event
        for event in events
        if event["event"] == "audit" and event["data"]["report"] is not None
    )
    assert audit["data"] == {
        "node": "audit",
        "report": {"accepted": True, "findings": []},
    }
    assert "verify_output" not in {event["data"].get("tool_name") for event in events}
    assert "output" not in {event["event"] for event in events}

    checkpoint_path = sample_artifact_root / "checkpoints.sqlite"
    assert checkpoint_path.is_file()
    with SqliteSaver.from_conn_string(str(checkpoint_path)) as checkpointer:
        checkpoints = list(checkpointer.list(None))

    assert checkpoints
    thread_ids = {
        checkpoint.config["configurable"]["thread_id"] for checkpoint in checkpoints
    }
    assert len(thread_ids) == 1
    assert thread_ids == {events[0]["run_id"]}


def test_run_sample_preserves_workdir_when_graph_fails(tmp_path: Path) -> None:
    manifest = _manifest_without_renders(tmp_path, "failed-run")
    model = ScriptedChatModel(
        responses=(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "run_shell",
                        "args": {
                            "command": _write_text_command("failure.txt", "preserved")
                        },
                        "id": "call-write-before-failure",
                        "type": "tool_call",
                    }
                ],
            ),
        )
    )
    artifact_root = tmp_path / "artifacts"
    runner = PipelineRunner(
        graph_factory=_graph_factory(model),
        artifact_presenter=_artifact_presenter_without_renders(),
        sandbox_runner=_sandbox_runner(),
        artifact_root=artifact_root,
        renderer=_renderer(),
    )

    with pytest.raises(AssertionError, match="ran out of responses"):
        runner.run_sample(manifest)

    sample_artifact_root = artifact_root / manifest.sample_id
    assert (sample_artifact_root / "workspace" / "failure.txt").read_text(
        encoding="utf-8"
    ) == "preserved"
    events = [
        json.loads(line)
        for line in (sample_artifact_root / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert events[-1]["event"] == "run_failed"


def test_run_sample_verifies_and_preserves_valid_cadquery_output(
    tmp_path: Path,
) -> None:
    manifest = _manifest_without_renders(tmp_path, "valid-box")
    model = ScriptedChatModel(
        responses=(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "run_shell",
                        "args": {
                            "command": _write_text_command(
                                "model.py",
                                VALID_BOX_SOURCE,
                            )
                        },
                        "id": "call-write-model",
                        "type": "tool_call",
                    }
                ],
            ),
            _CODING_ANSWER,
        )
    )
    artifact_root = tmp_path / "artifacts"
    console_output = StringIO()
    runner = PipelineRunner(
        graph_factory=_graph_factory(model),
        artifact_presenter=_artifact_presenter_without_renders(),
        sandbox_runner=_sandbox_runner(),
        artifact_root=artifact_root,
        renderer=_renderer(),
        console_reporter=ConsoleReporter(
            Console(
                file=console_output,
                color_system=None,
                force_terminal=False,
                highlight=False,
            )
        ),
    )

    result = runner.run_sample(manifest)

    assert result is not None
    report = _final_verification(result)
    assert report is not None
    assert report.status == "VERIFIED"
    # 000 is the build the coder's write triggered without asking for it; 001
    # is the workflow's own final verification of the same source.
    assert report.verification_id == "001"
    rendered_console = console_output.getvalue()
    assert "[node] model started — waiting" in rendered_console
    assert "tool call: run_shell" in rendered_console
    assert "ticket_initial" in rendered_console
    assert "[verification]" in rendered_console
    assert "run completed" in rendered_console

    attempts = artifact_root / "valid-box" / "workspace" / "attempts"
    assert (attempts / "000" / "model.py").read_text(
        encoding="utf-8"
    ) == VALID_BOX_SOURCE
    final_attempt = attempts / "001"
    assert (final_attempt / "model.py").read_text(encoding="utf-8") == VALID_BOX_SOURCE
    CadQueryExecutor.verify_step(final_attempt / "output.step")

    events = [
        json.loads(line)
        for line in (artifact_root / "valid-box" / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    verification = next(event for event in events if event["event"] == "verification")
    assert verification["data"]["report"]["status"] == "VERIFIED"
    assert verification["data"]["report"]["verification_id"] == "001"
    assert verification["data"]["report"]["source"] is None


def test_run_sample_repairs_model_after_intermediate_verification_failure(
    tmp_path: Path,
) -> None:
    manifest = _manifest_without_renders(tmp_path, "repair-box")
    model = ScriptedChatModel(
        responses=(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "run_shell",
                        "args": {
                            "command": _write_text_command("model.py", "result = (")
                        },
                        "id": "call-write-invalid-model",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "run_shell",
                        "args": {
                            "command": _write_text_command(
                                "model.py",
                                VALID_BOX_SOURCE,
                            )
                        },
                        "id": "call-repair-model",
                        "type": "tool_call",
                    }
                ],
            ),
            _CODING_ANSWER,
        )
    )
    artifact_root = tmp_path / "artifacts"
    runner = PipelineRunner(
        graph_factory=_graph_factory(model),
        artifact_presenter=_artifact_presenter_without_renders(),
        sandbox_runner=_sandbox_runner(),
        artifact_root=artifact_root,
        renderer=_renderer(),
    )

    result = runner.run_sample(manifest)

    assert result is not None
    # The coder never asked to be told: the broken write is reported to it at
    # the start of the turn that repairs it, which is a turn it still has.
    reports = [
        json.loads(block["text"])
        for messages in model.received_messages
        for message in messages
        if isinstance(message.content, list)
        for block in message.content
        if isinstance(block, dict) and str(block.get("text", "")).startswith("{")
    ]
    assert reports[0]["status"] == "REJECTED"
    assert reports[0]["verification_id"] == "000"

    final_report = _final_verification(result)
    assert final_report is not None
    assert final_report.status == "VERIFIED"
    assert final_report.verification_id == "002"

    attempts = artifact_root / "repair-box" / "workspace" / "attempts"
    assert (attempts / "000" / "model.py").read_text(encoding="utf-8") == "result = ("
    assert not (attempts / "000" / "output.step").exists()
    assert (attempts / "001" / "model.py").read_text(
        encoding="utf-8"
    ) == VALID_BOX_SOURCE
    CadQueryExecutor.verify_step(attempts / "002" / "output.step")


def _runner_for_rerun(
    artifact_root: Path,
    on_existing: str = "fail",
    resume_from: Path | None = None,
) -> PipelineRunner:
    return PipelineRunner(
        # This coder answers without writing model.py, which the graph would
        # otherwise send back to it. These tests are about rerun policy.
        graph_factory=_graph_factory(
            ScriptedChatModel(responses=(_CODING_ANSWER,)),
            max_stage_validation_retries=0,
        ),
        artifact_presenter=_artifact_presenter_without_renders(),
        sandbox_runner=_sandbox_runner(),
        artifact_root=artifact_root,
        renderer=_renderer(),
        on_existing=on_existing,  # type: ignore[arg-type]
        resume_from=resume_from,
    )


def test_a_completed_sample_is_refused_and_left_untouched(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    manifest = _manifest_without_renders(tmp_path, "rerun-fail")
    assert _runner_for_rerun(artifact_root).run_sample(manifest) is not None

    events_path = artifact_root / manifest.sample_id / "events.jsonl"
    before = events_path.read_text(encoding="utf-8")

    with pytest.raises(FileExistsError, match="already ran"):
        _runner_for_rerun(artifact_root).run_sample(manifest)

    assert events_path.read_text(encoding="utf-8") == before


@pytest.mark.parametrize("on_existing", ["skip", "retry"])
def test_a_completed_sample_is_passed_over(tmp_path: Path, on_existing: str) -> None:
    """`retry` redoes interrupted samples only; a finished one is never redone."""
    artifact_root = tmp_path / "artifacts"
    manifest = _manifest_without_renders(tmp_path, f"rerun-{on_existing}")
    _runner_for_rerun(artifact_root).run_sample(manifest)

    events_path = artifact_root / manifest.sample_id / "events.jsonl"
    before = events_path.read_text(encoding="utf-8")

    assert _runner_for_rerun(artifact_root, on_existing).run_sample(manifest) is None
    assert events_path.read_text(encoding="utf-8") == before


@pytest.mark.parametrize("on_existing", ["fail", "skip"])
def test_an_interrupted_sample_is_never_skipped(
    tmp_path: Path, on_existing: str
) -> None:
    """`skip` resumes a sweep, so it must not report a hole in it as done."""
    artifact_root = tmp_path / "artifacts"
    manifest = _manifest_without_renders(tmp_path, "interrupted")
    sample_root = artifact_root / manifest.sample_id
    sample_root.mkdir(parents=True)
    (sample_root / "events.jsonl").write_text(
        json.dumps({"event": "run_started", "data": {}}) + "\n", encoding="utf-8"
    )

    with pytest.raises(FileExistsError, match="incomplete run"):
        _runner_for_rerun(artifact_root, on_existing).run_sample(manifest)


def test_a_failed_sample_is_not_treated_as_completed(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    manifest = _manifest_without_renders(tmp_path, "failed-then-skip")
    runner = PipelineRunner(
        graph_factory=_graph_factory(ScriptedChatModel(responses=())),
        artifact_presenter=_artifact_presenter_without_renders(),
        sandbox_runner=_sandbox_runner(),
        artifact_root=artifact_root,
        renderer=_renderer(),
        on_existing="skip",
    )
    with pytest.raises(AssertionError, match="ran out of responses"):
        runner.run_sample(manifest)

    with pytest.raises(FileExistsError, match="incomplete run"):
        _runner_for_rerun(artifact_root, "skip").run_sample(manifest)


def test_a_directory_without_events_does_not_block_a_run(tmp_path: Path) -> None:
    """Hydra writes its resolved config there before the job body runs."""
    artifact_root = tmp_path / "artifacts"
    manifest = _manifest_without_renders(tmp_path, "hydra-first")
    hydra_dir = artifact_root / manifest.sample_id / ".hydra"
    hydra_dir.mkdir(parents=True)
    (hydra_dir / "config.yaml").write_text("artifact_root: x\n", encoding="utf-8")

    assert _runner_for_rerun(artifact_root).run_sample(manifest) is not None
    assert (hydra_dir / "config.yaml").is_file()


def test_on_existing_rejects_an_unknown_policy(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="on_existing"):
        _runner_for_rerun(tmp_path / "artifacts", "overwrite")


def test_the_runner_hands_a_graph_only_the_run_environment(tmp_path: Path) -> None:
    """The kwargs below are the contract an alternate graph has to accept.

    A graph's own settings are bound into the factory beforehand, so adding one
    must never widen this call.
    """
    captured: dict[str, Any] = {}

    def recording_factory(**kwargs: Any) -> Any:
        captured.update(kwargs)
        # A cast is a graph's own setting, so a real factory arrives with one
        # already bound; only what the runner adds is under test here.
        return create_reconstruction_graph(
            semantics_agent_builder=_semantic_stage(),
            operations_agent_builder=_operations_stage(),
            coding_agent_builder=_agent(
                "coder", ScriptedChatModel(responses=(_CODING_ANSWER,))
            ),
            audit_agent_builder=_agent(
                "output_auditor", ScriptedChatModel(responses=(_ACCEPTED_AUDIT,))
            ),
            max_stage_validation_retries=0,
            **kwargs,
        )

    artifact_root = tmp_path / "artifacts"
    manifest = _manifest_without_renders(tmp_path, "injected-graph")
    PipelineRunner(
        artifact_presenter=_artifact_presenter_without_renders(),
        sandbox_runner=_sandbox_runner(),
        artifact_root=artifact_root,
        renderer=_renderer(),
        graph_factory=recording_factory,
    ).run_sample(manifest)

    assert set(captured) == {
        "sandbox_runner",
        "sandbox_workdir",
        "renderer",
        "artifact_presenter",
        "input_manifest",
        "output_filename",
        "verification_dirname",
        "checkpointer",
    }


def test_a_graphs_own_settings_reach_it_through_the_factory(tmp_path: Path) -> None:
    """Hydra binds them with `_partial_`, so the runner never sees them."""
    manifest = _manifest_without_renders(tmp_path, "bound-budget")
    responses = tuple(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "run_shell",
                    "args": {"command": "true"},
                    "id": f"call-{turn}",
                    "type": "tool_call",
                }
            ],
        )
        for turn in range(5)
    )
    coder = ScriptedChatModel(responses=responses)
    runner = PipelineRunner(
        artifact_presenter=_artifact_presenter_without_renders(),
        sandbox_runner=_sandbox_runner(),
        artifact_root=tmp_path / "artifacts",
        renderer=_renderer(),
        graph_factory=_graph_factory(
            coder,
            max_turns=2,
            max_stage_validation_retries=0,
        ),
    )

    result = runner.run_sample(manifest)

    assert result is not None
    # The bound budget is what stopped this agent, whatever the rest of the
    # run went on to spend.
    assert len(coder.received_messages) == 2
    assert result["coding_state"]["total_turns"] == 2
    assert result["coding_state"]["stop_reason"] is StopReason.BUDGET_EXHAUSTED


def test_retry_redoes_an_interrupted_sample(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    manifest = _manifest_without_renders(tmp_path, "retry-redo")
    with pytest.raises(AssertionError, match="ran out of responses"):
        PipelineRunner(
            graph_factory=_graph_factory(ScriptedChatModel(responses=())),
            artifact_presenter=_artifact_presenter_without_renders(),
            sandbox_runner=_sandbox_runner(),
            artifact_root=artifact_root,
            renderer=_renderer(),
        ).run_sample(manifest)

    sample_root = artifact_root / manifest.sample_id
    (sample_root / "workspace" / "leftover.txt").write_text("stale", encoding="utf-8")

    result = _runner_for_rerun(artifact_root, "retry").run_sample(manifest)

    assert result is not None
    assert has_run_completed(sample_root / "events.jsonl")
    assert not (sample_root / "workspace" / "leftover.txt").exists()


def test_retry_keeps_the_job_output_hydra_already_wrote(tmp_path: Path) -> None:
    """Hydra writes those before the job body runs, so they describe this run.

    Deleting the directory outright would take them with it, and the open log
    handle would keep writing to an unlinked file.
    """
    artifact_root = tmp_path / "artifacts"
    manifest = _manifest_without_renders(tmp_path, "retry-hydra")
    sample_root = artifact_root / manifest.sample_id
    (sample_root / ".hydra").mkdir(parents=True)
    (sample_root / ".hydra" / "config.yaml").write_text("a: 1\n", encoding="utf-8")
    (sample_root / "run_pipeline.log").write_text("earlier\n", encoding="utf-8")
    (sample_root / "events.jsonl").write_text(
        json.dumps({"event": "run_started", "data": {}}) + "\n", encoding="utf-8"
    )
    (sample_root / "score.json").write_text("{}", encoding="utf-8")

    _runner_for_rerun(artifact_root, "retry").run_sample(manifest)

    assert (sample_root / ".hydra" / "config.yaml").read_text(
        encoding="utf-8"
    ) == "a: 1\n"
    assert (sample_root / "run_pipeline.log").read_text(encoding="utf-8") == "earlier\n"
    assert not (sample_root / "score.json").exists()
    assert has_run_completed(sample_root / "events.jsonl")


def _events(sample_artifact_root: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (sample_artifact_root / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]


def _logged_stop_reasons(sample_artifact_root: Path) -> Mapping[str, str]:
    """What an offline reader recovers from the log, as `aggregate_run` does."""
    return read_events(
        sample_artifact_root / "events.jsonl", sample_artifact_root.name
    ).stop_reasons


def test_what_the_agent_was_told_about_its_budget_reaches_the_event_log(
    tmp_path: Path,
) -> None:
    """Without it a run cannot afterwards show whether it was announced at all,
    which is the one thing an A/B over this flag rests on."""

    artifact_root = tmp_path / "announced"
    model = ScriptedChatModel(
        responses=tuple(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "run_shell",
                        "args": {"command": "true"},
                        "id": f"call-{n}",
                        "type": "tool_call",
                    }
                ],
            )
            for n in range(2)
        )
    )
    PipelineRunner(
        artifact_presenter=_artifact_presenter_without_renders(),
        sandbox_runner=_sandbox_runner(),
        artifact_root=artifact_root,
        renderer=_renderer(),
        graph_factory=_graph_factory(
            model,
            max_turns=2,
            announce_turns=True,
            max_stage_validation_retries=0,
        ),
    ).run_sample(_manifest_without_renders(tmp_path, "announced"))

    assert [
        messages[-1].text.split("]", 1)[0] for messages in model.received_messages
    ] == ["[turn 1/2", "[turn 2/2"]
    # Keyed by id: a notice is recorded where the agent produced it and again
    # in the workflow transcript that adopts the stage, and this is about what
    # was announced, not how many times the log mentions it.
    notices = {
        message["id"]: str(message["content"])
        for event in _events(artifact_root / "announced")
        if event["event"] == "message"
        for message in event["data"]["messages"]
        if message["type"] == "human" and str(message["content"]).startswith("[turn ")
    }
    assert [notice.split("]", 1)[0] for notice in notices.values()] == [
        "[turn 1/2",
        "[turn 2/2",
    ]


def test_the_prompt_each_role_was_given_reaches_the_event_log(
    tmp_path: Path,
) -> None:
    """A system prompt never enters agent state and an entry instruction is
    handed to `invoke` rather than produced by a node, so without this the log
    holds every answer and none of the questions. The stage agents are
    subgraphs, so this is also what says their reports reach the run's log."""

    artifact_root = tmp_path / "prompted"
    PipelineRunner(
        artifact_presenter=_artifact_presenter_without_renders(),
        sandbox_runner=_sandbox_runner(),
        artifact_root=artifact_root,
        renderer=_renderer(),
        graph_factory=_graph_factory(
            ScriptedChatModel(responses=(_CODING_ANSWER,)),
            announce_turns=False,
            max_stage_validation_retries=0,
        ),
    ).run_sample(_manifest_without_renders(tmp_path, "prompted"))

    events = _events(artifact_root / "prompted")
    prompts = [event["data"] for event in events if event["event"] == "prompt"]

    # One per ask, not one per model call: the report is what an agent was
    # asked when it was asked, and a retry re-asks nothing new.
    assert [prompt["role"] for prompt in prompts] == [
        "semantic_hypothesizer",
        "operation_planner",
        "coder",
        "output_auditor",
    ]

    coder = next(prompt for prompt in prompts if prompt["role"] == "coder")
    assert "expert CAD engineer" in coder["system"]
    instruction = "\n".join(
        str(block.get("text", ""))
        for message in coder["messages"]
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if isinstance(block, Mapping) and block.get("type") == "text"
    )
    assert "Implement the complete CadQuery program" in instruction
    # The run's paths reach the stage that needs them through its instruction,
    # not through a role that every stage of a shared thread would read.
    assert "/work/model.py" in instruction
    assert "/work/model.py" not in coder["system"]
    assert "/work/inputs/techdraw.dxf" in instruction


def test_why_the_run_stopped_reaches_the_event_log(tmp_path: Path) -> None:
    """It lives only in graph state, so an offline reader needs it projected.

    Whether `max_turns` is set sensibly is answered by counting how often
    a multi-sample run ends this way.
    """
    tool_call = [
        {
            "name": "run_shell",
            "args": {"command": "true"},
            "id": "call-0",
            "type": "tool_call",
        }
    ]
    cases = {
        "stopped-by-budget": (
            tuple(
                AIMessage(content="", tool_calls=[{**tool_call[0], "id": f"call-{n}"}])
                for n in range(4)
            ),
            "BUDGET_EXHAUSTED",
        ),
        "stopped-by-agent": ((_CODING_ANSWER,), "COMPLETED"),
    }

    for sample_id, (responses, expected) in cases.items():
        artifact_root = tmp_path / sample_id
        PipelineRunner(
            artifact_presenter=_artifact_presenter_without_renders(),
            sandbox_runner=_sandbox_runner(),
            artifact_root=artifact_root,
            renderer=_renderer(),
            graph_factory=_graph_factory(
                ScriptedChatModel(responses=responses),
                max_turns=2,
                max_stage_validation_retries=0,
            ),
        ).run_sample(_manifest_without_renders(tmp_path, sample_id))

        events = [
            event
            for event in _events(artifact_root / sample_id)
            if event["event"] == "stop_reason"
        ]
        expected_reasons = {
            "semantic_hypothesizer": "COMPLETED",
            "operation_planner": "COMPLETED",
            "coder": expected,
        }
        if expected == "COMPLETED":
            expected_reasons["output_auditor"] = "COMPLETED"
        assert {
            event["data"]["role"]: event["data"]["reason"] for event in events
        } == expected_reasons, sample_id
        assert _logged_stop_reasons(artifact_root / sample_id) == expected_reasons, (
            sample_id
        )
