import json
import shlex
import sys
from collections.abc import Mapping
from functools import partial
from inspect import cleandoc
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import ezdxf
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from PIL import Image
from rich.console import Console

from tests.zeroshot.chat_models import ScriptedChatModel
from tests.zeroshot.contracts import drawing, interpretation
from zeroshot.evaluation.aggregate_run import read_events
from zeroshot.pipeline.event_logging import ConsoleReporter, has_run_completed
from zeroshot.pipeline.messages.artifact import ArtifactPresenter
from zeroshot.pipeline.messages.manifest import InputManifest, register_view
from zeroshot.pipeline.runner import (
    GraphFactory,
    PipelineRunner,
    _latest_program_source,
)
from zeroshot.pipeline.sandbox import SandboxRunner
from zeroshot.pipeline.stages.coding.verify import VerifyOutputResult
from zeroshot.pipeline.stages.interpretation.contracts import (
    DrawingView,
    Region,
    View,
)
from zeroshot.pipeline.stages.operations.contracts import (
    Operation,
    OperationPlan,
    OperationVerb,
)
from zeroshot.pipeline.stages.tickets.contracts import (
    StageReport,
    TicketAnswers,
)
from zeroshot.pipeline.verification import CadQueryExecutor, ExecutionStatus
from zeroshot.pipeline.workflow import (
    StopReason,
    create_agent,
)
from zeroshot.pipeline.workflow.graph import AgentBuilder, create_reconstruction_graph
from zeroshot.pipeline.workflow.lifecycle import (
    advance_reconstruction,
    save_reconstruction,
    start_reconstruction,
)


def _agent(
    role: str, model: BaseChatModel, *, max_turns: int = 30, **overrides: Any
) -> AgentBuilder:
    return partial(
        create_agent, role=role, model=model, max_turns=max_turns, **overrides
    )


_ACCEPTED_AUDIT = AIMessage(
    content=(
        '{"accepted": true, "ticket_reviews": {"ticket_initial": '
        '{"summary": "The reconstruction answers the order.", "solved": true}'
        '}, "findings": [], "concern_reviews": {}}'
    )
)


def _ticket_response(stage: str, summary: str) -> dict[str, str]:
    del stage  # the pipeline stamps it; a submission is keyed by ticket
    return {"ticket_initial": summary}


_A_READING = AIMessage(
    content=TicketAnswers(
        responses=_ticket_response("interpretation", "Established sem_feature_1."),
    ).model_dump_json()
)


def _writing_interpretation() -> AIMessage:
    artifact = interpretation("a box").model_dump(mode="json")
    script = cleandoc(
        f"""
        import json
        import shutil
        from pathlib import Path
        from PIL import Image
        import ezdxf
        from ezdxf import bbox

        artifact = {artifact!r}
        inputs = json.loads(Path('/work/reconstruction.json').read_text())['input_drawings']
        views = []
        for given in inputs:
            file = given['file']
            name = given['name']
            if Path(file).suffix == '.dxf':
                extent = bbox.extents(ezdxf.readfile(file).modelspace()).size
                region = {{'view': name, 'box_uv': [0, 0, extent.x, extent.y]}}
            else:
                with Image.open(file) as image:
                    width, height = image.size
                region = {{'view': name, 'box_px': [0, 0, width, height]}}
            views.append(dict(name=name, file=file, role=given['role'], region=region, dimensions=[]))
        front = dict(views[0], name='view_front', role='front')
        front['file'] = '/work/view_front' + Path(front['file']).suffix
        shutil.copyfile(views[0]['file'], front['file'])
        views.append(front)
        artifact['views'] = views
        artifact['features'][0]['evidence'] = [dict(front['region'], view='view_front')]
        Path('/work/interpretation.json').write_text(json.dumps(artifact))
        """
    )
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "run_shell",
                "args": {"command": f"python -c {shlex.quote(script)}"},
                "id": "call-write-interpretation",
                "type": "tool_call",
            }
        ],
    )


def _interpretation_stage():
    return _agent(
        "drawing_interpreter",
        ScriptedChatModel(responses=(_writing_interpretation(), _A_READING)),
        announce_turns=False,
    )


_A_PLAN = OperationPlan(
    proposal=[
        Operation(
            name="op_base",
            verb=OperationVerb.EXTRUDE,
            detail="extrude it",
            semantics=["sem_feature_1"],
        )
    ],
    rationale="one extrude",
)


def _operations_stage():
    write = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "run_shell",
                "args": {
                    "command": "python -c "
                    + shlex.quote(
                        "from pathlib import Path;"
                        f"Path('/work/operations.json').write_text({_A_PLAN.model_dump_json()!r})"
                    )
                },
                "id": "call-write-operations",
                "type": "tool_call",
            }
        ],
    )
    submission = TicketAnswers(
        responses=_ticket_response("operations", "Established op_base."),
    )
    return _agent(
        "operation_planner",
        ScriptedChatModel(
            responses=(write, AIMessage(content=submission.model_dump_json()))
        ),
        announce_turns=False,
    )


def _writing_model(call_id: str = "call-write-model") -> AIMessage:
    """One turn that puts a building program in the workspace.

    The gate in `VerifyOnWriteMiddleware` refuses an answer while the program
    does not build, so a coder fixture has to write one before it answers.
    """
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "run_shell",
                "args": {"command": _write_text_command("model.py", VALID_BOX_SOURCE)},
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


_CODING_ANSWER = AIMessage(
    content=TicketAnswers(
        responses=_ticket_response("coding", "Implemented ret_base and result."),
        stage_report=StageReport(dimension_checks={}),
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
        interpretation_agent_builder=_interpretation_stage(),
        dxf_mm_per_unit={"view_drawing": 1.0, "view_front": 1.0},
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
    run = start_reconstruction("run_sample", "Reconstruct the drawing.", drawing())
    run = advance_reconstruction(
        run,
        TicketAnswers(
            responses=_ticket_response("interpretation", "Established sem_feature_1."),
        ),
        workspace_output=interpretation("a box"),
    )
    run = advance_reconstruction(
        run,
        TicketAnswers(
            responses=_ticket_response("operations", "Established op_base."),
        ),
        workspace_output=_A_PLAN,
    )
    run = advance_reconstruction(
        run,
        TicketAnswers(
            responses=_ticket_response("coding", "Implemented ret_base and result."),
            stage_report=StageReport(dimension_checks={}),
        ),
        workspace_output=VerifyOutputResult(
            verification_id="007",
            status=ExecutionStatus.VERIFIED,
            source=VALID_BOX_SOURCE,
            returncode=0,
        ),
    )

    return run


@pytest.mark.parametrize("stage", ["interpretation", "operations"])
def test_resume_copies_an_external_attempt_directly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    run = _verified_resume_run()

    source_workspace = tmp_path / "source" / "workspace"
    attempt = source_workspace / "attempts" / "round_000" / "coding" / "007"
    attempt.mkdir(parents=True)
    (attempt / "output.step").write_bytes(b"STEP")
    diagnostic = Path(f"attempts/round_000/{stage}/001/_{stage}_validation_log.json")
    (source_workspace / diagnostic).parent.mkdir(parents=True)
    (source_workspace / diagnostic).write_text(
        '{"reports": {"view_front": {"status": "ok"}}}'
    )
    artifact = diagnostic.parent / f"{stage}.json"
    (source_workspace / artifact).write_text(
        getattr(run.snapshots[-1], stage).model_dump_json()
    )
    future_diagnostic = Path(
        f"attempts/round_001/{stage}/000/_{stage}_validation_log.json"
    )
    (source_workspace / future_diagnostic).parent.mkdir(parents=True)
    (source_workspace / future_diagnostic).write_text('{"reports": {}}')
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
    assert (
        workspace / "attempts" / "round_000" / "coding" / "007" / "output.step"
    ).read_bytes() == b"STEP"
    assert (attempt / "output.step").is_file()
    assert (workspace / diagnostic).read_bytes() == (
        source_workspace / diagnostic
    ).read_bytes()
    assert (workspace / artifact).read_bytes() == (
        source_workspace / artifact
    ).read_bytes()
    assert not (workspace / future_diagnostic).exists()
    assert (source_workspace / future_diagnostic).is_file()


@pytest.mark.parametrize("stage", ["interpretation", "operations"])
def test_resume_temporarily_protects_an_attempt_cleared_by_retry(
    tmp_path: Path, stage: str
) -> None:
    run = _verified_resume_run()
    artifact_root = tmp_path / "artifacts"
    sample_root = artifact_root / "sample"
    workspace = sample_root / "workspace"
    attempt = workspace / "attempts" / "round_000" / "coding" / "007"
    attempt.mkdir(parents=True)
    (attempt / "output.step").write_bytes(b"STEP")
    (workspace / "stale.txt").write_text("stale", encoding="utf-8")
    diagnostic = Path(f"attempts/round_000/{stage}/001/_{stage}_validation_log.json")
    (workspace / diagnostic).parent.mkdir(parents=True)
    diagnostic_bytes = b'{"reports": {"view_front": {"status": "ok"}}}'
    (workspace / diagnostic).write_bytes(diagnostic_bytes)
    artifact = diagnostic.parent / f"{stage}.json"
    artifact_json = getattr(run.snapshots[-1], stage).model_dump_json()
    (workspace / artifact).write_text(artifact_json)
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
    assert (
        prepared / "attempts" / "round_000" / "coding" / "007" / "output.step"
    ).read_bytes() == b"STEP"
    assert (prepared / diagnostic).read_bytes() == diagnostic_bytes
    assert (prepared / artifact).read_text() == artifact_json


@pytest.mark.parametrize("same_workspace", [False, True])
@pytest.mark.parametrize("file_state", ["present", "missing", "outside_symlink"])
def test_resume_restores_drawing_stage_crops(
    tmp_path: Path,
    same_workspace: bool,
    file_state: str,
) -> None:
    artifact_root = tmp_path / "destination"
    sample_root = artifact_root / "sample"
    source_workspace = (
        sample_root / "workspace"
        if same_workspace
        else tmp_path / "source" / "workspace"
    )
    source_workspace.mkdir(parents=True)
    page = DrawingView(
        name="view_drawing",
        role=View.FULL_PAGE,
        file="/work/inputs/view_drawing.dxf",
        region=Region(view="view_drawing", box_uv=(0, 0, 20, 10)),
        dimensions=[],
    )
    crop = DrawingView(
        name="view_front",
        role=View.FRONT,
        file="/work/derived/sheet_front.dxf",
        region=Region(view="view_drawing", box_uv=(0, 0, 20, 10)),
        dimensions=[],
    )
    relative_crop = crop.model_copy(
        update={
            "name": "view_detail",
            "role": View.DETAIL,
            "file": "views/sheet_detail.png",
        }
    )
    temporary_crop = crop.model_copy(
        update={
            "name": "view_section",
            "role": View.SECTION,
            "file": "/tmp/sheet_section.png",
        }
    )
    run = start_reconstruction(
        "run_sample",
        "Reconstruct the drawing.",
        [page],
    )
    run = advance_reconstruction(
        run,
        TicketAnswers(
            responses=_ticket_response("interpretation", "Read view_front."),
        ),
        workspace_output=interpretation(
            views=[page, crop, relative_crop, temporary_crop]
        ),
    )
    derived = source_workspace / "derived" / "sheet_front.dxf"
    derived.parent.mkdir()
    derived.write_bytes(b"DERIVED DXF")
    relative = source_workspace / "views" / "sheet_detail.png"
    relative.parent.mkdir()
    relative.write_bytes(b"DETAIL PNG")
    temporary = source_workspace / "tmp" / "sheet_section.png"
    temporary.parent.mkdir()
    temporary.write_bytes(b"SECTION PNG")
    resume_path = source_workspace / "reconstruction.json"
    save_reconstruction(resume_path, run)
    runner = _runner_for_rerun(
        artifact_root,
        "retry",
        resume_from=resume_path,
    )

    if file_state != "present":
        temporary.unlink()
        if file_state == "outside_symlink":
            outside = tmp_path / "outside.png"
            outside.write_bytes(b"outside the resumed workspace")
            temporary.symlink_to(outside)
            error_type, message = ValueError, "escapes the workspace"
        else:
            error_type, message = FileNotFoundError, "resume drawing file is missing"
        with pytest.raises(error_type, match=message):
            runner._prepare_workspace(sample_root, sample_root / "events.jsonl", run)
        # A bad resume must not clear the source before it reports the error.
        assert resume_path.is_file()
        assert derived.read_bytes() == b"DERIVED DXF"
        assert relative.read_bytes() == b"DETAIL PNG"
        return

    prepared = runner._prepare_workspace(
        sample_root,
        sample_root / "events.jsonl",
        run,
    )

    restored = prepared / "derived" / "sheet_front.dxf"
    assert restored.read_bytes() == b"DERIVED DXF"
    assert (prepared / "views" / "sheet_detail.png").read_bytes() == b"DETAIL PNG"
    assert (prepared / "tmp" / "sheet_section.png").read_bytes() == b"SECTION PNG"
    accepted = run.snapshots[-1].interpretation
    assert accepted is not None
    assert (
        next(sheet.file for sheet in accepted.views if sheet.name == "view_front")
        == "/work/derived/sheet_front.dxf"
    )
    assert derived.is_file()


def _resume_input_case(tmp_path: Path, same_workspace: bool):
    artifact_root = tmp_path / "destination"
    sample_id = "renamed-sample"
    source_workspace = (
        artifact_root / sample_id / "workspace"
        if same_workspace
        else tmp_path / "source" / "workspace"
    )
    source_input = source_workspace / "inputs" / "view_drawing.png"
    source_input.parent.mkdir(parents=True)
    Image.new("RGB", (20, 30), "white").save(source_input)
    original = register_view("view_drawing", View.FULL_PAGE, source_input)
    original.file = "/work/inputs/view_drawing.png"
    run = _verified_resume_run()
    run.run_id = "run_original_sample"
    run.input_drawings = [original]
    run.snapshots[-1].interpretation.views = [original.model_copy(deep=True)]
    run.snapshots[-1].interpretation.features[0].evidence = [
        Region(view="view_drawing", box_px=(0, 0, 10, 10))
    ]
    attempt = source_workspace / "attempts" / "round_000" / "coding" / "007"
    attempt.mkdir(parents=True)
    (attempt / "output.step").write_bytes(b"fixture STEP")
    resume_path = source_workspace / "reconstruction.json"
    save_reconstruction(resume_path, run)
    (source_workspace / "stale.txt").write_text("keep until inputs match")
    supplied = tmp_path / "same-drawing-at-another-path.png"
    supplied.write_bytes(source_input.read_bytes())
    manifest = InputManifest(
        sample_id=sample_id,
        drawing=[register_view("view_drawing", View.FULL_PAGE, supplied)],
    )
    runner = _runner_for_rerun(
        artifact_root, "retry" if same_workspace else "fail", resume_from=resume_path
    )
    # This checkpoint has finished its allowed coding rounds; no model runs.
    runner.graph_factory.keywords["max_audit_reject_count"] = 0
    return runner, manifest, run, source_input


@pytest.mark.parametrize("same_workspace", [False, True])
def test_public_resume_accepts_the_same_input_at_another_path_and_sample_name(
    tmp_path: Path, same_workspace: bool
):
    runner, manifest, original, _ = _resume_input_case(tmp_path, same_workspace)

    result = runner.run_sample(manifest)

    assert result["reconstruction"] == original
    destination = runner.artifact_root / manifest.sample_id
    assert has_run_completed(destination / "events.jsonl")
    restored = destination / "workspace" / "inputs" / "view_drawing.png"
    assert restored.read_bytes() == Path(manifest.drawing[0].file).read_bytes()
    assert (destination / "workspace" / "model.py").read_text() == VALID_BOX_SOURCE


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize(
    "source_location",
    ["inside", "inside_symlink", "outside_symlink", "outside_alias_symlink", "outside"],
)
def test_retry_preserves_current_inputs_before_clearing_the_sample(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resume: bool,
    source_location: str,
) -> None:
    runner, manifest, _, saved_input = _resume_input_case(
        tmp_path, same_workspace=source_location != "outside"
    )
    runner.on_existing = "retry"
    if not resume:
        runner.resume_from = None
    sample_root = runner.artifact_root / manifest.sample_id
    sample_root.mkdir(parents=True, exist_ok=True)
    leftover = sample_root / "discard-me.txt"
    leftover.write_text("stale")
    external_input = Path(manifest.drawing[0].file)
    expected = external_input.read_bytes()
    if source_location == "inside":
        supplied = saved_input
    elif source_location == "inside_symlink":
        supplied = sample_root / "current.png"
        supplied.symlink_to(external_input)
    elif source_location == "outside_symlink":
        supplied = tmp_path / "current.png"
        supplied.symlink_to(saved_input)
    elif source_location == "outside_alias_symlink":
        (sample_root / "current.png").symlink_to(external_input)
        alias = tmp_path / "sample-alias"
        alias.symlink_to(sample_root, target_is_directory=True)
        supplied = alias / "current.png"
    else:
        supplied = external_input

        def reject_temporary_directory(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("external inputs need no temporary copy")

        monkeypatch.setattr(
            "zeroshot.pipeline.runner.tempfile.TemporaryDirectory",
            reject_temporary_directory,
        )
    manifest.drawing[0].file = str(supplied)
    original_view = manifest.drawing[0].model_copy(deep=True)

    def inspect_inputs(**kwargs: Any) -> Any:
        staged = kwargs["input_manifest"].drawing[0]
        assert staged.model_dump(exclude={"file"}) == original_view.model_dump(
            exclude={"file"}
        )
        assert Path(staged.file) == (
            sample_root / "workspace" / "inputs" / "view_drawing.png"
        )
        assert Path(staged.file).read_bytes() == expected
        assert not Path(staged.file).is_symlink()
        # Stop at the graph boundary: no model or sandbox command is needed.
        return SimpleNamespace(
            stream_events=lambda *args, **kwargs: SimpleNamespace(output={})
        )

    runner.graph_factory = inspect_inputs

    assert runner.run_sample(manifest) == {}
    assert not leftover.exists()
    assert external_input.read_bytes() == expected
    assert manifest.drawing[0] == original_view


@pytest.mark.parametrize("same_workspace", [False, True])
@pytest.mark.parametrize(
    "difference",
    ["contents", "role", "name", "bounds", "suffix", "missing", "legacy_path"],
)
def test_public_resume_rejects_different_or_missing_inputs_before_clearing_anything(
    tmp_path: Path, same_workspace: bool, difference: str
):
    runner, manifest, original, saved_input = _resume_input_case(
        tmp_path, same_workspace
    )
    supplied = Path(manifest.drawing[0].file)
    if difference == "contents":
        # Same dimensions and metadata, different source content.
        Image.new("RGB", (20, 30), "black").save(supplied)
    elif difference == "role":
        manifest.drawing[0].role = View.FRONT
    elif difference == "name":
        manifest.drawing[0].name = "view_other"
    elif difference == "bounds":
        Image.new("RGB", (40, 50), "white").save(supplied)
        manifest.drawing[0] = register_view("view_drawing", View.FULL_PAGE, supplied)
    elif difference == "suffix":
        renamed = supplied.with_suffix(".jpg")
        renamed.write_bytes(supplied.read_bytes())
        manifest.drawing[0].file = str(renamed)
    elif difference == "missing":
        saved_input.unlink()
    else:
        legacy = saved_input.with_name("drawing.png")
        saved_input.rename(legacy)
        saved_input = legacy
        original.input_drawings[0].file = "/work/inputs/drawing.png"
        original.snapshots[-1].interpretation.views[0].file = "/work/inputs/drawing.png"
        save_reconstruction(runner.resume_from, original)
    source_workspace = runner.resume_from.parent
    history_before = runner.resume_from.read_bytes()
    source_before = saved_input.read_bytes() if saved_input.exists() else None

    error_type = FileNotFoundError if difference == "missing" else ValueError
    with pytest.raises(error_type, match="resume input"):
        runner.run_sample(manifest)

    assert runner.resume_from.read_bytes() == history_before
    assert (source_workspace / "stale.txt").read_text() == "keep until inputs match"
    if source_before is not None:
        assert saved_input.read_bytes() == source_before
    assert not (runner.artifact_root / manifest.sample_id / "events.jsonl").exists()


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


def _write_fixture_dxf(path: Path) -> str:
    doc = ezdxf.new()
    doc.modelspace().add_lwpolyline([(0, 0), (10, 0), (10, 10), (0, 10)], close=True)
    doc.saveas(path)
    return path.read_text()


def _manifest_without_renders(tmp_path: Path, sample_id: str) -> InputManifest:
    dxf_path = tmp_path / f"{sample_id}.dxf"
    _write_fixture_dxf(dxf_path)
    return InputManifest(
        sample_id=sample_id,
        drawing=[register_view("view_drawing", View.FULL_PAGE, dxf_path, 1.0)],
    )


def _artifact_presenter_without_renders() -> ArtifactPresenter:
    return ArtifactPresenter(input_mode="path", feedback_mode="none")


def _sandbox_runner() -> SandboxRunner:
    return SandboxRunner(
        python_executable=Path(sys.executable),
        default_timeout_s=30,
    )


def test_run_sample_stages_only_allowed_inputs_and_preserves_workdir(
    tmp_path: Path,
) -> None:
    dxf_path = tmp_path / "source.dxf"
    selected_render_path = tmp_path / "selected.png"
    hidden_render_path = tmp_path / "hidden.png"
    original_dxf = _write_fixture_dxf(dxf_path)
    Image.new("RGB", (10, 10), "green").save(selected_render_path)
    selected_bytes = selected_render_path.read_bytes()
    hidden_render_path.write_bytes(b"HIDDEN_RENDER")

    manifest = InputManifest(
        sample_id="sample-1",
        drawing=[
            register_view("view_drawing", View.FULL_PAGE, dxf_path, 1.0),
            register_view("view_style_a", View.PERSPECTIVE, selected_render_path),
        ],
    )
    inspect_inputs = cleandoc(
        """
        from pathlib import Path

        dxf = Path('/work/inputs/view_drawing.dxf')
        assert 'ENTITIES' in dxf.read_text()
        assert Path('/work/inputs/view_style_a.png').read_bytes().startswith(bytes.fromhex('89504e47'))
        assert not Path('/work/inputs/view_hidden.png').exists()
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
            _writing_model(),
            _CODING_ANSWER,
        )
    )
    artifact_presenter = ArtifactPresenter(input_mode="path", feedback_mode="none")
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
    )

    result = runner.run_sample(manifest)

    assert result is not None
    assert model.bound_tool_names == (
        "run_shell",
        "load_image",
    )
    # Two inspections, the write, and the answer.
    assert len(model.received_messages) == 4

    # The coder opens on the workflow transcript: its own prompt, the run's
    # input, then the current interpretation and plan.
    initial_messages = model.received_messages[0]
    initial_human_message = initial_messages[1]
    assert isinstance(initial_human_message, HumanMessage)
    initial_text = _message_text(initial_human_message)
    assert "/work/inputs/view_drawing.dxf" in initial_text
    assert "/work/inputs/view_style_a.png" in initial_text
    # A sheet the input config does not declare never reaches the run at all.
    # By its name rather than the bare word, which the format advice uses of a
    # hidden edge.
    assert "view_hidden" not in initial_text
    assert "hidden.png" not in initial_text
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
        AIMessage,
        ToolMessage,
        HumanMessage,
    ]

    inspect_result = messages[-6]
    assert isinstance(inspect_result, ToolMessage)
    assert inspect_result.tool_call_id == "call-inspect-inputs"
    assert isinstance(inspect_result.content, str)
    assert json.loads(inspect_result.content) == {
        "status": "COMPLETED",
        "returncode": 0,
        "stdout": "staged-ok\n",
        "stderr": "",
    }

    persistence_result = messages[-4]
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

    assert dxf_path.read_text(encoding="utf-8") == original_dxf
    assert selected_render_path.read_bytes() == selected_bytes
    assert hidden_render_path.read_bytes() == b"HIDDEN_RENDER"

    sample_artifact_root = tmp_path / "artifacts" / "sample-1"
    saved_workdir = sample_artifact_root / "workspace"
    assert (saved_workdir / "inputs" / "view_drawing.dxf").read_text(
        encoding="utf-8"
    ) == original_dxf
    assert (
        saved_workdir / "inputs" / "view_style_a.png"
    ).read_bytes() == selected_bytes
    assert not (saved_workdir / "inputs" / "view_hidden.png").exists()
    assert (saved_workdir / "scratch.txt").read_text(encoding="utf-8") == "persisted"
    # The coding stage renders even when feedback_mode="none"; the auditor
    # still needs these artifacts, without a renderer supplied by the runner.
    snapshot = result["reconstruction"].snapshots[-1]
    assert snapshot.verification is not None
    coding_attempt = (
        saved_workdir
        / "attempts"
        / f"round_{snapshot.round:03d}"
        / "coding"
        / str(snapshot.verification.verification_id)
    )
    assert (coding_attempt / "output.step").is_file()
    assert (coding_attempt / "projection" / "front.dxf").is_file()
    assert (coding_attempt / "projection" / "front.png").is_file()
    assert (coding_attempt / "render_3d" / "hlg_perspective.png").is_file()

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
        "report": {
            "accepted": True,
            "ticket_reviews": {
                "ticket_initial": {
                    "summary": "The reconstruction answers the order.",
                    "solved": True,
                }
            },
            "findings": [],
            "concern_reviews": {},
        },
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
    # One build: the coder's write triggered it without being asked, and the
    # workflow's own final verification found the same source already built.
    assert report.verification_id == "000"
    rendered_console = console_output.getvalue()
    assert "[node] model started — waiting" in rendered_console
    assert "tool call: run_shell" in rendered_console
    assert "ticket_initial" in rendered_console
    assert "[verification]" in rendered_console
    assert "run completed" in rendered_console

    attempts = artifact_root / "valid-box" / "workspace" / "attempts"
    assert [path.name for path in attempts.iterdir()] == ["round_000"]
    round_attempts = attempts / "round_000"
    assert sorted(path.name for path in round_attempts.iterdir()) == [
        "coding",
        "interpretation",
        "operations",
    ]
    final_attempt = round_attempts / "coding" / "000"
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
    assert verification["data"]["report"]["verification_id"] == "000"
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
    assert final_report.verification_id == "001"

    # One attempt per program: the broken one, then the repair the workflow's
    # own verification found already built.
    attempts = (
        artifact_root / "repair-box" / "workspace" / "attempts" / "round_000" / "coding"
    )
    assert sorted(path.name for path in attempts.iterdir()) == ["000", "001"]
    assert (attempts / "000" / "model.py").read_text(encoding="utf-8") == "result = ("
    assert not (attempts / "000" / "output.step").exists()
    assert (attempts / "001" / "model.py").read_text(
        encoding="utf-8"
    ) == VALID_BOX_SOURCE
    CadQueryExecutor.verify_step(attempts / "001" / "output.step")


def _runner_for_rerun(
    artifact_root: Path,
    on_existing: str = "fail",
    resume_from: Path | None = None,
) -> PipelineRunner:
    return PipelineRunner(
        # This coder answers without writing model.py, which the graph would
        # otherwise send back to it. These tests are about rerun policy.
        graph_factory=_graph_factory(
            ScriptedChatModel(responses=(_writing_model(), _CODING_ANSWER)),
            max_stage_validation_retries=0,
        ),
        artifact_presenter=_artifact_presenter_without_renders(),
        sandbox_runner=_sandbox_runner(),
        artifact_root=artifact_root,
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
            interpretation_agent_builder=_interpretation_stage(),
            dxf_mm_per_unit={"view_drawing": 1.0, "view_front": 1.0},
            operations_agent_builder=_operations_stage(),
            coding_agent_builder=_agent(
                "coder",
                ScriptedChatModel(responses=(_writing_model(), _CODING_ANSWER)),
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
        graph_factory=recording_factory,
    ).run_sample(manifest)

    assert set(captured) == {
        "sandbox_runner",
        "sandbox_workdir",
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
        graph_factory=_graph_factory(
            ScriptedChatModel(responses=(_writing_model(), _CODING_ANSWER)),
            announce_turns=False,
            max_stage_validation_retries=0,
        ),
    ).run_sample(_manifest_without_renders(tmp_path, "prompted"))

    events = _events(artifact_root / "prompted")
    prompts = [event["data"] for event in events if event["event"] == "prompt"]

    # One per ask, not one per model call: the report is what an agent was
    # asked when it was asked, and a retry re-asks nothing new.
    assert [prompt["role"] for prompt in prompts] == [
        "drawing_interpreter",
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
    assert "/work/inputs/view_drawing.dxf" in instruction


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
        "stopped-by-agent": ((_writing_model(), _CODING_ANSWER), "COMPLETED"),
    }

    for sample_id, (responses, expected) in cases.items():
        artifact_root = tmp_path / sample_id
        PipelineRunner(
            artifact_presenter=_artifact_presenter_without_renders(),
            sandbox_runner=_sandbox_runner(),
            artifact_root=artifact_root,
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
            "drawing_interpreter": "COMPLETED",
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
