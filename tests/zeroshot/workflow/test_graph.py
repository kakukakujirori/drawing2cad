"""C4 workflow wiring: submissions, integration, retries, and audit rounds."""

import base64
import json
import sys
from dataclasses import replace
from functools import partial
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.messages.content import ContentBlock
from omegaconf import OmegaConf

from tests.zeroshot.chat_models import ScriptedChatModel, tool_call
from tests.zeroshot.contracts import (
    bootstrap_review,
    drawing,
    evidence,
    interpretation,
    interpreted_feature,
)
from tests.zeroshot.prompt_paths import ROLE_PATHS
from zeroshot.pipeline.messages.artifact import ArtifactPresenter
from zeroshot.pipeline.messages.manifest import InputManifest, register_view
from zeroshot.pipeline.sandbox import SandboxRunner, SandboxWorkdir
from zeroshot.pipeline.stages.audit import verify as audit_verify_module
from zeroshot.pipeline.stages.audit.contracts import (
    AuditFinding,
    AuditReport,
    AuditSubmission,
    TicketReview,
)
from zeroshot.pipeline.stages.coding import stage as coding_stage_module
from zeroshot.pipeline.stages.coding.validate import CodingOutput
from zeroshot.pipeline.stages.coding.verify import VerifyOutputResult
from zeroshot.pipeline.stages.contracts import ReconstructionHistory
from zeroshot.pipeline.stages.interpretation.contracts import (
    DrawingInterpretation,
    DrawingView,
    ProjectionMethod,
    Region,
    View,
)
from zeroshot.pipeline.stages.tickets.contracts import (
    StageReport,
    TicketAnswers,
)
from zeroshot.pipeline.stages.types import PipelineStage
from zeroshot.pipeline.verification import ExecutionStatus
from zeroshot.pipeline.verification.run_cadquery import CadQueryExecutionReport
from zeroshot.pipeline.verification.run_drawing_diff import DrawingDiffReport
from zeroshot.pipeline.workflow import create_agent
from zeroshot.pipeline.workflow.components.compact import SUMMARY_PREAMBLE
from zeroshot.pipeline.workflow.graph import AgentBuilder, create_reconstruction_graph
from zeroshot.pipeline.workflow.lifecycle import (
    advance_reconstruction,
    start_reconstruction,
)

_ROUND_ZERO_TICKET = "ticket_initial"
_ROUND_ONE_TICKET = "ticket_001_missing_hole"
_PROGRAM = "result = object()\n"


def _message(answer: object) -> AIMessage:
    return AIMessage(content=answer.model_dump_json())  # type: ignore[attr-defined]


def _response(ticket_id: str, stage: PipelineStage) -> dict[str, str]:
    return {ticket_id: f"Addressed {ticket_id} in {stage.value}."}


def _responses(ticket_id: str | None, stage: PipelineStage) -> dict[str, str]:
    return _response(ticket_id, stage) if ticket_id is not None else {}


def _interpretation_candidate(
    artifact: DrawingInterpretation | None = None,
) -> DrawingInterpretation:
    artifact = artifact or interpretation("a plate")
    handed = DrawingView(
        name="view_input",
        role="full_page",
        file="/work/drawing.png",
        region=Region(view="view_input", box_px=(0, 0, 20, 20)),
        dimensions=[],
    )
    views = [
        item.model_copy(
            update={
                "file": "/work/front.png",
                "region": Region(view="view_input", box_px=(0, 0, 20, 20)),
            }
        )
        for item in artifact.views
    ]
    return artifact.model_copy(update={"views": [handed, *views]})


def _write_interpretation(
    artifact: DrawingInterpretation | None = None, call_id: str = "draw"
) -> AIMessage:
    payload = base64.b64encode(
        (_interpretation_candidate(artifact).model_dump_json(indent=2) + "\n").encode()
    ).decode()
    command = (
        'python -c "import base64;'
        "open('/work/interpretation.json','wb').write(base64.b64decode('"
        + payload
        + "'))\""
    )
    return tool_call("run_shell", {"command": command}, call_id)


def _coding_script(
    ticket_id: str | None = _ROUND_ZERO_TICKET,
    *,
    artifact: DrawingInterpretation | None = None,
    call_id: str = "draw",
) -> tuple[AIMessage, AIMessage]:
    """Write interpretation.json, then answer the round's tickets."""
    return _write_interpretation(artifact, call_id), _coding_submission(ticket_id)


def _coding_submission(ticket_id: str | None = _ROUND_ZERO_TICKET) -> AIMessage:
    return _message(
        TicketAnswers(
            responses=_responses(ticket_id, PipelineStage.CODING),
            stage_report=StageReport(concerns={}),
        )
    )


def _audit_script(*reports: AIMessage, call_id: str = "audit") -> tuple[AIMessage, ...]:
    """Write candidate reports, then confirm only the final valid candidate."""
    turns = []
    for index, report in enumerate(reports):
        payload = base64.b64encode(report.text.encode()).decode()
        command = (
            'python -c "import base64;'
            "open('/work/audit.json','wb').write(base64.b64decode('" + payload + "'))\""
        )
        turns.append(tool_call("run_shell", {"command": command}, f"{call_id}_{index}"))
    turns.append(
        _message(AuditSubmission(accepted=not json.loads(reports[-1].text)["findings"]))
    )
    return tuple(turns)


def _accepted_audit() -> AIMessage:
    return _message(
        AuditReport(
            concern_reviews={},
            ticket_reviews=bootstrap_review(),
            findings=[],
        )
    )


def _rejected_audit(
    cause: str = "coding", targets: list[str] | None = None
) -> AIMessage:
    return _message(
        AuditReport(
            concern_reviews={},
            ticket_reviews=bootstrap_review(),
            findings=[
                AuditFinding(
                    name="find_missing_hole",
                    observation="The drawing contains a hole that the model omits.",
                    evidence=evidence("projection/front.dxf"),
                    cause=cause,  # type: ignore[arg-type]
                    targets=targets or [],
                    revision_request="The model omits the hole.",
                    related_ticket_ids=[],
                )
            ],
        )
    )


def _interpretation_rejected_audit() -> AIMessage:
    return _message(
        AuditReport(
            concern_reviews={},
            ticket_reviews=bootstrap_review(),
            findings=[
                AuditFinding(
                    name="find_wrong_edge",
                    observation="The front edge starts at the wrong coordinate.",
                    evidence=evidence("front.png", "projection/front.dxf"),
                    cause="interpretation",
                    targets=["view_front"],
                    revision_request="The front sheet's edge reading is wrong.",
                    related_ticket_ids=[],
                )
            ],
        )
    )


def _invalid_audit() -> AIMessage:
    return _message(
        AuditReport(
            concern_reviews={},
            ticket_reviews=bootstrap_review(),
            findings=[
                AuditFinding(
                    name="find_unknown_feature",
                    observation="The model is incorrect.",
                    evidence=evidence("projection/front.dxf"),
                    cause="interpretation",
                    targets=["sem_missing"],
                    revision_request="The absent feature is wrong.",
                    related_ticket_ids=[],
                )
            ],
        )
    )


def _agent(role: str, model: BaseChatModel, **overrides: Any) -> AgentBuilder:
    return partial(create_agent, role=role, model=model, **overrides)


def _artifact_presenter() -> ArtifactPresenter:
    return ArtifactPresenter(
        input="path",
        output_renders="path",
        unmatched="path",
        overlay="path",
    )


def _write_projection_dxf(path: Path) -> None:
    """The measurable projection every audit finding must cite."""
    import ezdxf

    path.parent.mkdir(parents=True, exist_ok=True)
    document = ezdxf.new()
    document.modelspace().add_lwpolyline(
        [(0, 0), (20, 0), (20, 20), (0, 20)], close=True
    )
    document.saveas(path)


def _graph(
    workdir: SandboxWorkdir,
    *,
    coder: ScriptedChatModel,
    auditor: ScriptedChatModel,
    history_filename: str = "reconstruction.json",
    coder_options: dict[str, Any] | None = None,
    auditor_options: dict[str, Any] | None = None,
    projection_method: ProjectionMethod = ProjectionMethod.THIRD_ANGLE,
    **overrides: Any,
):
    from PIL import Image

    path = workdir.host_bind_dir / "drawing.png"
    Image.new("RGB", (20, 20), "white").save(path)
    Image.new("RGB", (20, 20), "white").save(workdir.host_bind_dir / "front.png")
    _write_projection_dxf(workdir.host_bind_dir / "projection" / "front.dxf")
    common = {
        "announce_turns": False,
        "model_call_retries": 0,
        "checkpointer": False,
        "max_turns": 5,
    }
    return create_reconstruction_graph(
        coding_agent_builder=_agent("coder", coder, **common | (coder_options or {})),
        audit_agent_builder=_agent(
            "output_auditor", auditor, **common | (auditor_options or {})
        ),
        sandbox_runner=SandboxRunner(
            python_executable=Path(sys.executable), default_timeout_s=10
        ),
        sandbox_workdir=workdir,
        artifact_presenter=_artifact_presenter(),
        input_manifest=InputManifest(
            sample_id="test",
            drawing=[register_view("view_input", View.FULL_PAGE, path)],
            projection_method=projection_method,
        ),
        reconstruction_history_filename=history_filename,
        **overrides,
    )


def _stub_verification(
    monkeypatch: pytest.MonkeyPatch,
    *reports: VerifyOutputResult,
    views_seen: list[list[View]] | None = None,
) -> list[str]:
    remaining = list(reports)
    calls: list[str] = []

    class StubVerifier:
        def __init__(self, workdir: SandboxWorkdir, source_filename: str) -> None:
            self.source_filename = source_filename
            self.source_path = workdir.host_bind_dir / source_filename
            self.interpretation = None

        accepted_source = None
        last_report = None

        @property
        def confirmed(self) -> bool:
            return True

        @property
        def blockers(self) -> list[str]:
            return []

        def reset(self) -> None:
            pass

        def source_digest(self) -> str | None:
            return (
                sha256(self.source_path.read_bytes()).hexdigest()
                if self.source_path.is_file()
                else None
            )

        def verify(self) -> VerifyOutputResult:
            calls.append("verify")
            if views_seen is not None:
                views_seen.append(list(self.interpretation.view_frames()))
            return remaining.pop(0)

        def feedback(self) -> list[ContentBlock]:
            return []

    monkeypatch.setattr(
        coding_stage_module,
        "ProgressOutputVerifier",
        lambda **kwargs: StubVerifier(
            workdir=kwargs["workdir"],
            source_filename=kwargs["source_filename"],
        ),
    )
    return calls


def _verified(identifier: str = "000", source: str = _PROGRAM) -> VerifyOutputResult:
    return VerifyOutputResult(
        verification_id=identifier,
        exec_report=CadQueryExecutionReport(
            status=ExecutionStatus.VERIFIED, source=source, returncode=0
        ),
    )


def _last_instruction(messages: list[BaseMessage]) -> str:
    return next(
        message.text
        for message in reversed(messages)
        if isinstance(message, HumanMessage) and not message.text.startswith("[turn ")
    )


def _coded_seed() -> ReconstructionHistory:
    return advance_reconstruction(
        start_reconstruction("run_test", "Reconstruct the drawing.", drawing()),
        TicketAnswers(
            stage_report=StageReport(concerns={}),
            responses=_response(_ROUND_ZERO_TICKET, PipelineStage.CODING),
        ),
        workspace_output=CodingOutput(interpretation("a plate"), _verified()),
    )


def test_an_accepted_round_is_integrated_and_persisted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    views_seen: list[list[View]] = []
    coder = ScriptedChatModel(responses=_coding_script())
    auditor = ScriptedChatModel(responses=_audit_script(_accepted_audit()))

    with SandboxWorkdir() as workdir:
        projection_dir = (
            workdir.host_bind_dir / "attempts/round_000/coding/000/projection"
        )
        from PIL import Image

        projection_dir.mkdir(parents=True, exist_ok=True)
        for kind in ("overlay", "unmatched"):
            Image.new("RGB", (4, 4), "white").save(projection_dir / f"front_{kind}.png")
        diff_reports = {
            "view_front": DrawingDiffReport(
                drawing_path=workdir.host_bind_dir / "view_front.png",
                projection_path=projection_dir / "front.png",
                paths={
                    "overlay_path": projection_dir / "front_overlay.png",
                    "unmatched_path": projection_dir / "front_unmatched.png",
                },
                warnings=("Alignment may be inaccurate",),
            ),
        }
        verified = _verified()
        calls = _stub_verification(
            monkeypatch,
            replace(
                verified,
                drawing_diff_report=diff_reports,
                sandbox_verification_dir="/work/attempts/round_000/coding/000",
            ),
            views_seen=views_seen,
        )
        result = _graph(
            workdir,
            coder=coder,
            auditor=auditor,
            diff_drawer_config=OmegaConf.create(
                {"backend": "directional_chamfer", "alignment_options": {}}
            ),
        ).invoke({})
        persisted = ReconstructionHistory.model_validate_json(
            (workdir.host_bind_dir / "reconstruction.json").read_text(encoding="utf-8")
        )
        working_interpretation = DrawingInterpretation.model_validate_json(
            (workdir.host_bind_dir / "interpretation.json").read_text(encoding="utf-8")
        )
        attempted_interpretation = DrawingInterpretation.model_validate_json(
            (
                workdir.host_bind_dir
                / "attempts"
                / "round_000"
                / "interpretation"
                / "000"
                / "interpretation.json"
            ).read_text(encoding="utf-8")
        )

    assert calls == ["verify"]
    # The full_page input is not projected; the view identified on it is.
    assert views_seen == [[View.FRONT]]
    assert persisted == result["reconstruction"]
    snapshot = persisted.snapshots[0]
    assert snapshot.last_completed_stage is PipelineStage.CODING
    assert snapshot.interpretation.features == interpretation("a plate").features
    assert working_interpretation == attempted_interpretation
    assert snapshot.interpretation == working_interpretation
    assert snapshot.interpretation.views[0].image_size == (20, 20)
    assert snapshot.program_source == _PROGRAM
    assert snapshot.verification.drawing_diff_report == diff_reports
    assert [response.stage for response in snapshot.open_tickets[0].responses] == [
        PipelineStage.CODING
    ]
    assert result["audit_state"]["structured_response"].accepted is True
    assert result["stage_submission"] is None
    assert result["stage_validation_error"] is None
    audit_instruction = _last_instruction(auditor.received_messages[0])
    assert "/work/attempts/round_000/coding/000" in audit_instruction
    assert "/work/attempts/round_000/interpretation/000" not in audit_instruction
    assert "Addressed ticket_initial in coding." not in audit_instruction
    assert "open_tickets" in audit_instruction
    assert "Alignment may be inaccurate" in audit_instruction
    assert (
        audit_instruction.count(
            "/work/attempts/round_000/coding/000/projection/front_overlay.png"
        )
        == 1
    )
    assert (
        "/work/attempts/round_000/coding/000/projection/front_unmatched.png"
        in audit_instruction
    )
    assert "intermediate_returns" not in audit_instruction


def test_a_coded_checkpoint_resumes_at_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _stub_verification(monkeypatch)
    coder = ScriptedChatModel(responses=())
    auditor = ScriptedChatModel(responses=_audit_script(_accepted_audit()))

    with SandboxWorkdir() as workdir:
        result = _graph(workdir, coder=coder, auditor=auditor).invoke(
            {"reconstruction": _coded_seed()}
        )

    assert coder.received_messages == []
    assert len(auditor.received_messages) == 2
    assert calls == []
    assert result["audit_state"]["structured_response"].accepted is True


def test_a_coding_answer_that_contradicts_its_round_is_refused_inside_the_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _stub_verification(monkeypatch, _verified())
    responses = _responses(_ROUND_ZERO_TICKET, PipelineStage.CODING)
    unchecked = TicketAnswers(
        stage_report=StageReport(concerns={}),
        responses={},
    )
    checked = TicketAnswers(
        responses=responses,
        stage_report=StageReport(concerns={}),
    )
    coder = ScriptedChatModel(
        responses=(
            _write_interpretation(),
            tool_call("TicketAnswers", unchecked.model_dump(mode="json"), "answer-1"),
            tool_call("TicketAnswers", checked.model_dump(mode="json"), "answer-2"),
        )
    )

    with SandboxWorkdir() as workdir:
        result = _graph(
            workdir,
            coder=coder,
            auditor=ScriptedChatModel(responses=_audit_script(_accepted_audit())),
            coder_options={"response_format_strategy": "tool"},
        ).invoke({})

    refusal = next(
        message.text
        for message in coder.received_messages[-1]
        if isinstance(message, HumanMessage)
        and "TicketAnswers are not ready" in message.text
    )
    assert "missing ticket responses" in refusal
    assert calls == ["verify"]
    assert result["stage_validation_failure_count"] == 0
    snapshot = result["reconstruction"].snapshots[-1]
    assert snapshot.last_completed_stage is PipelineStage.CODING


def test_every_stage_reads_the_same_history_path_and_current_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_verification(monkeypatch, _verified())
    coder = ScriptedChatModel(responses=_coding_script())
    auditor = ScriptedChatModel(responses=_audit_script(_accepted_audit()))

    with SandboxWorkdir() as workdir:
        _graph(
            workdir,
            coder=coder,
            auditor=auditor,
            history_filename="history.json",
        ).invoke({})
        assert (workdir.host_bind_dir / "history.json").is_file()

    for model in (coder, auditor):
        prompt = "\n".join(message.text for message in model.received_messages[0])
        assert "/work/history.json" in prompt
        assert "round 0" in prompt


def test_the_samples_projection_method_reaches_every_prompt_and_the_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_verification(monkeypatch, _verified())
    methods: list[ProjectionMethod] = []
    verifier = coding_stage_module.InterpretationVerifier

    def recorded(**kwargs: Any) -> Any:
        methods.append(kwargs["projection_method"])
        return verifier(**kwargs)

    monkeypatch.setattr(coding_stage_module, "InterpretationVerifier", recorded)
    coder = ScriptedChatModel(responses=_coding_script())
    auditor = ScriptedChatModel(responses=_audit_script(_accepted_audit()))

    with SandboxWorkdir() as workdir:
        _graph(
            workdir,
            coder=coder,
            auditor=auditor,
            projection_method=ProjectionMethod.FIRST_ANGLE,
        ).invoke({})

    assert methods == [ProjectionMethod.FIRST_ANGLE]
    for model in (coder, auditor):
        prompt = "\n".join(message.text for message in model.received_messages[0])
        assert "uses first-angle projection" in prompt


def test_stage_specific_tools_stay_with_their_assigned_stage(monkeypatch):
    _stub_verification(monkeypatch, _verified())
    coder = ScriptedChatModel(responses=_coding_script())
    auditor = ScriptedChatModel(responses=_audit_script(_accepted_audit()))
    with SandboxWorkdir() as workdir:
        _graph(workdir, coder=coder, auditor=auditor).invoke({})
    assert auditor.bound_tool_names == (
        "run_shell",
        "load_image",
        "fit_edge",
        "evaluate_cad_hypothesis",
    )
    assert coder.bound_tool_names == (
        "run_shell",
        "load_image",
        "fit_edge",
        "calculate_drawing_scale",
        "evaluate_cad_hypothesis",
    )


_TRIAL = "/work/trials/round_000/coding/000"


def test_coder_can_build_score_and_inspect_a_trial(monkeypatch):
    calls = _stub_verification(monkeypatch, _verified())
    drawing = interpretation("a plate")
    drawing.views[0].role = View.RIGHT
    drawing.views[0].u_axis, drawing.views[0].v_axis = "-z", "+y"
    build = """python - <<'PY'
from pathlib import Path
p = Path('/work')
(p / 'tmp/history_before.json').write_bytes((p / 'reconstruction.json').read_bytes())
(p / 'tmp/trial.py').write_text(
    "import cadquery as cq\\nresult = cq.Workplane().box(30, 20, 10)\\n"
)
PY"""
    unchanged = """python - <<'PY'
from pathlib import Path
p = Path('/work')
assert (p / 'reconstruction.json').read_bytes() == (p / 'tmp/history_before.json').read_bytes()
print('history unchanged')
PY"""
    coder = ScriptedChatModel(
        responses=(
            _write_interpretation(drawing),
            tool_call("run_shell", {"command": build}, "build-trial"),
            tool_call(
                "evaluate_cad_hypothesis",
                {"source_path": "/work/tmp/trial.py"},
                "trial",
            ),
            tool_call(
                "load_image",
                {"image_path": f"{_TRIAL}/projection/right.png"},
                "inspect-trial",
            ),
            tool_call("run_shell", {"command": unchanged}, "check-history"),
            _coding_submission(),
        )
    )
    with SandboxWorkdir() as workdir:
        (workdir.host_bind_dir / "model.py").write_text(_PROGRAM)
        _graph(
            workdir,
            coder=coder,
            auditor=ScriptedChatModel(responses=_audit_script(_accepted_audit())),
            coder_options={"max_turns": 6},
        ).invoke({})
        assert (workdir.host_bind_dir / "model.py").read_text() == _PROGRAM
        trial = workdir.sandbox_to_host_path(_TRIAL)
        assert (trial / "output.step").is_file()
        assert "box(30, 20, 10)" in (trial / "hypothesis.py").read_text()

    results = {
        message.tool_call_id: message
        for message in coder.received_messages[-1]
        if isinstance(message, ToolMessage)
    }
    trial_text = results["trial"].text
    assert trial_text.startswith("[Trial build]")
    assert f"kept as {_TRIAL}/hypothesis.py" in trial_text
    assert "No build to compare with yet." in trial_text
    assert '"status":"VERIFIED"' in trial_text
    assert f"{_TRIAL}/projection/right.dxf" in trial_text
    assert any(
        block["type"] == "image" for block in results["inspect-trial"].content_blocks
    )
    assert "history unchanged" in results["check-history"].text
    assert calls == ["verify"]  # Only normal stage integration verifies a submission.


@pytest.mark.parametrize("recovers", [True, False])
def test_invalid_answers_retry_or_exhaust_before_audit(monkeypatch, recovers):
    _stub_verification(monkeypatch, *([_verified()] * 2))
    coder = ScriptedChatModel(
        responses=(
            _write_interpretation(),
            _coding_submission("ticket_absent"),
            _coding_submission() if recovers else _coding_submission("ticket_absent"),
        )
    )
    auditor = ScriptedChatModel(
        responses=_audit_script(_accepted_audit()) if recovers else ()
    )
    with SandboxWorkdir() as workdir:
        result = _graph(
            workdir,
            coder=coder,
            auditor=auditor,
            max_stage_validation_retries=1,
        ).invoke({})
    assert len(coder.received_messages) == 3
    assert "Coding Validation Error" in _last_instruction(coder.received_messages[2])
    assert "ticket_absent" in _last_instruction(coder.received_messages[2])
    snapshot = result["reconstruction"].snapshots[0]
    if recovers:
        assert snapshot.last_completed_stage is PipelineStage.CODING
        assert len(auditor.received_messages) == 2
        assert result["stage_validation_error"] is None
    else:
        assert snapshot.last_completed_stage is None
        assert auditor.received_messages == []
        assert result["stage_validation_failure_count"] == 2
        assert "ticket_absent" in result["stage_validation_error"]


@pytest.mark.parametrize("role", ["coder", "auditor"])
def test_an_agent_out_of_turns_is_told_so_and_retried_with_its_retry_turns(
    monkeypatch: pytest.MonkeyPatch, role: str
) -> None:
    _stub_verification(monkeypatch, _verified())
    looks = tuple(
        tool_call("run_shell", {"command": "true"}, f"look-{n}") for n in range(2)
    )
    script = _coding_script() if role == "coder" else _audit_script(_accepted_audit())
    agent = ScriptedChatModel(responses=(*looks, *script))
    options = {"max_turns": 2, "retry_turns": 2}
    with SandboxWorkdir() as workdir:
        graph = _graph(
            workdir,
            coder=agent if role == "coder" else ScriptedChatModel(responses=()),
            auditor=agent if role == "auditor" else ScriptedChatModel(responses=()),
            coder_options=options,
            auditor_options=options,
            max_audit_reject_count=0 if role == "coder" else 1,
        )
        result = graph.invoke(
            {} if role == "coder" else {"reconstruction": _coded_seed()}
        )

    retry = [message.text for message in agent.received_messages[2]]
    assert "before the turn budget ran out" in retry[-2]
    assert retry[-1] == "[turn set back to 0/2 to correct the refused answer]"
    assert result["stage_validation_error"] is None


def test_a_persisted_unfinished_round_can_restart_the_graph(monkeypatch):
    _stub_verification(monkeypatch, *([_verified()] * 3))
    with SandboxWorkdir() as workdir:
        stopped = _graph(
            workdir,
            coder=ScriptedChatModel(
                responses=(
                    _write_interpretation(),
                    _coding_submission("ticket_absent"),
                    _coding_submission("ticket_absent"),
                )
            ),
            auditor=ScriptedChatModel(responses=()),
            max_stage_validation_retries=1,
        ).invoke({})
        history_path = workdir.host_bind_dir / "reconstruction.json"
        checkpoint = ReconstructionHistory.model_validate_json(history_path.read_text())
        resumed = _graph(
            workdir,
            coder=ScriptedChatModel(responses=_coding_script(call_id="resume")),
            auditor=ScriptedChatModel(responses=_audit_script(_accepted_audit())),
        ).invoke({"reconstruction": checkpoint})
        persisted = ReconstructionHistory.model_validate_json(history_path.read_text())
    assert stopped["reconstruction"] == checkpoint
    assert checkpoint.snapshots[0].last_completed_stage is None
    assert resumed["reconstruction"] == persisted
    assert persisted.run_id == checkpoint.run_id
    assert persisted.snapshots[0].last_completed_stage is PipelineStage.CODING


def test_an_invalid_audit_is_corrected_before_the_agent_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_verification(monkeypatch, _verified())
    auditor = ScriptedChatModel(
        responses=_audit_script(_invalid_audit(), _accepted_audit())
    )

    with SandboxWorkdir() as workdir:
        result = _graph(
            workdir,
            coder=ScriptedChatModel(responses=_coding_script()),
            auditor=auditor,
            auditor_options={"model_call_retries": 1},
            max_stage_validation_retries=0,
        ).invoke({})

    assert len(auditor.received_messages) == 3
    correction = _last_instruction(auditor.received_messages[1])
    assert "sem_missing" in correction
    assert "Audit Validation Error" not in correction
    assert any("sem_missing" in m.text for m in result["audit_state"]["messages"])
    assert result["audit_state"]["structured_response"].accepted is True
    assert result["stage_validation_error"] is None


def test_the_graph_still_validates_an_audit_that_skipped_the_agent_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_verification(monkeypatch, _verified())
    monkeypatch.setattr(
        audit_verify_module, "validate_audit_report", lambda *_, **__: None
    )
    auditor = ScriptedChatModel(
        responses=(
            *_audit_script(_invalid_audit()),
            *_audit_script(_accepted_audit(), call_id="repair_audit"),
        )
    )

    with SandboxWorkdir() as workdir:
        result = _graph(
            workdir,
            coder=ScriptedChatModel(responses=_coding_script()),
            auditor=auditor,
            max_stage_validation_retries=1,
        ).invoke({})

    assert len(auditor.received_messages) == 4
    assert "Audit Validation Error" in _last_instruction(auditor.received_messages[2])
    assert "sem_missing" in _last_instruction(auditor.received_messages[2])
    assert "[Input artifacts]" not in _last_instruction(auditor.received_messages[2])
    assert "Recorded directory:" not in _last_instruction(auditor.received_messages[2])
    assert len(result["reconstruction"].snapshots) == 1
    assert result["audit_state"]["structured_response"].accepted is True
    assert result["stage_validation_error"] is None


def test_a_rejected_audit_opens_a_fresh_round_for_the_coder(monkeypatch):
    calls = _stub_verification(monkeypatch, _verified("000"), _verified("001"))
    coder = ScriptedChatModel(
        responses=(
            *_coding_script(),
            *_coding_script(
                _ROUND_ONE_TICKET,
                artifact=interpretation("a revised plate"),
                call_id="revision",
            ),
        )
    )
    auditor = ScriptedChatModel(
        responses=_audit_script(
            _rejected_audit("interpretation", ["sem_feature_1"]),
        )
    )
    with SandboxWorkdir() as workdir:
        result = _graph(
            workdir,
            coder=coder,
            auditor=auditor,
            max_audit_reject_count=1,
        ).invoke({})
        # Tickets reuse the evidence the auditor was asked to review.
        evidence_path = (
            result["reconstruction"].snapshots[-1].open_tickets[0].evidence_renders[0]
        )
        crop = workdir.sandbox_to_host_path(evidence_path)
        assert "/round_000/audit/000/find_missing_hole/" in evidence_path
        assert crop.is_file()
    first, second = result["reconstruction"].snapshots
    assert calls == ["verify", "verify"]
    assert first.open_tickets[0].ticket_id == _ROUND_ZERO_TICKET
    assert second.open_tickets[0].ticket_id == _ROUND_ONE_TICKET
    assert second.open_tickets[0].subject.cause == "interpretation"
    assert first.interpretation.features == interpretation("a plate").features
    assert second.interpretation.features == interpretation("a revised plate").features
    assert len(second.open_tickets[0].responses) == 1
    assert second.open_tickets[0].evidence_renders == [
        "/work/attempts/round_000/audit/000/find_missing_hole/evidence_0.png"
    ]
    assert len(coder.received_messages) == 4
    assert len(auditor.received_messages) == 2
    assert "round 1" in _last_instruction(coder.received_messages[2])


@pytest.mark.parametrize("has_findings", [False, True])
def test_audit_decision_must_match_the_validated_file_before_integration(
    monkeypatch, has_findings
):
    _stub_verification(monkeypatch, _verified())
    report = _rejected_audit() if has_findings else _accepted_audit()
    auditor = ScriptedChatModel(
        responses=(
            *_audit_script(report)[:-1],
            _message(AuditSubmission(accepted=has_findings)),
            *(
                _audit_script(_accepted_audit(), call_id="corrected_audit")
                if has_findings
                else (_message(AuditSubmission(accepted=True)),)
            ),
        )
    )
    with SandboxWorkdir() as workdir:
        result = _graph(
            workdir,
            coder=ScriptedChatModel(responses=_coding_script()),
            auditor=auditor,
            max_stage_validation_retries=1,
        ).invoke({})
    assert "accepted must be true exactly when" in _last_instruction(
        auditor.received_messages[2]
    )
    assert result["audit_state"]["structured_response"].accepted is True
    assert result["audit_report"].findings == []
    assert result["stage_validation_error"] is None
    assert len(result["reconstruction"].snapshots) == 1


def test_revision_audit_corrects_missing_reviews_without_opening_another_round(
    monkeypatch,
):
    _stub_verification(monkeypatch, _verified("000"), _verified("001"))
    audited = AuditReport(
        concern_reviews={},
        findings=[],
        ticket_reviews={
            _ROUND_ONE_TICKET: TicketReview(
                solved=True,
                summary="The previously missing hole is present in the current solid.",
            )
        },
    )
    auditor = ScriptedChatModel(
        responses=(
            *_audit_script(_rejected_audit()),
            *_audit_script(
                _accepted_audit(), _message(audited), call_id="revision_audit"
            ),
        )
    )
    with SandboxWorkdir() as workdir:
        result = _graph(
            workdir,
            coder=ScriptedChatModel(
                responses=(
                    *_coding_script(),
                    *_coding_script(_ROUND_ONE_TICKET, call_id="revision"),
                )
            ),
            auditor=auditor,
            auditor_options={"model_call_retries": 1},
            max_audit_reject_count=2,
            max_stage_validation_retries=0,
        ).invoke({})

    # Round 0 accepts [] for bootstrap work; round 1 checks [] against its ticket.
    assert len(auditor.received_messages) == 5
    correction = _last_instruction(auditor.received_messages[3])
    assert f"missing=['{_ROUND_ONE_TICKET}']" in correction
    assert "Audit Validation Error" not in correction
    assert len(result["reconstruction"].snapshots) == 2
    assert result["audit_report"] == audited
    assert result["stage_validation_error"] is None


def test_an_interpretation_revision_refreshes_parameter_values_and_preserves_history(
    monkeypatch, tmp_path
):
    ticket = "ticket_001_wrong_edge"

    def measured(value):
        return interpretation(
            features=[interpreted_feature(1, "plate", parameters={"offset": value})]
        )

    calls = _stub_verification(monkeypatch, _verified("000"), _verified("001"))
    with SandboxWorkdir(host_bind_dir=tmp_path) as workdir:
        result = _graph(
            workdir,
            coder=ScriptedChatModel(
                responses=(
                    *_coding_script(artifact=measured(1.0)),
                    *_coding_script(ticket, artifact=measured(2.5), call_id="revision"),
                )
            ),
            auditor=ScriptedChatModel(
                responses=_audit_script(_interpretation_rejected_audit())
            ),
            max_audit_reject_count=1,
        ).invoke({})
        persisted = ReconstructionHistory.model_validate_json(
            (tmp_path / "reconstruction.json").read_text()
        )
    assert calls == ["verify", "verify"]
    assert persisted == result["reconstruction"]
    first, second = persisted.snapshots
    assert first.interpretation.features[0].parameters["offset"] == 1.0
    assert second.interpretation.features[0].parameters["offset"] == 2.5
    assert [r.stage for r in second.open_tickets[0].responses] == [PipelineStage.CODING]
    assert (tmp_path / "attempts" / "round_000" / "interpretation" / "000").is_dir()
    assert (tmp_path / "attempts" / "round_001" / "interpretation" / "000").is_dir()


def test_a_revision_round_reseeds_the_accepted_interpretation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_verification(monkeypatch, _verified("000"), _verified("001"))
    coder = ScriptedChatModel(
        responses=(
            *_coding_script(),
            # The seeded file has not been reported in this stage yet.
            _coding_submission(_ROUND_ONE_TICKET),
            _coding_submission(_ROUND_ONE_TICKET),
        )
    )
    auditor = ScriptedChatModel(responses=_audit_script(_rejected_audit()))

    with SandboxWorkdir(host_bind_dir=tmp_path) as workdir:
        result = _graph(
            workdir,
            coder=coder,
            auditor=auditor,
            max_audit_reject_count=1,
        ).invoke({})

    first, second = result["reconstruction"].snapshots
    ticket = second.open_tickets[0]
    assert ticket.subject.cause == "coding"
    assert [response.stage for response in ticket.responses] == [PipelineStage.CODING]
    assert len(coder.received_messages) == 4
    assert "first verification report" in coder.received_messages[3][-1].text
    assert second.interpretation == first.interpretation
    assert (tmp_path / "attempts" / "round_001" / "interpretation").is_dir()


def test_rejection_at_the_round_limit_finishes_without_opening_another_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_verification(monkeypatch, _verified())

    auditor = ScriptedChatModel(responses=_audit_script(_rejected_audit()))
    with SandboxWorkdir() as workdir:
        result = _graph(
            workdir,
            coder=ScriptedChatModel(responses=_coding_script()),
            auditor=auditor,
            max_audit_reject_count=0,
        ).invoke({})

    assert len(result["reconstruction"].snapshots) == 1
    assert auditor.received_messages == []
    assert result["audit_report"] is None
    assert result["stage_validation_error"] is None


_COMPACTION_NOTES = (
    "## OBJECTIVE\nrebuild the part\n## FINDINGS\nmeasured the views\n"
    "## WORK DONE\nNone\n## OPEN QUESTIONS\nNone"
)


@pytest.mark.parametrize("rounds", [1, 2])
def test_the_coder_starts_its_next_round_from_notes(monkeypatch, rounds):
    """Compaction runs before the audit only when the audit may open a round."""
    _stub_verification(monkeypatch, _verified("000"), _verified("001"))
    notetaker = ScriptedChatModel(responses=(AIMessage(content=_COMPACTION_NOTES),))
    coder = ScriptedChatModel(
        responses=(
            *_coding_script(),
            *_coding_script(_ROUND_ONE_TICKET, call_id="revision"),
        )
    )
    auditor = ScriptedChatModel(responses=_audit_script(_rejected_audit()))
    with SandboxWorkdir() as workdir:
        _graph(
            workdir,
            coder=coder,
            auditor=auditor,
            compaction_model=notetaker,
            max_audit_reject_count=rounds - 1,
        ).invoke({})

    assert len(notetaker.received_messages) == rounds - 1
    assert len(coder.received_messages) == 2 * rounds
    system = coder.received_messages[0][0].text
    assert system.startswith("## Reconstruction workflow")
    assert system.rstrip().endswith(ROLE_PATHS["coder"].read_text().strip())
    if rounds == 2:
        round_one = [message.text for message in coder.received_messages[2]]
        assert SUMMARY_PREAMBLE.format(notes=_COMPACTION_NOTES) in round_one
        assert sum("[Input artifacts]" in text for text in round_one) == 1


@pytest.mark.parametrize("mode", ["never", "once", "every"])
def test_a_coder_out_of_turns_is_retried_from_notes_and_restated_instructions(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """Only a coder whose turns ran out is compacted, and only when asked for."""
    _stub_verification(monkeypatch, _verified())
    notetaker = ScriptedChatModel(responses=(AIMessage(content=_COMPACTION_NOTES),))
    looks = tuple(
        tool_call("run_shell", {"command": "true"}, f"look-{n}") for n in range(2)
    )
    coder = ScriptedChatModel(responses=(*looks, *_coding_script()))
    with SandboxWorkdir() as workdir:
        result = _graph(
            workdir,
            coder=coder,
            auditor=ScriptedChatModel(responses=()),
            coder_options={"max_turns": 2, "retry_turns": 2},
            compaction_model=notetaker,
            compact_exhausted_coder=mode,
            max_audit_reject_count=0,
        ).invoke({})

    compact = mode != "never"
    assert result["reconstruction"].snapshots[0].last_completed_stage is (
        PipelineStage.CODING
    )
    retry = coder.received_messages[2]
    texts = [message.text for message in retry]
    assert "before the turn budget ran out" in texts[-2]
    assert texts[-1] == "[turn set back to 0/2 to correct the refused answer]"
    assert len(notetaker.received_messages) == int(compact)
    if compact:
        # The notes replace every turn; the instructions come back in full.
        assert texts[1] == SUMMARY_PREAMBLE.format(notes=_COMPACTION_NOTES)
        assert not any(isinstance(message, AIMessage) for message in retry)
        assert sum("[Input artifacts]" in text for text in texts) == 1
    else:
        assert sum(isinstance(message, AIMessage) for message in retry) == 2


def test_compacting_an_exhausted_coder_needs_a_compaction_model() -> None:
    with SandboxWorkdir() as workdir, pytest.raises(ValueError, match="compact"):
        _graph(
            workdir,
            coder=ScriptedChatModel(responses=()),
            auditor=ScriptedChatModel(responses=()),
            compact_exhausted_coder="once",
        )


@pytest.mark.parametrize(("mode", "compactions"), [("once", 1), ("every", 2)])
def test_once_compacts_only_the_first_time_a_rounds_turns_run_out(
    monkeypatch: pytest.MonkeyPatch, mode: str, compactions: int
) -> None:
    _stub_verification(monkeypatch, _verified())
    notes = AIMessage(content=_COMPACTION_NOTES)
    notetaker = ScriptedChatModel(responses=(notes, notes))
    looks = tuple(
        tool_call("run_shell", {"command": "true"}, f"look-{n}") for n in range(4)
    )
    coder = ScriptedChatModel(responses=(*looks, *_coding_script()))
    with SandboxWorkdir() as workdir:
        result = _graph(
            workdir,
            coder=coder,
            auditor=ScriptedChatModel(responses=()),
            coder_options={"max_turns": 2, "retry_turns": 2},
            compaction_model=notetaker,
            compact_exhausted_coder=mode,
            max_audit_reject_count=0,
        ).invoke({})

    assert result["reconstruction"].snapshots[0].last_completed_stage is (
        PipelineStage.CODING
    )
    assert len(notetaker.received_messages) == compactions
    # The second retry keeps the first retry's turns unless compacted again.
    second_retry = coder.received_messages[4]
    kept = sum(isinstance(message, AIMessage) for message in second_retry)
    assert kept == (2 if mode == "once" else 0)


def test_once_compacts_again_in_the_next_round(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_verification(monkeypatch, _verified("000"), _verified("001"))
    notes = AIMessage(content=_COMPACTION_NOTES)
    notetaker = ScriptedChatModel(responses=(notes, notes, notes))

    def looks(tag: str) -> tuple[AIMessage, ...]:
        return tuple(
            tool_call("run_shell", {"command": "true"}, f"{tag}-{n}") for n in range(2)
        )

    coder = ScriptedChatModel(
        responses=(
            *looks("round-0"),
            *_coding_script(),
            *looks("round-1"),
            *_coding_script(_ROUND_ONE_TICKET, call_id="revision"),
        )
    )
    with SandboxWorkdir() as workdir:
        _graph(
            workdir,
            coder=coder,
            auditor=ScriptedChatModel(responses=_audit_script(_rejected_audit())),
            coder_options={"max_turns": 2, "retry_turns": 2},
            compaction_model=notetaker,
            compact_exhausted_coder="once",
            max_audit_reject_count=1,
        ).invoke({})

    # Round 0's retry, the handover before the audit, and round 1's retry.
    assert len(notetaker.received_messages) == 3


def test_an_unknown_exhausted_coder_compaction_is_refused() -> None:
    with (
        SandboxWorkdir() as workdir,
        pytest.raises(ValueError, match="RetryCompaction"),
    ):
        _graph(
            workdir,
            coder=ScriptedChatModel(responses=()),
            auditor=ScriptedChatModel(responses=()),
            compaction_model=ScriptedChatModel(responses=()),
            compact_exhausted_coder="sometimes",
        )
