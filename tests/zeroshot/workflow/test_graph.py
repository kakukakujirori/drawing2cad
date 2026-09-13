"""C4 workflow wiring: submissions, integration, retries, and audit rounds."""

import base64
import sys
from collections.abc import Sequence
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.messages.content import ContentBlock

from tests.zeroshot.chat_models import ScriptedChatModel, tool_call
from tests.zeroshot.contracts import (
    drawing,
    interpretation,
    interpreted_feature,
)
from zeroshot.pipeline.messages.artifact import ArtifactPresenter
from zeroshot.pipeline.messages.manifest import InputManifest, register_view
from zeroshot.pipeline.messages.tickets import TicketAnswers, TicketResponse
from zeroshot.pipeline.sandbox import SandboxRunner, SandboxWorkdir
from zeroshot.pipeline.stages.audit.contracts import (
    AuditFinding,
    AuditReport,
    RevisionRequest,
    StageOutputRef,
)
from zeroshot.pipeline.stages.coding import stage as coding_stage_module
from zeroshot.pipeline.stages.contracts import ReconstructionRun
from zeroshot.pipeline.stages.interpretation.contracts import (
    DrawingInterpretation,
    DrawingView,
    Region,
    View,
)
from zeroshot.pipeline.stages.operations.contracts import (
    Operation,
    OperationPlan,
    OperationVerb,
)
from zeroshot.pipeline.stages.types import PipelineStage
from zeroshot.pipeline.verification import (
    ExecutionStatus,
    VerifyOutputResult,
)
from zeroshot.pipeline.workflow import create_agent
from zeroshot.pipeline.workflow.graph import AgentBuilder, create_reconstruction_graph
from zeroshot.pipeline.workflow.lifecycle import (
    advance_reconstruction,
    start_reconstruction,
)

_ROUND_ZERO_TICKET = "ticket_initial"
_ROUND_ONE_TICKET = "ticket_001_missing_hole"
_PROGRAM = "ret_step1 = object()\nresult = ret_step1\n"


def _message(answer: object) -> AIMessage:
    return AIMessage(content=answer.model_dump_json())  # type: ignore[attr-defined]


def _response(ticket_id: str, stage: PipelineStage) -> TicketResponse:
    return TicketResponse(
        ticket_id=ticket_id,
        stage=stage,  # type: ignore[arg-type]
        summary=f"Addressed {ticket_id} in {stage.value}.",
    )


def _responses(ticket_id: str | None, stage: PipelineStage) -> list[TicketResponse]:
    return [_response(ticket_id, stage)] if ticket_id is not None else []


def _interpretation_submission(
    ticket_id: str | None = _ROUND_ZERO_TICKET,
) -> AIMessage:
    return _message(
        TicketAnswers(
            responses=_responses(ticket_id, PipelineStage.INTERPRETATION),
        )
    )


def _invalid_interpretation_submission() -> AIMessage:
    return _message(
        TicketAnswers(
            responses=_responses("ticket_absent", PipelineStage.INTERPRETATION),
        )
    )


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
                "region": Region(view="view_input", box_px=(0, 0, 10, 10)),
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


def _interpretation_script(
    ticket_id: str | None = _ROUND_ZERO_TICKET,
    *,
    artifact: DrawingInterpretation | None = None,
    call_id: str = "draw",
) -> tuple[AIMessage, AIMessage]:
    return _write_interpretation(artifact, call_id), _interpretation_submission(
        ticket_id
    )


def _plan(*, builds: Sequence[int | str] = (1,), detail: str = "extrude"):
    return OperationPlan(
        proposal=[
            Operation(
                name="op_step1",
                verb=OperationVerb.EXTRUDE,
                detail=detail,
                depends_on=[],
                semantics=[
                    f"sem_feature_{value}" if isinstance(value, int) else value
                    for value in builds
                ],
            )
        ],
        rationale="The plate is one extrusion.",
    )


def _operation_submission(
    ticket_id: str | None = _ROUND_ZERO_TICKET,
) -> AIMessage:
    return _message(
        TicketAnswers(responses=_responses(ticket_id, PipelineStage.OPERATIONS))
    )


def _write_operations(plan: OperationPlan, call_id: str = "plan") -> AIMessage:
    payload = base64.b64encode(
        (plan.model_dump_json(indent=2) + "\n").encode()
    ).decode()
    command = (
        'python -c "import base64;'
        "open('/work/operations.json','wb').write(base64.b64decode('"
        + payload
        + "'))\""
    )
    return tool_call("run_shell", {"command": command}, call_id)


def _operations_script(
    ticket_id: str | None = _ROUND_ZERO_TICKET,
    *,
    builds: Sequence[int | str] = (1,),
    detail: str = "extrude",
    call_id: str = "plan",
) -> tuple[AIMessage, AIMessage]:
    return (
        _write_operations(_plan(builds=builds, detail=detail), call_id),
        _operation_submission(ticket_id),
    )


def _coding_submission(ticket_id: str | None = _ROUND_ZERO_TICKET) -> AIMessage:
    return _message(
        TicketAnswers(
            responses=_responses(ticket_id, PipelineStage.CODING),
        )
    )


def _accepted_audit() -> AIMessage:
    return _message(AuditReport(accepted=True, findings=[]))


def _rejected_audit(root: StageOutputRef | None = None) -> AIMessage:
    return _message(
        AuditReport(
            accepted=False,
            findings=[
                AuditFinding(
                    name="find_missing_hole",
                    observation="The drawing contains a hole that the model omits.",
                    evidence=["render_3d/hlg_front.png"],
                    backtrace=[],
                    revision_request=RevisionRequest(
                        action="modify",
                        targets=[
                            root
                            or StageOutputRef(
                                stage=PipelineStage.CODING,
                                name="ret_step1",
                            )
                        ],
                        instruction="Implement the missing hole.",
                        proposed_names=[],
                    ),
                )
            ],
        )
    )


def _interpretation_rejected_audit() -> AIMessage:
    return _message(
        AuditReport(
            accepted=False,
            findings=[
                AuditFinding(
                    name="find_wrong_edge",
                    observation="The front edge starts at the wrong coordinate.",
                    evidence=["view_front", "sem_feature_1.offset"],
                    backtrace=[],
                    revision_request=RevisionRequest(
                        action="modify",
                        targets=[
                            StageOutputRef(
                                stage=PipelineStage.INTERPRETATION,
                                name="view_front",
                            )
                        ],
                        instruction="Correct the front sheet's edge reading.",
                        proposed_names=[],
                    ),
                )
            ],
        )
    )


def _invalid_audit() -> AIMessage:
    return _message(
        AuditReport(
            accepted=False,
            findings=[
                AuditFinding(
                    name="find_unknown_operation",
                    observation="The model is incorrect.",
                    evidence=["verification.status"],
                    backtrace=[],
                    revision_request=RevisionRequest(
                        action="modify",
                        targets=[
                            StageOutputRef(
                                stage=PipelineStage.OPERATIONS,
                                name="op_missing",
                            )
                        ],
                        instruction="Correct the absent operation.",
                        proposed_names=[],
                    ),
                )
            ],
        )
    )


def _agent(role: str, model: BaseChatModel, **overrides: Any) -> AgentBuilder:
    return partial(create_agent, role=role, model=model, **overrides)


def _artifact_presenter() -> ArtifactPresenter:
    return ArtifactPresenter(input_mode="path", feedback_mode="none")


def _graph(
    workdir: SandboxWorkdir,
    *,
    interpreter: ScriptedChatModel,
    planner: ScriptedChatModel,
    coder: ScriptedChatModel,
    auditor: ScriptedChatModel,
    history_filename: str = "reconstruction.json",
    **overrides: Any,
):
    from PIL import Image

    path = workdir.host_bind_dir / "drawing.png"
    Image.new("RGB", (20, 20), "white").save(path)
    Image.new("RGB", (20, 20), "white").save(workdir.host_bind_dir / "front.png")
    common = {
        "announce_turns": False,
        "model_retries": 0,
        "checkpointer": False,
        "max_turns": 5,
    }
    return create_reconstruction_graph(
        interpretation_agent_builder=_agent(
            "drawing_interpreter", interpreter, **common
        ),
        operations_agent_builder=_agent("operation_planner", planner, **common),
        coding_agent_builder=_agent("coder", coder, **common),
        audit_agent_builder=_agent("output_auditor", auditor, **common),
        sandbox_runner=SandboxRunner(
            python_executable=Path(sys.executable), default_timeout_s=10
        ),
        sandbox_workdir=workdir,
        artifact_presenter=_artifact_presenter(),
        input_manifest=InputManifest(
            sample_id="test",
            drawing=[register_view("view_input", View.FULL_PAGE, path)],
        ),
        reconstruction_history_filename=history_filename,
        **overrides,
    )


def _stub_verification(
    monkeypatch: pytest.MonkeyPatch,
    *reports: VerifyOutputResult,
) -> list[str]:
    remaining = list(reports)
    calls: list[str] = []

    class StubVerifier:
        def __init__(self, workdir: SandboxWorkdir, source_filename: str) -> None:
            self.source_path = workdir.host_bind_dir / source_filename

        @property
        def confirmed(self) -> bool:
            return True

        def reset(self) -> None:
            pass

        def verify(self) -> tuple[VerifyOutputResult, None]:
            calls.append("verify")
            return remaining.pop(0), None

        def feedback(self) -> list[ContentBlock]:
            return []

    monkeypatch.setattr(
        coding_stage_module,
        "OutputVerifier",
        lambda **kwargs: StubVerifier(
            workdir=kwargs["workdir"],
            source_filename=kwargs["source_filename"],
        ),
    )
    return calls


def _verified(identifier: str = "000", source: str = _PROGRAM) -> VerifyOutputResult:
    return VerifyOutputResult(
        verification_id=identifier,
        status=ExecutionStatus.VERIFIED,
        source=source,
        returncode=0,
    )


def _last_instruction(messages: list[BaseMessage]) -> str:
    return next(
        message.text
        for message in reversed(messages)
        if isinstance(message, HumanMessage) and not message.text.startswith("[turn ")
    )


def _interpretation_seed() -> ReconstructionRun:
    return advance_reconstruction(
        start_reconstruction("run_test", "Reconstruct the drawing.", drawing()),
        TicketAnswers(
            responses=[_response(_ROUND_ZERO_TICKET, PipelineStage.INTERPRETATION)]
        ),
        workspace_output=interpretation("a plate"),
    )


def _operations_resume() -> ReconstructionRun:
    return advance_reconstruction(
        _interpretation_seed(),
        TicketAnswers(
            responses=[_response(_ROUND_ZERO_TICKET, PipelineStage.OPERATIONS)]
        ),
        workspace_output=_plan(),
    )


@pytest.mark.parametrize("has_returns", [False, True])
def test_an_accepted_round_is_integrated_and_persisted(
    monkeypatch: pytest.MonkeyPatch,
    has_returns: bool,
) -> None:
    calls = _stub_verification(
        monkeypatch,
        replace(
            _verified(), intermediate_returns="ret_step1: solid" if has_returns else ""
        ),
    )
    interpreter = ScriptedChatModel(responses=_interpretation_script())
    planner = ScriptedChatModel(responses=_operations_script())
    coder = ScriptedChatModel(responses=(_coding_submission(),))
    auditor = ScriptedChatModel(responses=(_accepted_audit(),))

    with SandboxWorkdir() as workdir:
        result = _graph(
            workdir,
            interpreter=interpreter,
            planner=planner,
            coder=coder,
            auditor=auditor,
        ).invoke({})
        persisted = ReconstructionRun.model_validate_json(
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
    assert persisted == result["reconstruction"]
    snapshot = persisted.snapshots[0]
    assert snapshot.last_completed_stage is PipelineStage.CODING
    assert snapshot.interpretation.features == interpretation("a plate").features
    assert snapshot.operations == _plan()
    assert working_interpretation == attempted_interpretation
    assert snapshot.interpretation == working_interpretation
    assert snapshot.interpretation.views[0].image_size == (20, 20)
    assert snapshot.program_source == _PROGRAM
    assert [response.stage for response in snapshot.open_tickets[0].responses] == [
        PipelineStage.INTERPRETATION,
        PipelineStage.OPERATIONS,
        PipelineStage.CODING,
    ]
    assert result["audit_report"].accepted is True
    assert result["stage_submission"] is None
    assert result["stage_validation_error"] is None
    assert "within 5 turns" in auditor.received_messages[0][0].text
    audit_instruction = _last_instruction(auditor.received_messages[0])
    assert "/work/attempts/round_000/coding/000" in audit_instruction
    assert "/work/attempts/round_000/interpretation/000" not in audit_instruction
    assert "Addressed ticket_initial in coding." in audit_instruction
    returns_dir = "/work/attempts/round_000/coding/000/intermediate_returns"
    assert (returns_dir in audit_instruction) is has_returns
    assert ("Recorded directory: unavailable" in audit_instruction) is not has_returns
    assert "what the plan meant it to" in auditor.received_messages[0][0].text


def test_an_interpretation_seed_starts_at_operations_without_calling_interpretation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _stub_verification(monkeypatch, _verified())
    interpreter = ScriptedChatModel(responses=())
    planner = ScriptedChatModel(responses=_operations_script())
    coder = ScriptedChatModel(responses=(_coding_submission(),))
    auditor = ScriptedChatModel(responses=(_accepted_audit(),))

    with SandboxWorkdir() as workdir:
        result = _graph(
            workdir,
            interpreter=interpreter,
            planner=planner,
            coder=coder,
            auditor=auditor,
        ).invoke({"reconstruction": _interpretation_seed()})
        persisted = ReconstructionRun.model_validate_json(
            (workdir.host_bind_dir / "reconstruction.json").read_text(encoding="utf-8")
        )

    assert interpreter.received_messages == []
    assert len(planner.received_messages) == 2
    assert calls == ["verify"]
    assert persisted == result["reconstruction"]
    assert persisted.snapshots[-1].interpretation == interpretation("a plate")
    assert persisted.snapshots[-1].last_completed_stage is PipelineStage.CODING


def test_an_operations_checkpoint_resumes_at_coding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _stub_verification(monkeypatch, _verified())
    interpreter = ScriptedChatModel(responses=())
    planner = ScriptedChatModel(responses=())
    coder = ScriptedChatModel(responses=(_coding_submission(),))
    auditor = ScriptedChatModel(responses=(_accepted_audit(),))

    with SandboxWorkdir() as workdir:
        result = _graph(
            workdir,
            interpreter=interpreter,
            planner=planner,
            coder=coder,
            auditor=auditor,
        ).invoke({"reconstruction": _operations_resume()})

    assert interpreter.received_messages == []
    assert planner.received_messages == []
    assert len(coder.received_messages) == 1
    assert len(auditor.received_messages) == 1
    assert calls == ["verify"]
    assert (
        result["reconstruction"].snapshots[-1].last_completed_stage
        is PipelineStage.CODING
    )


def test_every_stage_reads_the_same_history_path_and_current_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_verification(monkeypatch, _verified())
    interpreter = ScriptedChatModel(responses=_interpretation_script())
    planner = ScriptedChatModel(responses=_operations_script())
    coder = ScriptedChatModel(responses=(_coding_submission(),))
    auditor = ScriptedChatModel(responses=(_accepted_audit(),))

    with SandboxWorkdir() as workdir:
        _graph(
            workdir,
            interpreter=interpreter,
            planner=planner,
            coder=coder,
            auditor=auditor,
            history_filename="history.json",
        ).invoke({})
        assert (workdir.host_bind_dir / "history.json").is_file()

    for model in (interpreter, planner, coder, auditor):
        prompt = "\n".join(message.text for message in model.received_messages[0])
        assert "/work/history.json" in prompt
        assert "round 0" in prompt


def test_only_the_interpretation_stage_receives_the_scale_tool(monkeypatch):
    _stub_verification(monkeypatch, _verified())
    interpreter = ScriptedChatModel(responses=_interpretation_script())
    planner = ScriptedChatModel(responses=_operations_script())
    coder = ScriptedChatModel(responses=(_coding_submission(),))
    auditor = ScriptedChatModel(responses=(_accepted_audit(),))
    with SandboxWorkdir() as workdir:
        _graph(
            workdir,
            interpreter=interpreter,
            planner=planner,
            coder=coder,
            auditor=auditor,
        ).invoke({})
    assert interpreter.bound_tool_names == (
        "run_shell",
        "load_image",
        "calculate_drawing_scale",
    )
    for model in (planner, coder, auditor):
        assert model.bound_tool_names == ("run_shell", "load_image")


def test_invalid_operations_retry_without_reaching_coding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _stub_verification(monkeypatch, _verified())
    interpreter = ScriptedChatModel(responses=_interpretation_script())
    planner = ScriptedChatModel(
        responses=(
            *_operations_script("ticket_absent"),
            _operation_submission(),
        )
    )
    coder = ScriptedChatModel(responses=(_coding_submission(),))

    with SandboxWorkdir() as workdir:
        result = _graph(
            workdir,
            interpreter=interpreter,
            planner=planner,
            coder=coder,
            auditor=ScriptedChatModel(responses=(_accepted_audit(),)),
            max_stage_validation_retries=1,
        ).invoke({})

    assert len(planner.received_messages) == 3
    assert len(coder.received_messages) == 1
    assert calls == ["verify"]
    retry = _last_instruction(planner.received_messages[-1])
    assert "Operations Validation Error" in retry
    assert "ticket_absent" in retry
    validation_message = next(
        message
        for message in planner.received_messages[-1]
        if isinstance(message, HumanMessage)
        and "Operations Validation Error" in message.text
    )
    assert isinstance(validation_message.content, list)
    assert all(isinstance(block, dict) for block in validation_message.content)

    # The round's terms and guidelines already stand in the transcript, so a
    # re-ask that repeated them would pay for them twice.
    assert "Revise the operation plan" not in retry
    assert "Guidelines:" not in retry

    assert result["reconstruction"].snapshots[0].operations == _plan()
    assert result["stage_validation_failure_count"] == 0


@pytest.mark.parametrize("recovers", [True, False])
def test_invalid_interpretations_retry_or_exhaust_before_operations(
    monkeypatch, recovers
):
    _stub_verification(monkeypatch, *([_verified()] if recovers else []))
    interpreter = ScriptedChatModel(
        responses=(
            _write_interpretation(),
            _invalid_interpretation_submission(),
            _interpretation_submission()
            if recovers
            else _invalid_interpretation_submission(),
        )
    )
    planner = ScriptedChatModel(responses=_operations_script() if recovers else ())
    coder = ScriptedChatModel(responses=(_coding_submission(),) if recovers else ())
    auditor = ScriptedChatModel(responses=(_accepted_audit(),) if recovers else ())
    with SandboxWorkdir() as workdir:
        result = _graph(
            workdir,
            interpreter=interpreter,
            planner=planner,
            coder=coder,
            auditor=auditor,
            max_stage_validation_retries=1,
        ).invoke({})
    assert len(interpreter.received_messages) == 3
    assert "Interpretation Validation Error" in _last_instruction(
        interpreter.received_messages[2]
    )
    assert "ticket_absent" in _last_instruction(interpreter.received_messages[2])
    snapshot = result["reconstruction"].snapshots[0]
    if recovers:
        assert snapshot.last_completed_stage is PipelineStage.CODING
        assert len(planner.received_messages) == 2
        assert result["stage_validation_error"] is None
    else:
        assert snapshot.last_completed_stage is None
        assert planner.received_messages == []
        assert result["stage_validation_failure_count"] == 2
        assert "ticket_absent" in result["stage_validation_error"]


def test_stage_validation_retry_limit_stops_before_downstream_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _stub_verification(monkeypatch)
    planner = ScriptedChatModel(
        responses=(
            *_operations_script("ticket_absent"),
            _operation_submission("ticket_absent"),
        )
    )
    coder = ScriptedChatModel(responses=())
    auditor = ScriptedChatModel(responses=())

    with SandboxWorkdir() as workdir:
        result = _graph(
            workdir,
            interpreter=ScriptedChatModel(responses=_interpretation_script()),
            planner=planner,
            coder=coder,
            auditor=auditor,
            max_stage_validation_retries=1,
        ).invoke({})

    assert len(planner.received_messages) == 3
    assert coder.received_messages == []
    assert auditor.received_messages == []
    assert calls == []
    snapshot = result["reconstruction"].snapshots[0]
    assert snapshot.last_completed_stage is PipelineStage.INTERPRETATION
    assert snapshot.operations is None
    assert result["stage_validation_failure_count"] == 2
    assert "ticket_absent" in result["stage_validation_error"]


def test_a_persisted_interpretation_checkpoint_can_restart_the_graph(monkeypatch):
    _stub_verification(monkeypatch, _verified())
    with SandboxWorkdir() as workdir:
        stopped = _graph(
            workdir,
            interpreter=ScriptedChatModel(
                responses=(
                    _write_interpretation(),
                    _invalid_interpretation_submission(),
                    _invalid_interpretation_submission(),
                )
            ),
            planner=ScriptedChatModel(responses=()),
            coder=ScriptedChatModel(responses=()),
            auditor=ScriptedChatModel(responses=()),
            max_stage_validation_retries=1,
        ).invoke({})
        history_path = workdir.host_bind_dir / "reconstruction.json"
        checkpoint = ReconstructionRun.model_validate_json(history_path.read_text())
        resumed = _graph(
            workdir,
            interpreter=ScriptedChatModel(
                responses=_interpretation_script(call_id="resume")
            ),
            planner=ScriptedChatModel(responses=_operations_script()),
            coder=ScriptedChatModel(responses=(_coding_submission(),)),
            auditor=ScriptedChatModel(responses=(_accepted_audit(),)),
        ).invoke({"reconstruction": checkpoint})
        persisted = ReconstructionRun.model_validate_json(history_path.read_text())
    assert stopped["reconstruction"] == checkpoint
    assert checkpoint.snapshots[0].last_completed_stage is None
    assert resumed["reconstruction"] == persisted
    assert persisted.run_id == checkpoint.run_id
    assert persisted.snapshots[0].last_completed_stage is PipelineStage.CODING


def test_an_invalid_audit_is_retried_against_the_same_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_verification(monkeypatch, _verified())
    auditor = ScriptedChatModel(responses=(_invalid_audit(), _accepted_audit()))

    with SandboxWorkdir() as workdir:
        result = _graph(
            workdir,
            interpreter=ScriptedChatModel(responses=_interpretation_script()),
            planner=ScriptedChatModel(responses=_operations_script()),
            coder=ScriptedChatModel(responses=(_coding_submission(),)),
            auditor=auditor,
            max_stage_validation_retries=1,
        ).invoke({})

    assert len(auditor.received_messages) == 2
    assert "Audit Validation Error" in _last_instruction(auditor.received_messages[1])
    assert "op_missing" in _last_instruction(auditor.received_messages[1])
    assert "[Input artifacts]" not in _last_instruction(auditor.received_messages[1])
    assert "Recorded directory:" not in _last_instruction(auditor.received_messages[1])
    assert len(result["reconstruction"].snapshots) == 1
    assert result["audit_report"].accepted is True
    assert result["stage_validation_error"] is None


def test_a_rejected_audit_opens_a_fresh_round_for_all_reasoning_stages(monkeypatch):
    calls = _stub_verification(monkeypatch, _verified("000"), _verified("001"))
    interpreter = ScriptedChatModel(
        responses=(
            *_interpretation_script(),
            *_interpretation_script(
                _ROUND_ONE_TICKET,
                artifact=interpretation("a revised plate"),
                call_id="revision",
            ),
        )
    )
    planner = ScriptedChatModel(
        responses=(
            *_operations_script(),
            *_operations_script(
                _ROUND_ONE_TICKET, detail="extrude revised plate", call_id="replan"
            ),
        )
    )
    coder = ScriptedChatModel(
        responses=(_coding_submission(), _coding_submission(_ROUND_ONE_TICKET))
    )
    auditor = ScriptedChatModel(
        responses=(
            _rejected_audit(
                StageOutputRef(stage=PipelineStage.INTERPRETATION, name="sem_feature_1")
            ),
        )
    )
    with SandboxWorkdir() as workdir:
        result = _graph(
            workdir,
            interpreter=interpreter,
            planner=planner,
            coder=coder,
            auditor=auditor,
            max_audit_reject_count=1,
        ).invoke({})
    first, second = result["reconstruction"].snapshots
    assert calls == ["verify", "verify"]
    assert first.open_tickets[0].ticket_id == _ROUND_ZERO_TICKET
    assert second.open_tickets[0].ticket_id == _ROUND_ONE_TICKET
    assert first.interpretation.features == interpretation("a plate").features
    assert second.interpretation.features == interpretation("a revised plate").features
    assert first.operations == _plan()
    assert second.operations == _plan(detail="extrude revised plate")
    assert len(second.open_tickets[0].responses) == 3
    assert len(interpreter.received_messages) == 4
    assert len(planner.received_messages) == 4
    assert len(coder.received_messages) == 2
    assert len(auditor.received_messages) == 1
    assert "round 1" in _last_instruction(planner.received_messages[2])


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
            interpreter=ScriptedChatModel(
                responses=(
                    *_interpretation_script(artifact=measured(1.0)),
                    *_interpretation_script(
                        ticket, artifact=measured(2.5), call_id="revision"
                    ),
                )
            ),
            planner=ScriptedChatModel(
                responses=(
                    *_operations_script(detail="start at sem_feature_1.offset"),
                    # A round that changes nothing still hands on the seeded plan,
                    # whose references are refreshed against the new interpretation.
                    _operation_submission(ticket),
                )
            ),
            coder=ScriptedChatModel(
                responses=(_coding_submission(), _coding_submission(ticket))
            ),
            auditor=ScriptedChatModel(responses=(_interpretation_rejected_audit(),)),
            max_audit_reject_count=1,
        ).invoke({})
        persisted = ReconstructionRun.model_validate_json(
            (tmp_path / "reconstruction.json").read_text()
        )
    assert calls == ["verify", "verify"]
    assert persisted == result["reconstruction"]
    first, second = persisted.snapshots
    assert first.interpretation.features[0].parameters["offset"] == 1.0
    assert second.interpretation.features[0].parameters["offset"] == 2.5
    assert first.operations.proposal[0].detail.endswith("(= 1.0)")
    assert second.operations.proposal[0].detail.endswith("(= 2.5)")
    assert [r.stage for r in second.open_tickets[0].responses] == [
        PipelineStage.INTERPRETATION,
        PipelineStage.OPERATIONS,
        PipelineStage.CODING,
    ]
    assert (tmp_path / "attempts" / "round_000" / "interpretation" / "000").is_dir()
    assert (tmp_path / "attempts" / "round_001" / "interpretation" / "000").is_dir()


def test_a_coding_rooted_finding_reopens_the_round_for_coding_alone(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_verification(monkeypatch, _verified("000"), _verified("001"))
    interpreter = ScriptedChatModel(responses=_interpretation_script())
    planner = ScriptedChatModel(responses=_operations_script())
    coder = ScriptedChatModel(
        responses=(_coding_submission(), _coding_submission(_ROUND_ONE_TICKET))
    )
    auditor = ScriptedChatModel(responses=(_rejected_audit(),))

    with SandboxWorkdir(host_bind_dir=tmp_path) as workdir:
        result = _graph(
            workdir,
            interpreter=interpreter,
            planner=planner,
            coder=coder,
            auditor=auditor,
            max_audit_reject_count=1,
        ).invoke({})

    first, second = result["reconstruction"].snapshots
    ticket = second.open_tickets[0]
    assert ticket.assigned_stages == [PipelineStage.CODING]
    assert [response.stage for response in ticket.responses] == [PipelineStage.CODING]

    # The unassigned stages were not asked again, and their artifacts stand.
    assert len(interpreter.received_messages) == 2
    assert len(planner.received_messages) == 2
    assert len(coder.received_messages) == 2
    assert second.interpretation == first.interpretation
    assert second.operations == first.operations
    assert not (tmp_path / "attempts" / "round_001" / "interpretation").exists()


def test_rejection_at_the_round_limit_finishes_without_opening_another_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_verification(monkeypatch, _verified())

    auditor = ScriptedChatModel(responses=(_rejected_audit(),))
    with SandboxWorkdir() as workdir:
        result = _graph(
            workdir,
            interpreter=ScriptedChatModel(responses=_interpretation_script()),
            planner=ScriptedChatModel(responses=_operations_script()),
            coder=ScriptedChatModel(responses=(_coding_submission(),)),
            auditor=auditor,
            max_audit_reject_count=0,
            share_thread=True,
        ).invoke({})

    assert len(result["reconstruction"].snapshots) == 1
    assert auditor.received_messages == []
    assert result["audit_report"] is None
    assert result["stage_validation_error"] is None
