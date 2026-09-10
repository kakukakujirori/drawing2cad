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
from tests.zeroshot.contracts import drawing, evidence, hypothesis, replacing, sheet
from zeroshot.pipeline.messages import (
    ArtifactPresenter,
    DrawingSource,
    InputManifest,
    View,
    unread_sheet,
)
from zeroshot.pipeline.messages.contracts import (
    CropOf,
    Operation,
    OperationPlan,
    OperationVerb,
    PipelineStage,
)
from zeroshot.pipeline.messages.contracts.audit import (
    AuditFinding,
    AuditReport,
    RevisionRequest,
    StageOutputRef,
)
from zeroshot.pipeline.messages.contracts.reconstruction import (
    CodingSubmission,
    DrawingSubmission,
    OperationSubmission,
    ReconstructionRun,
    SemanticSubmission,
    TicketResponse,
)
from zeroshot.pipeline.sandbox import SandboxRunner, SandboxWorkdir
from zeroshot.pipeline.stages.coding import stage as coding_stage_module
from zeroshot.pipeline.verification import (
    ExecutionStatus,
    VerifyOutputResult,
)
from zeroshot.pipeline.workflow import create_agent
from zeroshot.pipeline.workflow.graph import AgentBuilder, create_reconstruction_graph
from zeroshot.pipeline.workflow.reconstruction import (
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


def _drawing_submission(
    ticket_id: str | None = _ROUND_ZERO_TICKET,
) -> AIMessage:
    return _message(
        DrawingSubmission(
            responses=_responses(ticket_id, PipelineStage.DRAWINGS),
        )
    )


def _invalid_drawing_submission() -> AIMessage:
    return _message(
        DrawingSubmission(
            responses=_responses("ticket_absent", PipelineStage.DRAWINGS),
        )
    )


def _drawing_candidate(artifact: DrawingSource | None = None) -> DrawingSource:
    """A complete reading that retains the graph fixture's handed sheet."""
    artifact = artifact or drawing()
    handed = unread_sheet("sheet_drawing", View.FULL_PAGE, "/work/drawing.dxf")
    views = [
        item.model_copy(
            update={
                "crop_of": CropOf(sheet="sheet_drawing", box=[0.0, 0.0, 10.0, 10.0]),
                "file": f"/work/{item.name}.dxf",
            }
        )
        for item in artifact.sheets
    ]
    return DrawingSource(sheets=[handed, *views])


def _write_drawing(
    artifact: DrawingSource | None = None, call_id: str = "draw"
) -> AIMessage:
    payload = base64.b64encode(
        (_drawing_candidate(artifact).model_dump_json(indent=2) + "\n").encode()
    ).decode()
    command = (
        'python -c "import base64;'
        "open('/work/drawing.json','wb').write(base64.b64decode('" + payload + "'))\""
    )
    return tool_call("run_shell", {"command": command}, call_id)


def _drawing_script(
    ticket_id: str | None = _ROUND_ZERO_TICKET,
    *,
    artifact: DrawingSource | None = None,
    call_id: str = "draw",
) -> tuple[AIMessage, AIMessage]:
    return _write_drawing(artifact, call_id), _drawing_submission(ticket_id)


def _semantic_submission(
    ticket_id: str | None = _ROUND_ZERO_TICKET,
    *features: str,
) -> AIMessage:
    return _message(
        SemanticSubmission(
            **replacing(hypothesis(*(features or ("a plate",)))),
            responses=_responses(ticket_id, PipelineStage.SEMANTICS),
        )
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
    *,
    builds: Sequence[int | str] = (1,),
    detail: str = "extrude",
) -> AIMessage:
    return _message(
        OperationSubmission(
            **replacing(_plan(builds=builds, detail=detail)),
            responses=_responses(ticket_id, PipelineStage.OPERATIONS),
        )
    )


def _coding_submission(ticket_id: str | None = _ROUND_ZERO_TICKET) -> AIMessage:
    return _message(
        CodingSubmission(
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


def _drawing_rejected_audit() -> AIMessage:
    return _message(
        AuditReport(
            accepted=False,
            findings=[
                AuditFinding(
                    name="find_wrong_edge",
                    observation="The front edge starts at the wrong coordinate.",
                    evidence=["sheet_front", "ev_front_line.start"],
                    backtrace=[],
                    revision_request=RevisionRequest(
                        action="modify",
                        targets=[
                            StageOutputRef(
                                stage=PipelineStage.DRAWINGS,
                                name="sheet_front",
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
    head: ScriptedChatModel,
    drawer: ScriptedChatModel | None = None,
    planner: ScriptedChatModel,
    coder: ScriptedChatModel,
    auditor: ScriptedChatModel,
    history_filename: str = "reconstruction.json",
    **overrides: Any,
):
    dxf_path = workdir.host_bind_dir / "drawing.dxf"
    dxf_path.write_text("0\nSECTION\n0\nEOF\n", encoding="utf-8")
    (workdir.host_bind_dir / "sheet_front.dxf").write_text(
        "0\nSECTION\n0\nEOF\n", encoding="utf-8"
    )
    common = {
        "announce_turns": False,
        "model_retries": 0,
        "checkpointer": False,
    }
    return create_reconstruction_graph(
        drawings_agent_builder=_agent(
            "drawing_analyzer",
            drawer or ScriptedChatModel(responses=_drawing_script()),
            max_turns=5,
            **common,
        ),
        semantics_agent_builder=_agent(
            "semantic_hypothesizer", head, max_turns=5, **common
        ),
        operations_agent_builder=_agent(
            "operation_planner", planner, max_turns=5, **common
        ),
        coding_agent_builder=_agent("coder", coder, max_turns=5, **common),
        audit_agent_builder=_agent("output_auditor", auditor, max_turns=5, **common),
        sandbox_runner=SandboxRunner(
            python_executable=Path(sys.executable),
            default_timeout_s=10,
        ),
        sandbox_workdir=workdir,
        artifact_presenter=_artifact_presenter(),
        input_manifest=InputManifest(
            sample_id="test",
            drawing=DrawingSource(
                sheets=[unread_sheet("sheet_drawing", View.FULL_PAGE, dxf_path)]
            ),
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


def _drawing_seed() -> ReconstructionRun:
    return advance_reconstruction(
        start_reconstruction("run_test", "Reconstruct the drawing.", drawing()),
        DrawingSubmission(
            responses=[_response(_ROUND_ZERO_TICKET, PipelineStage.DRAWINGS)],
        ),
        workspace_output=drawing(),
    )


def _semantics_seed() -> ReconstructionRun:
    return advance_reconstruction(
        _drawing_seed(),
        SemanticSubmission(
            **replacing(hypothesis("a plate")),
            responses=[_response(_ROUND_ZERO_TICKET, PipelineStage.SEMANTICS)],
        ),
    )


def _operations_resume() -> ReconstructionRun:
    return advance_reconstruction(
        _semantics_seed(),
        OperationSubmission(
            **replacing(_plan()),
            responses=[_response(_ROUND_ZERO_TICKET, PipelineStage.OPERATIONS)],
        ),
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
    semantic = _semantic_submission()
    head = ScriptedChatModel(responses=(semantic,))
    planner = ScriptedChatModel(responses=(_operation_submission(),))
    coder = ScriptedChatModel(responses=(_coding_submission(),))
    auditor = ScriptedChatModel(responses=(_accepted_audit(),))

    with SandboxWorkdir() as workdir:
        result = _graph(
            workdir,
            head=head,
            planner=planner,
            coder=coder,
            auditor=auditor,
        ).invoke({})
        persisted = ReconstructionRun.model_validate_json(
            (workdir.host_bind_dir / "reconstruction.json").read_text(encoding="utf-8")
        )
        working_drawing = DrawingSource.model_validate_json(
            (workdir.host_bind_dir / "drawing.json").read_text(encoding="utf-8")
        )
        attempted_drawing = DrawingSource.model_validate_json(
            (
                workdir.host_bind_dir
                / "attempts"
                / "round_000"
                / "drawing"
                / "000"
                / "drawing.json"
            ).read_text(encoding="utf-8")
        )

    assert calls == ["verify"]
    assert persisted == result["reconstruction"]
    snapshot = persisted.snapshots[0]
    assert snapshot.last_completed_stage is PipelineStage.CODING
    assert snapshot.semantics == hypothesis("a plate")
    assert snapshot.operations == _plan()
    assert snapshot.drawings == working_drawing == attempted_drawing
    assert snapshot.program_source == _PROGRAM
    assert [response.stage for response in snapshot.open_tickets[0].responses] == [
        PipelineStage.DRAWINGS,
        PipelineStage.SEMANTICS,
        PipelineStage.OPERATIONS,
        PipelineStage.CODING,
    ]
    assert result["audit_report"].accepted is True
    assert result["stage_submission"] is None
    assert result["stage_validation_error"] is None
    assert "within 5 turns" in auditor.received_messages[0][0].text
    audit_instruction = _last_instruction(auditor.received_messages[0])
    assert "/work/attempts/round_000/coding/000" in audit_instruction
    assert "/work/attempts/round_000/drawing/000" in audit_instruction
    assert "Addressed ticket_initial in coding." in audit_instruction
    returns_dir = "/work/attempts/round_000/coding/000/intermediate_returns"
    assert (returns_dir in audit_instruction) is has_returns
    assert ("Recorded directory: unavailable" in audit_instruction) is not has_returns
    assert "what the plan meant it to" in audit_instruction


def test_a_semantics_seed_starts_at_operations_without_calling_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _stub_verification(monkeypatch, _verified())
    head = ScriptedChatModel(responses=())
    planner = ScriptedChatModel(responses=(_operation_submission(),))
    coder = ScriptedChatModel(responses=(_coding_submission(),))
    auditor = ScriptedChatModel(responses=(_accepted_audit(),))

    with SandboxWorkdir() as workdir:
        result = _graph(
            workdir,
            head=head,
            planner=planner,
            coder=coder,
            auditor=auditor,
        ).invoke({"reconstruction": _semantics_seed()})
        persisted = ReconstructionRun.model_validate_json(
            (workdir.host_bind_dir / "reconstruction.json").read_text(encoding="utf-8")
        )

    assert head.received_messages == []
    assert len(planner.received_messages) == 1
    assert calls == ["verify"]
    assert persisted == result["reconstruction"]
    assert persisted.snapshots[-1].semantics == hypothesis("a plate")
    assert persisted.snapshots[-1].last_completed_stage is PipelineStage.CODING


def test_an_operations_checkpoint_resumes_at_coding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _stub_verification(monkeypatch, _verified())
    head = ScriptedChatModel(responses=())
    planner = ScriptedChatModel(responses=())
    coder = ScriptedChatModel(responses=(_coding_submission(),))
    auditor = ScriptedChatModel(responses=(_accepted_audit(),))

    with SandboxWorkdir() as workdir:
        result = _graph(
            workdir,
            head=head,
            planner=planner,
            coder=coder,
            auditor=auditor,
        ).invoke({"reconstruction": _operations_resume()})

    assert head.received_messages == []
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
    semantic = _semantic_submission()
    head = ScriptedChatModel(responses=(semantic,))
    planner = ScriptedChatModel(responses=(_operation_submission(),))
    coder = ScriptedChatModel(responses=(_coding_submission(),))
    auditor = ScriptedChatModel(responses=(_accepted_audit(),))

    with SandboxWorkdir() as workdir:
        _graph(
            workdir,
            head=head,
            planner=planner,
            coder=coder,
            auditor=auditor,
            history_filename="history.json",
        ).invoke({})
        assert (workdir.host_bind_dir / "history.json").is_file()

    for model in (head, planner, coder, auditor):
        prompt = "\n".join(message.text for message in model.received_messages[0])
        assert "/work/history.json" in prompt
        assert "round 0" in prompt


def test_only_the_drawing_stage_receives_the_scale_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_verification(monkeypatch, _verified())
    drawer = ScriptedChatModel(responses=_drawing_script())
    head = ScriptedChatModel(responses=(_semantic_submission(),))
    planner = ScriptedChatModel(responses=(_operation_submission(),))
    coder = ScriptedChatModel(responses=(_coding_submission(),))
    auditor = ScriptedChatModel(responses=(_accepted_audit(),))

    with SandboxWorkdir() as workdir:
        _graph(
            workdir,
            drawer=drawer,
            head=head,
            planner=planner,
            coder=coder,
            auditor=auditor,
        ).invoke({})

    assert drawer.bound_tool_names == (
        "run_shell",
        "load_image",
        "calculate_drawing_scale",
    )
    for model in (head, planner, coder, auditor):
        assert model.bound_tool_names == ("run_shell", "load_image")


def test_invalid_operations_retry_without_reaching_coding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _stub_verification(monkeypatch, _verified())
    semantic = _semantic_submission(_ROUND_ZERO_TICKET, "a plate")
    head = ScriptedChatModel(responses=(semantic,))
    planner = ScriptedChatModel(
        responses=(
            _operation_submission(builds=("sem_absent",)),
            _operation_submission(),
        )
    )
    coder = ScriptedChatModel(responses=(_coding_submission(),))

    with SandboxWorkdir() as workdir:
        result = _graph(
            workdir,
            head=head,
            planner=planner,
            coder=coder,
            auditor=ScriptedChatModel(responses=(_accepted_audit(),)),
            max_stage_validation_retries=1,
        ).invoke({})

    assert len(planner.received_messages) == 2
    assert len(coder.received_messages) == 1
    assert calls == ["verify"]
    retry = _last_instruction(planner.received_messages[1])
    assert "Operations Validation Error" in retry
    assert "sem_feature_1" in retry
    validation_message = next(
        message
        for message in planner.received_messages[1]
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
def test_invalid_drawings_retry_or_exhaust_before_semantics(
    monkeypatch: pytest.MonkeyPatch,
    recovers: bool,
) -> None:
    _stub_verification(monkeypatch, *([_verified()] if recovers else []))
    drawer = ScriptedChatModel(
        responses=(
            _write_drawing(),
            _invalid_drawing_submission(),
            _drawing_submission() if recovers else _invalid_drawing_submission(),
        )
    )
    head = ScriptedChatModel(responses=(_semantic_submission(),) if recovers else ())
    planner = ScriptedChatModel(
        responses=(_operation_submission(),) if recovers else ()
    )
    coder = ScriptedChatModel(responses=(_coding_submission(),) if recovers else ())
    auditor = ScriptedChatModel(responses=(_accepted_audit(),) if recovers else ())

    with SandboxWorkdir() as workdir:
        result = _graph(
            workdir,
            drawer=drawer,
            head=head,
            planner=planner,
            coder=coder,
            auditor=auditor,
            max_stage_validation_retries=1,
        ).invoke({})

    assert len(drawer.received_messages) == 3
    retry = _last_instruction(drawer.received_messages[2])
    assert "Drawings Validation Error" in retry
    assert "ticket_absent" in retry
    snapshot = result["reconstruction"].snapshots[0]
    if recovers:
        assert snapshot.last_completed_stage is PipelineStage.CODING
        assert len(head.received_messages) == 1
        assert result["stage_validation_error"] is None
    else:
        assert snapshot.last_completed_stage is None
        assert head.received_messages == []
        assert result["stage_validation_failure_count"] == 2
        assert "ticket_absent" in result["stage_validation_error"]


def test_stage_validation_retry_limit_stops_before_downstream_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _stub_verification(monkeypatch)
    semantic = _semantic_submission()
    planner = ScriptedChatModel(
        responses=tuple(_operation_submission(builds=("sem_absent",)) for _ in range(2))
    )
    coder = ScriptedChatModel(responses=())
    auditor = ScriptedChatModel(responses=())

    with SandboxWorkdir() as workdir:
        result = _graph(
            workdir,
            head=ScriptedChatModel(responses=(semantic,)),
            planner=planner,
            coder=coder,
            auditor=auditor,
            max_stage_validation_retries=1,
        ).invoke({})

    assert len(planner.received_messages) == 2
    assert coder.received_messages == []
    assert auditor.received_messages == []
    assert calls == []
    snapshot = result["reconstruction"].snapshots[0]
    assert snapshot.last_completed_stage is PipelineStage.SEMANTICS
    assert snapshot.operations is None
    assert result["stage_validation_failure_count"] == 2
    assert "sem_absent" in result["stage_validation_error"]


def test_a_persisted_drawing_checkpoint_can_restart_the_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_verification(monkeypatch, _verified())

    with SandboxWorkdir() as workdir:
        stopped = _graph(
            workdir,
            drawer=ScriptedChatModel(
                responses=(
                    _write_drawing(),
                    _invalid_drawing_submission(),
                    _invalid_drawing_submission(),
                )
            ),
            head=ScriptedChatModel(responses=()),
            planner=ScriptedChatModel(responses=()),
            coder=ScriptedChatModel(responses=()),
            auditor=ScriptedChatModel(responses=()),
            max_stage_validation_retries=1,
        ).invoke({})
        history_path = workdir.host_bind_dir / "reconstruction.json"
        checkpoint = ReconstructionRun.model_validate_json(
            history_path.read_text(encoding="utf-8")
        )
        resumed = _graph(
            workdir,
            drawer=ScriptedChatModel(responses=_drawing_script(call_id="resume_draw")),
            head=ScriptedChatModel(responses=(_semantic_submission(),)),
            planner=ScriptedChatModel(responses=(_operation_submission(),)),
            coder=ScriptedChatModel(responses=(_coding_submission(),)),
            auditor=ScriptedChatModel(responses=(_accepted_audit(),)),
        ).invoke({"reconstruction": checkpoint})
        persisted = ReconstructionRun.model_validate_json(
            history_path.read_text(encoding="utf-8")
        )

    assert stopped["reconstruction"] == checkpoint
    assert checkpoint.snapshots[0].last_completed_stage is None
    assert resumed["reconstruction"] == persisted
    assert persisted.run_id == checkpoint.run_id
    assert persisted.snapshots[0].last_completed_stage is PipelineStage.CODING


def test_an_invalid_audit_is_retried_against_the_same_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_verification(monkeypatch, _verified())
    semantic = _semantic_submission()
    auditor = ScriptedChatModel(responses=(_invalid_audit(), _accepted_audit()))

    with SandboxWorkdir() as workdir:
        result = _graph(
            workdir,
            head=ScriptedChatModel(responses=(semantic,)),
            planner=ScriptedChatModel(responses=(_operation_submission(),)),
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


def test_a_rejected_audit_opens_a_fresh_round_for_all_reasoning_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _stub_verification(monkeypatch, _verified("000"), _verified("001"))
    first_semantics = _semantic_submission()
    second_semantics = _semantic_submission(
        _ROUND_ONE_TICKET,
        "a revised plate",
    )
    head = ScriptedChatModel(responses=(first_semantics, second_semantics))
    planner = ScriptedChatModel(
        responses=(
            _operation_submission(),
            _operation_submission(_ROUND_ONE_TICKET, detail="extrude revised plate"),
        )
    )
    coder = ScriptedChatModel(
        responses=(
            _coding_submission(),
            _coding_submission(_ROUND_ONE_TICKET),
        )
    )
    auditor = ScriptedChatModel(
        responses=(
            _rejected_audit(
                StageOutputRef(
                    stage=PipelineStage.SEMANTICS,
                    name="sem_feature_1",
                )
            ),
        )
    )

    with SandboxWorkdir() as workdir:
        result = _graph(
            workdir,
            head=head,
            planner=planner,
            coder=coder,
            auditor=auditor,
            max_audit_reject_count=1,
        ).invoke({})

    run = result["reconstruction"]
    assert calls == ["verify", "verify"]
    assert len(run.snapshots) == 2
    first, second = run.snapshots
    assert first.open_tickets[0].ticket_id == _ROUND_ZERO_TICKET
    assert second.open_tickets[0].ticket_id == _ROUND_ONE_TICKET
    assert first.semantics == hypothesis("a plate")
    assert second.semantics == hypothesis("a revised plate")
    assert first.operations == _plan()
    assert second.operations == _plan(detail="extrude revised plate")
    assert all(len(ticket.responses) == 3 for ticket in second.open_tickets)
    assert len(head.received_messages) == 2
    assert len(planner.received_messages) == 2
    assert len(coder.received_messages) == 2
    assert len(auditor.received_messages) == 1
    assert "round 1" in _last_instruction(planner.received_messages[1])


def test_a_drawing_rooted_revision_refreshes_values_and_preserves_history(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    revision_ticket = "ticket_001_wrong_edge"
    old_drawing = drawing(
        sheets=[
            sheet(
                "front",
                evidence=[evidence(name="ev_front_line", start=[1.0, 0.0])],
            )
        ]
    )
    revised_drawing = drawing(
        sheets=[
            sheet(
                "front",
                evidence=[evidence(name="ev_front_line", start=[2.5, 0.0])],
            )
        ]
    )
    first_semantics = _semantic_submission()
    unchanged_semantics = _message(
        SemanticSubmission(
            edits=[],
            deleted=[],
            rationale=None,
            responses=_responses(revision_ticket, PipelineStage.SEMANTICS),
        )
    )
    first_operations = _operation_submission(detail="start at ev_front_line.start.x")
    unchanged_operations = _message(
        OperationSubmission(
            edits=[],
            deleted=[],
            rationale=None,
            responses=_responses(revision_ticket, PipelineStage.OPERATIONS),
        )
    )
    calls = _stub_verification(monkeypatch, _verified("000"), _verified("001"))

    with SandboxWorkdir(host_bind_dir=tmp_path) as workdir:
        result = _graph(
            workdir,
            drawer=ScriptedChatModel(
                responses=(
                    *_drawing_script(artifact=old_drawing, call_id="draw_old"),
                    *_drawing_script(
                        revision_ticket,
                        artifact=revised_drawing,
                        call_id="draw_revised",
                    ),
                )
            ),
            head=ScriptedChatModel(responses=(first_semantics, unchanged_semantics)),
            planner=ScriptedChatModel(
                responses=(first_operations, unchanged_operations)
            ),
            coder=ScriptedChatModel(
                responses=(
                    _coding_submission(),
                    _coding_submission(revision_ticket),
                )
            ),
            auditor=ScriptedChatModel(responses=(_drawing_rejected_audit(),)),
            max_audit_reject_count=1,
        ).invoke({})
        persisted = ReconstructionRun.model_validate_json(
            (workdir.host_bind_dir / "reconstruction.json").read_text(encoding="utf-8")
        )

    assert calls == ["verify", "verify"]
    assert persisted == result["reconstruction"]
    first, second = persisted.snapshots
    assert first.drawings is not None and second.drawings is not None
    assert first.drawings.evidence()[0].parameters[0].values == [1.0, 0.0]
    assert second.drawings.evidence()[0].parameters[0].values == [2.5, 0.0]
    assert first.operations is not None and second.operations is not None
    assert first.operations.proposal[0].detail.endswith("(= 1.0)")
    assert second.operations.proposal[0].detail.endswith("(= 2.5)")
    assert [response.stage for response in second.open_tickets[0].responses] == [
        PipelineStage.DRAWINGS,
        PipelineStage.SEMANTICS,
        PipelineStage.OPERATIONS,
        PipelineStage.CODING,
    ]
    assert (tmp_path / "attempts" / "round_000" / "drawing" / "000").is_dir()
    assert (tmp_path / "attempts" / "round_001" / "drawing" / "000").is_dir()


def test_a_coding_rooted_finding_reopens_the_round_for_coding_alone(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_verification(monkeypatch, _verified("000"), _verified("001"))
    first_semantics = _semantic_submission()
    head = ScriptedChatModel(responses=(first_semantics,))
    planner = ScriptedChatModel(responses=(_operation_submission(),))
    coder = ScriptedChatModel(
        responses=(_coding_submission(), _coding_submission(_ROUND_ONE_TICKET))
    )
    auditor = ScriptedChatModel(responses=(_rejected_audit(),))

    with SandboxWorkdir(host_bind_dir=tmp_path) as workdir:
        result = _graph(
            workdir,
            head=head,
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
    assert len(head.received_messages) == 1
    assert len(planner.received_messages) == 1
    assert len(coder.received_messages) == 2
    assert second.semantics == first.semantics
    assert second.operations == first.operations
    assert second.drawings == first.drawings
    assert not (tmp_path / "attempts" / "round_001" / "drawing").exists()


def test_rejection_at_the_round_limit_finishes_without_opening_another_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_verification(monkeypatch, _verified())
    semantic = _semantic_submission()

    auditor = ScriptedChatModel(responses=(_rejected_audit(),))
    with SandboxWorkdir() as workdir:
        result = _graph(
            workdir,
            head=ScriptedChatModel(responses=(semantic,)),
            planner=ScriptedChatModel(responses=(_operation_submission(),)),
            coder=ScriptedChatModel(responses=(_coding_submission(),)),
            auditor=auditor,
            max_audit_reject_count=0,
            share_thread=True,
        ).invoke({})

    assert len(result["reconstruction"].snapshots) == 1
    assert auditor.received_messages == []
    assert result["audit_report"] is None
    assert result["stage_validation_error"] is None
