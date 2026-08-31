"""C4 workflow wiring: submissions, integration, retries, and audit rounds."""

import sys
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.messages.content import ContentBlock

from tests.zeroshot.chat_models import ScriptedChatModel
from tests.zeroshot.contracts import hypothesis, replacing, unchanged
from zeroshot.pipeline.messages import ArtifactPresenter, InputManifest
from zeroshot.pipeline.messages.contracts import (
    Operation,
    OperationPlan,
    OperationVerb,
    PipelineStage,
)
from zeroshot.pipeline.messages.contracts.audit import (
    AuditFinding,
    AuditReport,
    Backtrace,
    RevisionRequest,
    StageOutputRef,
)
from zeroshot.pipeline.messages.contracts.reconstruction import (
    CodingSubmission,
    OperationSubmission,
    ReconstructionRun,
    SemanticSubmission,
    TicketResponse,
)
from zeroshot.pipeline.sandbox import SandboxRunner, SandboxWorkdir
from zeroshot.pipeline.verification import (
    ExecutionStatus,
    StepRenderer,
    VerifyOutputResult,
)
from zeroshot.pipeline.workflow import create_agent, create_fanout_reduce_graph
from zeroshot.pipeline.workflow import graph as graph_module
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
            **unchanged(),
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
                    name="finding_missing_hole",
                    observation="The drawing contains a hole that the model omits.",
                    evidence=["render_3d/hlg_front.png"],
                    backtraces=[
                        Backtrace(
                            hops=[],
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
            ],
        )
    )


def _invalid_audit() -> AIMessage:
    return _message(
        AuditReport(
            accepted=False,
            findings=[
                AuditFinding(
                    name="finding_unknown_operation",
                    observation="The model is incorrect.",
                    evidence=["verification.status"],
                    backtraces=[
                        Backtrace(
                            hops=[],
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
            ],
        )
    )


def _agent(role: str, model: BaseChatModel, **overrides: Any) -> AgentBuilder:
    return partial(create_agent, role=role, model=model, **overrides)


def _artifact_presenter() -> ArtifactPresenter:
    return ArtifactPresenter(
        input_render3d_mode="none",
        input_render3d_styles=(),
        feedback_render3d_mode="none",
        feedback_render3d_styles=(),
    )


def _graph(
    workdir: SandboxWorkdir,
    *,
    head: ScriptedChatModel,
    peer: ScriptedChatModel,
    planner: ScriptedChatModel,
    coder: ScriptedChatModel,
    auditor: ScriptedChatModel,
    history_filename: str = "reconstruction.json",
    **overrides: Any,
):
    dxf_path = workdir.host_bind_dir / "drawing.dxf"
    dxf_path.write_text("0\nSECTION\n0\nEOF\n", encoding="utf-8")
    common = {
        "announce_turns": False,
        "model_retries": 0,
        "checkpointer": False,
    }
    return create_reconstruction_graph(
        semantics_agent_builder=partial(
            create_fanout_reduce_graph,
            proposer_role="semantic_hypothesizer",
            proposer_models=[head, peer],
            max_proposer_turns=5,
            max_reducer_turns=5,
            **common,
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
        renderer=StepRenderer(timeout_s=60.0),
        artifact_presenter=_artifact_presenter(),
        input_manifest=InputManifest(
            sample_id="test",
            dxf_path=dxf_path,
            render3d_paths={},
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

        def verify(self) -> tuple[VerifyOutputResult, None]:
            calls.append("verify")
            return remaining.pop(0), None

        def feedback(self) -> list[ContentBlock]:
            return []

    monkeypatch.setattr(
        graph_module,
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


def _semantics_seed() -> ReconstructionRun:
    return advance_reconstruction(
        start_reconstruction("run_test", "Reconstruct the drawing."),
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


def test_an_accepted_round_is_integrated_and_persisted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _stub_verification(monkeypatch, _verified())
    semantic = _semantic_submission()
    head = ScriptedChatModel(responses=(semantic, semantic))
    peer = ScriptedChatModel(responses=(semantic,))
    planner = ScriptedChatModel(responses=(_operation_submission(),))
    coder = ScriptedChatModel(responses=(_coding_submission(),))
    auditor = ScriptedChatModel(responses=(_accepted_audit(),))

    with SandboxWorkdir() as workdir:
        result = _graph(
            workdir,
            head=head,
            peer=peer,
            planner=planner,
            coder=coder,
            auditor=auditor,
        ).invoke({})
        persisted = ReconstructionRun.model_validate_json(
            (workdir.host_bind_dir / "reconstruction.json").read_text(encoding="utf-8")
        )

    assert calls == ["verify"]
    assert persisted == result["reconstruction"]
    snapshot = persisted.snapshots[0]
    assert snapshot.last_completed_stage is PipelineStage.CODING
    assert snapshot.semantics == hypothesis("a plate")
    assert snapshot.operations == _plan()
    assert snapshot.program_source == _PROGRAM
    assert [response.stage for response in snapshot.open_tickets[0].responses] == [
        PipelineStage.SEMANTICS,
        PipelineStage.OPERATIONS,
        PipelineStage.CODING,
    ]
    assert result["audit_report"].accepted is True
    assert result["stage_submission"] is None
    assert result["stage_validation_error"] is None


def test_a_semantics_seed_starts_at_operations_without_calling_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _stub_verification(monkeypatch, _verified())
    head = ScriptedChatModel(responses=())
    peer = ScriptedChatModel(responses=())
    planner = ScriptedChatModel(responses=(_operation_submission(),))
    coder = ScriptedChatModel(responses=(_coding_submission(),))
    auditor = ScriptedChatModel(responses=(_accepted_audit(),))

    with SandboxWorkdir() as workdir:
        result = _graph(
            workdir,
            head=head,
            peer=peer,
            planner=planner,
            coder=coder,
            auditor=auditor,
        ).invoke({"reconstruction": _semantics_seed()})
        persisted = ReconstructionRun.model_validate_json(
            (workdir.host_bind_dir / "reconstruction.json").read_text(encoding="utf-8")
        )

    assert head.received_messages == []
    assert peer.received_messages == []
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
    peer = ScriptedChatModel(responses=())
    planner = ScriptedChatModel(responses=())
    coder = ScriptedChatModel(responses=(_coding_submission(),))
    auditor = ScriptedChatModel(responses=(_accepted_audit(),))

    with SandboxWorkdir() as workdir:
        result = _graph(
            workdir,
            head=head,
            peer=peer,
            planner=planner,
            coder=coder,
            auditor=auditor,
        ).invoke({"reconstruction": _operations_resume()})

    assert head.received_messages == []
    assert peer.received_messages == []
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
    head = ScriptedChatModel(responses=(semantic, semantic))
    peer = ScriptedChatModel(responses=(semantic,))
    planner = ScriptedChatModel(responses=(_operation_submission(),))
    coder = ScriptedChatModel(responses=(_coding_submission(),))
    auditor = ScriptedChatModel(responses=(_accepted_audit(),))

    with SandboxWorkdir() as workdir:
        _graph(
            workdir,
            head=head,
            peer=peer,
            planner=planner,
            coder=coder,
            auditor=auditor,
            history_filename="history.json",
        ).invoke({})
        assert (workdir.host_bind_dir / "history.json").is_file()

    for model in (head, peer, planner, coder, auditor):
        prompt = "\n".join(message.text for message in model.received_messages[0])
        assert "/work/history.json" in prompt
        assert "round 0" in prompt


def test_invalid_operations_retry_without_reaching_coding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _stub_verification(monkeypatch, _verified())
    semantic = _semantic_submission(_ROUND_ZERO_TICKET, "a plate")
    head = ScriptedChatModel(responses=(semantic, semantic))
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
            peer=ScriptedChatModel(responses=(semantic,)),
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
    assert result["reconstruction"].snapshots[0].operations == _plan()
    assert result["stage_validation_failure_count"] == 0


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
            head=ScriptedChatModel(responses=(semantic, semantic)),
            peer=ScriptedChatModel(responses=(semantic,)),
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


def test_an_invalid_audit_is_retried_against_the_same_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_verification(monkeypatch, _verified())
    semantic = _semantic_submission()
    auditor = ScriptedChatModel(responses=(_invalid_audit(), _accepted_audit()))

    with SandboxWorkdir() as workdir:
        result = _graph(
            workdir,
            head=ScriptedChatModel(responses=(semantic, semantic)),
            peer=ScriptedChatModel(responses=(semantic,)),
            planner=ScriptedChatModel(responses=(_operation_submission(),)),
            coder=ScriptedChatModel(responses=(_coding_submission(),)),
            auditor=auditor,
            max_stage_validation_retries=1,
        ).invoke({})

    assert len(auditor.received_messages) == 2
    assert "Audit Validation Error" in _last_instruction(auditor.received_messages[1])
    assert "op_missing" in _last_instruction(auditor.received_messages[1])
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
    head = ScriptedChatModel(
        responses=(first_semantics, first_semantics, second_semantics)
    )
    peer = ScriptedChatModel(responses=(first_semantics,))
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
            peer=peer,
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
    assert len(head.received_messages) == 3
    assert len(peer.received_messages) == 1
    assert len(planner.received_messages) == 2
    assert len(coder.received_messages) == 2
    assert len(auditor.received_messages) == 1
    assert "round 1" in _last_instruction(planner.received_messages[1])


def test_a_coding_rooted_finding_reopens_the_round_for_coding_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_verification(monkeypatch, _verified("000"), _verified("001"))
    first_semantics = _semantic_submission()
    head = ScriptedChatModel(responses=(first_semantics, first_semantics))
    peer = ScriptedChatModel(responses=(first_semantics,))
    planner = ScriptedChatModel(responses=(_operation_submission(),))
    coder = ScriptedChatModel(
        responses=(_coding_submission(), _coding_submission(_ROUND_ONE_TICKET))
    )
    auditor = ScriptedChatModel(responses=(_rejected_audit(),))

    with SandboxWorkdir() as workdir:
        result = _graph(
            workdir,
            head=head,
            peer=peer,
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
    assert len(head.received_messages) == 2
    assert len(planner.received_messages) == 1
    assert len(coder.received_messages) == 2
    assert second.semantics == first.semantics
    assert second.operations == first.operations


def test_rejection_at_the_round_limit_finishes_without_opening_another_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_verification(monkeypatch, _verified())
    semantic = _semantic_submission()

    auditor = ScriptedChatModel(responses=(_rejected_audit(),))
    with SandboxWorkdir() as workdir:
        result = _graph(
            workdir,
            head=ScriptedChatModel(responses=(semantic, semantic)),
            peer=ScriptedChatModel(responses=(semantic,)),
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
