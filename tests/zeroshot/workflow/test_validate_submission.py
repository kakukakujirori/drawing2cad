"""Contextual validation of coding's answer against its round."""

import pytest

from tests.zeroshot.contracts import drawing, interpretation, interpreted_feature
from zeroshot.pipeline.stages.coding.validate import CodingOutput
from zeroshot.pipeline.stages.coding.verify import VerifyOutputResult
from zeroshot.pipeline.stages.contracts import ReconstructionSnapshot
from zeroshot.pipeline.stages.interpretation.contracts import (
    DrawingInterpretation,
    Region,
)
from zeroshot.pipeline.stages.tickets.contracts import (
    BootstrapWork,
    StageReport,
    Ticket,
    TicketAnswers,
    TicketResponse,
)
from zeroshot.pipeline.stages.types import PipelineStage, ReasoningStage
from zeroshot.pipeline.stages.validate import (
    SubmissionValidationError,
    validate_submission,
)
from zeroshot.pipeline.verification import ExecutionStatus
from zeroshot.pipeline.verification.run_cadquery import CadQueryExecutionReport
from zeroshot.pipeline.workflow.lifecycle import (
    advance_reconstruction,
    load_reconstruction,
    save_reconstruction,
    start_reconstruction,
)


def _interpretation() -> DrawingInterpretation:
    return interpretation("the base")


def _answer_for(ticket_id: str, stage: ReasoningStage) -> dict[str, str]:
    """The submitted shape: answers keyed by ticket, with no stage to disagree."""
    return {ticket_id: f"Reviewed {ticket_id} during {stage}."}


def _ticket(ticket_id: str, *, answered: bool = False) -> Ticket:
    return Ticket(
        ticket_id=ticket_id,
        subject=BootstrapWork(instruction="Reconstruct the part."),
        responses=[
            TicketResponse(
                ticket_id=ticket_id,
                stage=PipelineStage.CODING,
                summary=f"Reviewed {ticket_id} during coding.",
            )
        ]
        if answered
        else [],
    )


def _snapshot(
    completed_stage: ReasoningStage | None,
    *,
    tickets: list[Ticket] | None = None,
    held: DrawingInterpretation | None = None,
) -> ReconstructionSnapshot:
    coded = completed_stage is PipelineStage.CODING
    verification = (
        VerifyOutputResult(
            exec_report=CadQueryExecutionReport(
                status=ExecutionStatus.VERIFIED,
                source="result = object()\n",
                returncode=0,
            )
        )
        if coded
        else None
    )
    return ReconstructionSnapshot(
        open_tickets=tickets or [_ticket("ticket_initial", answered=coded)],
        round=0,
        last_completed_stage=completed_stage,
        interpretation=(held or _interpretation()) if coded else None,
        program_source="result = object()\n" if coded else None,
        verification=verification,
    )


def _output(status: ExecutionStatus = ExecutionStatus.REJECTED) -> CodingOutput:
    return CodingOutput(
        _interpretation(),
        VerifyOutputResult(exec_report=CadQueryExecutionReport(status=status)),
    )


def _answers(responses: dict[str, str]) -> TicketAnswers:
    return TicketAnswers(stage_report=StageReport(concerns={}), responses=responses)


def test_coding_accepts_its_interpretation_and_terminal_build() -> None:
    validate_submission(
        _answers(_answer_for("ticket_initial", PipelineStage.CODING)),
        _snapshot(None),
        deliverable=_output(),
    )


@pytest.mark.parametrize(
    ("responses", "message"),
    [
        (_answer_for("ticket_one", PipelineStage.CODING), "missing.*ticket_two"),
        (
            _answer_for("ticket_one", PipelineStage.CODING)
            | _answer_for("ticket_two", PipelineStage.CODING)
            | _answer_for("ticket_unknown", PipelineStage.CODING),
            "unknown.*ticket_unknown",
        ),
        # GLM left the old field name behind as a key; the answer is a key
        # short, so the check that knows the open tickets names them.
        (
            _answer_for("ticket_one", PipelineStage.CODING)
            | _answer_for("ticket_two", PipelineStage.CODING)
            | {"summary": "a leftover field name"},
            r"unknown ticket responses: summary\. Open tickets: ticket_one, ticket_two",
        ),
    ],
)
def test_ticket_responses_must_cover_the_current_snapshot_exactly_once(
    responses: dict[str, str],
    message: str,
) -> None:
    """Keying by ticket leaves only membership to check."""
    snapshot = _snapshot(None, tickets=[_ticket("ticket_one"), _ticket("ticket_two")])

    with pytest.raises(SubmissionValidationError, match=message):
        validate_submission(_answers(responses), snapshot, deliverable=_output())


def test_interpretation_rejects_an_evidence_view_absent_from_the_artifact() -> None:
    from pydantic import ValidationError

    with pytest.raises(
        ValidationError, match=r"sem_bore.evidence\[0\].view: unknown view view_absent"
    ):
        interpretation(
            features=[
                interpreted_feature(
                    "sem_bore",
                    "bore",
                    evidence=[Region(view="view_absent", box_px=(0, 0, 1, 1))],
                )
            ]
        )


def test_coding_requires_both_files_and_a_finished_build() -> None:
    answers = _answers(_answer_for("ticket_initial", PipelineStage.CODING))

    with pytest.raises(SubmissionValidationError, match="verified interpretation"):
        validate_submission(answers, _snapshot(None))
    with pytest.raises(SubmissionValidationError, match="not complete"):
        validate_submission(
            answers,
            _snapshot(None),
            deliverable=CodingOutput(_interpretation(), VerifyOutputResult()),
        )
    with pytest.raises(SubmissionValidationError, match="must be terminal"):
        validate_submission(
            answers,
            _snapshot(None),
            deliverable=_output(ExecutionStatus.UNINITIALIZED),
        )


def test_completed_coding_accepts_only_an_audit_report() -> None:
    with pytest.raises(SubmissionValidationError, match="only an AuditReport"):
        validate_submission(
            _answers(_answer_for("ticket_initial", PipelineStage.CODING)),
            _snapshot(PipelineStage.CODING),
            deliverable=_output(),
        )


def test_stage_reports_commit_with_artifacts_and_responses_and_survive_resume(tmp_path):
    run = start_reconstruction("run_reports", "Read the drawing.", drawing())
    held = _interpretation()
    held.features[0].parameters["width"] = 12.0
    submission = TicketAnswers(
        responses={"ticket_initial": "Established sem_feature_1.width."},
        stage_report=StageReport(
            concerns={
                "concern_height": "The height uses sem_feature_1.width provisionally."
            },
        ),
    )
    output = CodingOutput(
        held,
        VerifyOutputResult(
            exec_report=CadQueryExecutionReport(status=ExecutionStatus.REJECTED)
        ),
    )
    with pytest.raises(SubmissionValidationError, match="verified interpretation"):
        advance_reconstruction(run, submission)
    with pytest.raises(SubmissionValidationError, match="missing ticket responses"):
        advance_reconstruction(run, _answers({}), workspace_output=output)
    assert run.snapshots[-1].stage_reports == {}

    coded = advance_reconstruction(run, submission, workspace_output=output)
    snapshot = coded.snapshots[-1]
    assert snapshot.interpretation == held
    report = snapshot.stage_reports[PipelineStage.CODING]
    assert "sem_feature_1.width (= 12.0)" in report.concerns["concern_height"]
    assert (
        "sem_feature_1.width (= 12.0)" in snapshot.open_tickets[0].responses[0].summary
    )
    assert set(report.model_dump()) == {"concerns"}
    assert run.snapshots[-1].stage_reports == {}
    history = tmp_path / "reconstruction.json"
    save_reconstruction(history, coded)
    assert load_reconstruction(history).snapshots[-1].stage_reports == {
        PipelineStage.CODING: report
    }
