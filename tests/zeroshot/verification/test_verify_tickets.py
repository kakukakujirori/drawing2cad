from tests.zeroshot.contracts import drawing
from tests.zeroshot.workflow.test_validate_submission import _answer_for, _snapshot
from zeroshot.pipeline.stages.contracts import ReconstructionHistory
from zeroshot.pipeline.stages.tickets.contracts import StageReport, TicketAnswers
from zeroshot.pipeline.stages.tickets.verify import TicketVerifier
from zeroshot.pipeline.stages.types import PipelineStage


def test_answers_that_fit_the_round_need_no_feedback() -> None:
    verifier = TicketVerifier()
    verifier.reset(_planned_run())
    answers = TicketAnswers(
        responses=_answer_for("ticket_initial", PipelineStage.CODING),
        stage_report=StageReport(concerns={}),
    )

    assert verifier.feedback(answers) == []


def test_every_contradiction_in_the_answers_is_explained_at_once() -> None:
    verifier = TicketVerifier()
    verifier.reset(_planned_run())
    copied = TicketAnswers(
        stage_report=StageReport(concerns={}),
        responses=_answer_for("ticket_absent", PipelineStage.CODING),
    )

    (block,) = verifier.feedback(copied)

    assert "missing ticket responses: ticket_initial" in block["text"]
    assert "unknown ticket responses: ticket_absent" in block["text"]


def _planned_run() -> ReconstructionHistory:
    return ReconstructionHistory(
        run_id="run_tickets",
        input_drawings=drawing(),
        snapshots=[_snapshot(PipelineStage.INTERPRETATION)],
    )
