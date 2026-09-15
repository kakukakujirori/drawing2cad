from tests.zeroshot.workflow.test_validate_submission import _response, _snapshot
from zeroshot.pipeline.messages.tickets import StageReport, TicketAnswers
from zeroshot.pipeline.stages.types import PipelineStage
from zeroshot.pipeline.verification.verify_tickets import TicketVerifier


def test_answers_that_fit_the_round_need_no_feedback() -> None:
    verifier = TicketVerifier()
    verifier.reset(_snapshot(PipelineStage.OPERATIONS))
    answers = TicketAnswers(
        responses=[_response("ticket_initial", PipelineStage.CODING)],
        stage_report=StageReport(dimension_checks={}),
    )

    assert verifier.feedback(answers) == []


def test_every_contradiction_in_the_answers_is_explained_at_once() -> None:
    verifier = TicketVerifier()
    verifier.reset(_snapshot(PipelineStage.OPERATIONS))
    copied = TicketAnswers(
        responses=[_response("ticket_initial", PipelineStage.OPERATIONS)]
    )

    (block,) = verifier.feedback(copied)

    assert "ticket responses must belong to coding: ticket_initial" in block["text"]
    assert "coding requires dimension_checks" in block["text"]
