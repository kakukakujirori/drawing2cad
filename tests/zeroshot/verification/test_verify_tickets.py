from tests.zeroshot.contracts import drawing, interpretation
from tests.zeroshot.workflow.test_reconstruction_workflow import _stage_responses
from tests.zeroshot.workflow.test_revision_scope import (
    _answer,
    _renamed_hole,
    _revision,
)
from tests.zeroshot.workflow.test_validate_submission import _response, _snapshot
from zeroshot.pipeline.stages.contracts import ReconstructionHistory
from zeroshot.pipeline.stages.tickets.contracts import StageReport, TicketAnswers
from zeroshot.pipeline.stages.tickets.verify import TicketVerifier
from zeroshot.pipeline.stages.types import PipelineStage


def test_answers_that_fit_the_round_need_no_feedback() -> None:
    verifier = TicketVerifier(lambda: None)
    verifier.reset(_planned_run())
    answers = TicketAnswers(
        responses=[_response("ticket_initial", PipelineStage.CODING)],
        stage_report=StageReport(dimension_checks={}),
    )

    assert verifier.feedback(answers) == []


def test_every_contradiction_in_the_answers_is_explained_at_once() -> None:
    verifier = TicketVerifier(lambda: None)
    verifier.reset(_planned_run())
    copied = TicketAnswers(
        responses=[_response("ticket_initial", PipelineStage.OPERATIONS)]
    )

    (block,) = verifier.feedback(copied)

    assert "ticket responses must belong to coding: ticket_initial" in block["text"]
    assert "coding requires dimension_checks" in block["text"]


def test_changes_outside_the_tickets_are_explained_against_the_artifact() -> None:
    run = _revision("interpretation", "sem_feature_1")
    run = _answer(run, "interpretation", interpretation("the base", "the hole"))
    verifier = TicketVerifier(_renamed_hole)
    verifier.reset(run)

    (block,) = verifier.feedback(
        TicketAnswers(responses=_stage_responses(run, "operations"))
    )

    assert "op_bore (added), op_hole (removed)" in block["text"]


def _planned_run() -> ReconstructionHistory:
    return ReconstructionHistory(
        run_id="run_tickets",
        input_drawings=drawing(),
        snapshots=[_snapshot(PipelineStage.OPERATIONS)],
    )
