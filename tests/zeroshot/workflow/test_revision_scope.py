"""A revision changes only what its tickets cover, or says why."""

import pytest

from tests.zeroshot.contracts import interpretation
from tests.zeroshot.workflow.test_reconstruction_workflow import (
    _completed_run,
    _ref,
    _report,
    _stage_responses,
)
from zeroshot.pipeline.stages.contracts import ReconstructionHistory
from zeroshot.pipeline.stages.interpretation.contracts import DrawingInterpretation
from zeroshot.pipeline.stages.tickets.contracts import StageReport, TicketAnswers
from zeroshot.pipeline.stages.validate import SubmissionValidationError
from zeroshot.pipeline.workflow.lifecycle import advance_reconstruction, open_next_round


def _revision(stage: str, name: str | None) -> ReconstructionHistory:
    return open_next_round(_completed_run(), _report(target=_ref(stage, name)))


def _answer(
    history: ReconstructionHistory,
    artifact: DrawingInterpretation,
    **report: object,
) -> ReconstructionHistory:
    return advance_reconstruction(
        history,
        TicketAnswers(
            responses=_stage_responses(history, "interpretation"),
            stage_report=StageReport.model_validate(
                {
                    "concerns": {},
                    "dimension_checks": None,
                    "unticketed_changes": {},
                    **report,
                }
            ),
        ),
        workspace_output=artifact,
    )


def test_a_feature_changed_outside_the_tickets_needs_a_reason() -> None:
    run = _revision("interpretation", "sem_feature_1")
    wider = interpretation("the base", "a wider hole")

    with pytest.raises(SubmissionValidationError, match=r"sem_feature_2 \(changed\)"):
        _answer(run, wider)
    _answer(run, wider, unticketed_changes={"sem_feature_2": "The hole is wider."})


def test_a_reason_for_a_member_that_did_not_change_is_refused() -> None:
    run = _revision("interpretation", "sem_feature_1")

    with pytest.raises(
        SubmissionValidationError, match="did not change: sem_feature_2"
    ):
        _answer(
            run,
            interpretation("the base", "the hole"),
            unticketed_changes={"sem_feature_2": "."},
        )


def test_a_feature_that_cites_the_ticketed_view_may_change() -> None:
    run = _revision("interpretation", "view_front")

    _answer(run, interpretation("a thicker base", "the hole"))


def test_a_whole_stage_ticket_opens_that_stage() -> None:
    run = _revision("interpretation", None)

    _answer(run, interpretation("a thicker base", "a wider hole"))


def test_fields_validation_derives_are_not_changes() -> None:
    run = _revision("interpretation", "sem_feature_1")
    calibrated = interpretation("the base", "the hole")
    calibrated.views[0].scale = 0.5
    calibrated.views[0].image_size = (10, 10)
    _answer(run, calibrated)

    located = interpretation("the base", "the hole")
    evidence = located.features[1].evidence[0]
    located.features[1].evidence[0] = evidence.model_copy(
        update={"box_uv": (0.0, 0.0, 5.0, 5.0)}
    )
    _answer(run, located)

    located.features[1].evidence[0] = evidence.model_copy(
        update={"box_px": (1, 1, 9, 9)}
    )
    with pytest.raises(SubmissionValidationError, match=r"sem_feature_2 \(changed\)"):
        _answer(run, located)
