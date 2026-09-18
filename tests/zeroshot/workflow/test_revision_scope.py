"""A revision changes only what its tickets cover, or says why."""

import pytest

from tests.zeroshot.contracts import interpretation
from tests.zeroshot.workflow.test_reconstruction_workflow import (
    _SOURCE,
    _completed_run,
    _operations,
    _ref,
    _report,
    _stage_responses,
)
from zeroshot.pipeline.stages.coding.verify import VerifyOutputResult
from zeroshot.pipeline.stages.contracts import ReconstructionHistory
from zeroshot.pipeline.stages.operations.contracts import OperationPlan
from zeroshot.pipeline.stages.tickets.contracts import StageReport, TicketAnswers
from zeroshot.pipeline.stages.tickets.validate import StageArtifact
from zeroshot.pipeline.stages.validate import SubmissionValidationError
from zeroshot.pipeline.verification import ExecutionStatus
from zeroshot.pipeline.workflow.lifecycle import advance_reconstruction, open_next_round


def _revision(stage: str, name: str | None) -> ReconstructionHistory:
    return open_next_round(_completed_run(), _report(target=_ref(stage, name)))


def _answer(
    history: ReconstructionHistory,
    stage: str,
    artifact: StageArtifact | VerifyOutputResult,
    **report: object,
) -> ReconstructionHistory:
    return advance_reconstruction(
        history,
        TicketAnswers(
            responses=_stage_responses(history, stage),
            stage_report=StageReport.model_validate(report),
        ),
        workspace_output=artifact,  # type: ignore[arg-type]
    )


def _renamed_hole() -> OperationPlan:
    plan = _operations()
    plan.proposal[1].name = "op_bore"
    return plan


def test_an_operation_renamed_outside_the_tickets_needs_a_reason() -> None:
    run = _revision("interpretation", "sem_feature_1")
    run = _answer(run, "interpretation", interpretation("the base", "the hole"))

    with pytest.raises(
        SubmissionValidationError, match=r"op_bore \(added\), op_hole \(removed\)"
    ):
        _answer(run, "operations", _renamed_hole())
    _answer(
        run,
        "operations",
        _renamed_hole(),
        unticketed_changes={"op_hole": "Renamed to op_bore.", "op_bore": "Renamed."},
    )


def test_reordering_operations_outside_the_tickets_needs_a_reason() -> None:
    run = _revision("interpretation", "sem_feature_1")
    run = _answer(run, "interpretation", interpretation("the base", "the hole"))
    reordered = _operations()
    reordered.proposal.reverse()

    with pytest.raises(SubmissionValidationError, match=r"op_hole \(changed\)"):
        _answer(run, "operations", reordered)
    _answer(run, "operations", reordered, unticketed_changes={"op_hole": "Moved."})


def test_a_reason_for_a_member_that_did_not_change_is_refused() -> None:
    run = _revision("interpretation", "sem_feature_1")
    run = _answer(run, "interpretation", interpretation("the base", "the hole"))

    with pytest.raises(SubmissionValidationError, match="did not change: op_base"):
        _answer(run, "operations", _operations(), unticketed_changes={"op_base": "."})


def test_an_operation_follows_a_feature_changed_this_round() -> None:
    run = _revision("interpretation", "sem_feature_1")
    run = _answer(
        run,
        "interpretation",
        interpretation("the base", "a wider hole"),
        unticketed_changes={"sem_feature_2": "The drawing shows a wider hole."},
    )
    plan = _operations()
    plan.proposal[1].detail = "Cut the wider hole through the base."

    _answer(run, "operations", plan)


def test_an_operation_follows_a_feature_that_cites_the_ticketed_view() -> None:
    run = _revision("interpretation", "view_front")
    run = _answer(run, "interpretation", interpretation("the base", "the hole"))
    plan = _operations()
    plan.proposal[0].detail = "Extrude a thicker base."

    _answer(run, "operations", plan)


def test_a_whole_stage_ticket_opens_that_stage() -> None:
    run = _revision("operations", None)
    run = _answer(run, "interpretation", interpretation("the base", "the hole"))

    _answer(run, "operations", _renamed_hole())


def test_a_program_statement_follows_its_operation() -> None:
    run = _revision("operations", "op_hole")
    run = _answer(run, "interpretation", interpretation("the base", "the hole"))
    run = _answer(run, "operations", _operations())
    hole = _SOURCE.replace("ret_base.cut(object())", "ret_base.cut(object()).clean()")
    base = hole.replace("ret_base = object()", "ret_base = object().clean()")

    _answer(run, "coding", _verified(hole), dimension_checks={})
    with pytest.raises(SubmissionValidationError, match=r"ret_base \(changed\)"):
        _answer(run, "coding", _verified(base), dimension_checks={})


def test_fields_validation_derives_are_not_changes() -> None:
    run = _revision("interpretation", "sem_feature_1")
    calibrated = interpretation("the base", "the hole")
    calibrated.views[0].scale = 0.5
    calibrated.views[0].image_size = (10, 10)
    _answer(run, "interpretation", calibrated)

    located = interpretation("the base", "the hole")
    evidence = located.features[1].evidence[0]
    located.features[1].evidence[0] = evidence.model_copy(
        update={"box_uv": (0.0, 0.0, 5.0, 5.0)}
    )
    _answer(run, "interpretation", located)

    located.features[1].evidence[0] = evidence.model_copy(
        update={"box_px": (1, 1, 9, 9)}
    )
    with pytest.raises(SubmissionValidationError, match=r"sem_feature_2 \(changed\)"):
        _answer(run, "interpretation", located)


def _verified(source: str) -> VerifyOutputResult:
    return VerifyOutputResult(
        status=ExecutionStatus.VERIFIED, source=source, returncode=0
    )


def test_value_annotations_are_not_changes() -> None:
    written, annotated = _operations(), _operations()
    written.proposal[1].detail = "Cut sem_feature_2.radius deep."
    annotated.proposal[1].detail = "Cut sem_feature_2.radius (= 3.0) deep."

    assert written.members() == annotated.members()
