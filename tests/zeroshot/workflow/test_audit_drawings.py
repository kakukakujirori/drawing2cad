"""Audit's last upstream crossing addresses the owner of a drawing reading."""

import pytest
from pydantic import ValidationError

from tests.zeroshot.contracts import drawing, hypothesis
from tests.zeroshot.workflow.test_reconstruction_workflow import (
    _completed_run,
    _hop,
    _hop_from,
    _ref,
    _report,
    _snapshot,
)
from zeroshot.pipeline.messages.contracts.audit import RevisionRequest
from zeroshot.pipeline.messages.contracts.drawings import Dimension
from zeroshot.pipeline.messages.contracts.stages import REASONING_STAGES
from zeroshot.pipeline.workflow.reconstruction import open_next_round
from zeroshot.pipeline.workflow.validate_submission import (
    SubmissionValidationError,
    validate_submission,
)


def test_audit_traces_the_solid_to_the_sheet_owning_its_reading() -> None:
    report = _report(
        _hop("coding", "ret_base", "operations", "op_base"),
        _hop("operations", "op_base", "semantics", "sem_feature_1"),
        _hop("semantics", "sem_feature_1", "drawings", "sheet_front"),
    )
    run = _completed_run()
    original = run.model_dump_json()
    revised = open_next_round(run, report)
    assert revised.snapshots[-1].open_tickets[0].assigned_stages == list(
        REASONING_STAGES
    )
    assert run.model_dump_json() == original


@pytest.mark.parametrize("with_hop", [False, True])
def test_audit_rejects_a_sheet_absent_from_the_snapshot(with_hop: bool) -> None:
    report = (
        _report(_hop("semantics", "sem_feature_1", "drawings", "sheet_absent"))
        if with_hop
        else _report(target=_ref("drawings", "sheet_absent"))
    )
    with pytest.raises(SubmissionValidationError, match="sheet_absent.*does not exist"):
        validate_submission(report, _snapshot())


def test_audit_rejects_an_existing_sheet_not_cited_by_the_feature() -> None:
    snapshot = _snapshot().model_copy(update={"drawings": drawing("front", "top")})
    report = _report(_hop("semantics", "sem_feature_1", "drawings", "sheet_top"))
    with pytest.raises(SubmissionValidationError, match="sem_feature_1.evidence"):
        validate_submission(report, snapshot)


def test_a_dimension_citation_supports_the_hop_to_its_sheet() -> None:
    snapshot = _snapshot()
    assert snapshot.drawings is not None
    snapshot.drawings.sheets[0].dimensions.append(
        Dimension(
            name="dim_width",
            kind="linear",
            text="10",
            nominal=10,
            quantity=1,
            note=None,
        )
    )
    snapshot.semantics = hypothesis("the base")
    snapshot.semantics.proposal[0].evidence = ["dim_width"]
    validate_submission(
        _report(_hop("semantics", "sem_feature_1", "drawings", "sheet_front")),
        snapshot,
    )


@pytest.mark.parametrize(
    ("effect", "cause"),
    [("coding", "drawings"), ("drawings", "semantics")],
)
def test_audit_rejects_skipped_or_reversed_stages_even_without_names(
    effect: str,
    cause: str,
) -> None:
    report = _report(_hop_from(_ref(effect, None), _ref(cause, None)))
    with pytest.raises(SubmissionValidationError, match="adjacent upstream stage"):
        validate_submission(report, _snapshot())


def test_a_missing_sheet_can_be_requested_without_inventing_a_causal_chain() -> None:
    report = _report(target=_ref("drawings", None))
    report.findings[0].revision_request = RevisionRequest(
        action="add",
        targets=[_ref("drawings", None)],
        instruction="Read the omitted top view from the input page.",
        proposed_names=["sheet_top"],
    )
    validate_submission(report, _snapshot())


@pytest.mark.parametrize("name", ["ev_front_line", "dim_width", "sem_base"])
def test_drawing_revision_proposes_sheets_not_entries(name: str) -> None:
    with pytest.raises(ValidationError, match="invalid proposed names"):
        RevisionRequest(
            action="add",
            targets=[_ref("drawings", None)],
            instruction="Read the omitted top view.",
            proposed_names=[name],
        )
