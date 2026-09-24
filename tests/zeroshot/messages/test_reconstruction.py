import json
from copy import deepcopy

import pytest
from pydantic import ValidationError

from tests.zeroshot.contracts import answered, drawing, evidence, interpretation
from zeroshot.pipeline.stages.audit.contracts import (
    AuditFinding,
    AuditReport,
    RevisionRequest,
    StageOutputRef,
    TicketReview,
)
from zeroshot.pipeline.stages.coding.verify import VerifyOutputResult
from zeroshot.pipeline.stages.contracts import (
    ReconstructionHistory,
    ReconstructionSnapshot,
)
from zeroshot.pipeline.stages.operations.contracts import (
    Operation,
    OperationPlan,
    OperationVerb,
)
from zeroshot.pipeline.stages.tickets.contracts import (
    BootstrapWork,
    StageReport,
    Ticket,
    TicketAnswers,
    TicketResponse,
)
from zeroshot.pipeline.stages.types import PipelineStage
from zeroshot.pipeline.verification import ExecutionStatus
from zeroshot.pipeline.verification.run_cadquery import CadQueryExecutionReport


def _interpretation():
    return interpretation("a base body")


def _operations() -> OperationPlan:
    return OperationPlan(
        proposal=[
            Operation(
                name="op_base",
                verb=OperationVerb.EXTRUDE,
                detail="Extrude the base body.",
                semantics=["sem_feature_1"],
            )
        ],
        rationale="The base body is one extrusion.",
    )


def _finding() -> AuditFinding:
    target = StageOutputRef(
        stage=PipelineStage.INTERPRETATION,
        name="sem_feature_1",
    )
    return AuditFinding(
        name="find_wrong_base",
        observation="The reconstructed base is too wide.",
        evidence=evidence("render_3d/hlg_front.png"),
        backtrace=[],
        revision_request=RevisionRequest(
            action="modify",
            targets=[target],
            instruction="Correct the interpreted base width.",
            proposed_names=[],
        ),
        related_ticket_ids=[],
    )


def _responses(ticket_id: str, *stages: str) -> list[TicketResponse]:
    return [
        TicketResponse(
            ticket_id=ticket_id,
            stage=stage,  # type: ignore[arg-type]
            summary=f"Reviewed sem_feature_1 for {stage}.",
        )
        for stage in stages
    ]


def _ticket(
    ticket_id: str = "ticket_bootstrap",
    *,
    subject: BootstrapWork | AuditFinding | None = None,
    stages: tuple[str, ...] = (),
    assigned: tuple[str, ...] = ("interpretation", "operations", "coding"),
) -> Ticket:
    return Ticket(
        ticket_id=ticket_id,
        subject=subject or BootstrapWork(instruction="Reconstruct the part."),
        assigned_stages=list(assigned),  # type: ignore[arg-type]
        responses=_responses(ticket_id, *stages),
    )


def _snapshot(
    *,
    round: int = 0,
    ticket: Ticket | None = None,
    last_completed_stage: str | None = None,
    verification: VerifyOutputResult | None = None,
) -> ReconstructionSnapshot:
    interpreted = (
        _interpretation()
        if last_completed_stage in {"interpretation", "operations", "coding"}
        else None
    )
    operations = (
        _operations() if last_completed_stage in {"operations", "coding"} else None
    )
    return ReconstructionSnapshot(
        open_tickets=[ticket or _ticket()],
        round=round,
        last_completed_stage=last_completed_stage,  # type: ignore[arg-type]
        interpretation=interpreted,
        operations=operations,
        program_source=verification.exec_report.source
        if verification is not None and verification.exec_report is not None
        else None,
        verification=verification,
    )


def test_a_stage_carries_ticket_answers_and_optional_additional_concerns() -> None:
    """A field a stage must leave empty is a field it can get wrong: four of ten
    GLM runs died sending `rationale` a string against a validator that refused
    it. Every artifact now lives in a workspace file instead."""
    responses = answered(_responses("ticket_bootstrap", "coding"))

    submission = TicketAnswers(
        responses=responses,
        stage_report=StageReport(
            concerns={}, unticketed_changes={}, dimension_checks={}
        ),
    )

    assert submission.responses == responses
    assert submission.stage_report.concerns == {}
    assert submission.stage_report.dimension_checks == {}
    assert submission.stage_report == StageReport(
        concerns={}, unticketed_changes={}, dimension_checks={}
    )
    assert set(TicketAnswers.model_fields) == {
        "responses",
        "stage_report",
    }
    for revision in ("edits", "deleted", "rationale"):
        assert (
            TicketAnswers.model_validate(
                {**submission.model_dump(), revision: "anything"}
            )
            == submission
        )


def test_a_member_the_provider_stringified_is_read_rather_than_refused() -> None:
    """GLM sent `stage_report` as JSON text; the answer was whole, its encoding was not."""
    responses = answered(_responses("ticket_bootstrap", "coding"))
    report = StageReport(
        dimension_checks=None,
        unticketed_changes={},
        concerns={"concern_waist": "the profile must change"},
    )

    submission = TicketAnswers.model_validate(
        {"responses": responses, "stage_report": report.model_dump_json()}
    )

    assert submission.stage_report == report
    assert submission.responses == responses
    # A member that is genuinely wrong still fails as itself.
    with pytest.raises(ValidationError, match="dictionary"):
        TicketAnswers.model_validate({"responses": responses, "stage_report": "[]"})


@pytest.mark.parametrize("stringified", [False, True])
def test_extra_audit_fields_are_ignored_without_changing_declared_content(
    stringified: bool,
) -> None:
    """GLM adds type labels to findings and null type fields to references."""
    report = AuditReport(
        concern_reviews={},
        ticket_reviews={"ticket_initial": TicketReview(summary="Wrong.", solved=False)},
        findings=[
            _finding().model_copy(update={"related_ticket_ids": ["ticket_initial"]})
        ],
    )
    submitted = report.model_dump()
    submitted["findings"][0]["type"] = "wrong_geometry"
    submitted["findings"][0]["revision_request"]["type"] = "object"
    submitted["findings"][0]["revision_request"]["targets"][0]["type"] = None
    if stringified:
        submitted["findings"] = json.dumps(submitted["findings"])
    original = deepcopy(submitted)

    assert AuditReport.model_validate(submitted) == report
    assert submitted == original

    submitted["accepted"] = True
    assert AuditReport.model_validate(submitted) == report  # Legacy field is ignored.
    submitted["accepted"] = False

    if stringified:
        submitted["findings"] = json.loads(submitted["findings"])
    submitted["findings"][0]["severity"] = "high"
    assert AuditReport.model_validate(submitted) == report

    target = submitted["findings"][0]["revision_request"]["targets"][0]
    target["nmae"] = target.pop("name")
    with pytest.raises(ValidationError, match="Field required"):
        AuditReport.model_validate(submitted)


def test_schema_words_used_as_dictionary_keys_are_preserved() -> None:
    submission = TicketAnswers.model_validate(
        {
            "type": "object",
            "responses": {"type": "Keep this response."},
            "stage_report": {
                "concerns": {},
                "dimension_checks": None,
                "unticketed_changes": {},
            },
        }
    )
    assert submission.responses == {"type": "Keep this response."}


def test_interpretation_carries_ticket_answers_while_json_carries_the_artifact() -> (
    None
):
    responses = answered(_responses("ticket_bootstrap", "interpretation"))

    submission = TicketAnswers(
        stage_report=StageReport(
            concerns={}, dimension_checks=None, unticketed_changes={}
        ),
        responses=responses,
    )

    assert submission.responses == responses
    assert submission.stage_report.dimension_checks is None
    assert set(TicketAnswers.model_fields) == {
        "responses",
        "stage_report",
    }


def test_a_stage_submission_does_not_keep_extra_fields() -> None:
    submission = TicketAnswers.model_validate(
        {
            "responses": answered(_responses("ticket_bootstrap", "interpretation")),
            "stage_report": {
                "concerns": {},
                "dimension_checks": None,
                "unticketed_changes": {},
                "type": "empty",
            },
            "artifact": _interpretation(),
        }
    )
    assert "artifact" not in submission.model_dump()
    assert "type" not in submission.stage_report.model_dump()


def test_the_shared_submission_schema_is_provider_safe() -> None:
    schema = TicketAnswers.model_json_schema()
    assert set(schema["properties"]) == {"responses", "stage_report"}
    assert schema["title"] == TicketAnswers.__name__ == "TicketAnswers"


@pytest.mark.parametrize("explanation", ["", " ", "\t\n"])
def test_dimension_checks_require_nonblank_explanations(explanation):
    with pytest.raises(ValidationError, match="dim_width.*must not be blank"):
        TicketAnswers(
            responses={},
            stage_report=StageReport(
                concerns={},
                unticketed_changes={},
                dimension_checks={"dim_width": explanation},
            ),
        )


@pytest.mark.parametrize(
    "missing", ["stage_report", "concerns", "dimension_checks", "unticketed_changes"]
)
def test_missing_or_misspelled_reports_are_not_silently_empty(missing):
    payload = {
        "responses": {},
        "stage_report": {
            "concerns": {},
            "dimension_checks": None,
            "unticketed_changes": {},
        },
    }
    holder = payload if missing == "stage_report" else payload["stage_report"]
    holder[missing + "_typo"] = holder.pop(missing)
    with pytest.raises(ValidationError) as raised:
        TicketAnswers.model_validate(payload)
    assert any(
        e["type"] == "missing" and e["loc"][-1] == missing
        for e in raised.value.errors()
    )


def test_a_round_checkpoint_requires_every_ticket_response_in_stage_order() -> None:
    ticket = _ticket(stages=("interpretation",))

    with pytest.raises(ValidationError, match="responses must be"):
        _snapshot(
            ticket=ticket,
            last_completed_stage="operations",
        )


@pytest.mark.parametrize(
    "assigned",
    [(), ("interpretation",), ("interpretation", "coding"), ("coding", "operations")],
)
def test_a_ticket_assignment_runs_from_one_revision_root_through_coding(
    assigned: tuple[str, ...],
) -> None:
    with pytest.raises(ValidationError, match="assigned"):
        _ticket(assigned=assigned)


def test_a_ticket_rejects_a_response_from_a_stage_it_does_not_assign() -> None:
    with pytest.raises(ValidationError, match="not assigned to this ticket"):
        _ticket(stages=("interpretation",), assigned=("operations", "coding"))


def test_a_completed_stage_leaves_no_response_on_a_ticket_it_is_not_assigned() -> None:
    snapshot = _snapshot(
        ticket=_ticket(assigned=("operations", "coding")),
        last_completed_stage="interpretation",
    )

    assert snapshot.open_tickets[0].responses == []


def test_a_ticket_rejects_a_response_for_another_ticket() -> None:
    with pytest.raises(ValidationError, match="containing ticket"):
        Ticket(
            ticket_id="ticket_one",
            subject=BootstrapWork(instruction="Reconstruct the part."),
            assigned_stages=[
                PipelineStage.INTERPRETATION,
                PipelineStage.OPERATIONS,
                PipelineStage.CODING,
            ],
            responses=_responses("ticket_other", "interpretation"),
        )


def test_a_failed_verification_is_a_valid_completed_coding_checkpoint() -> None:
    report = VerifyOutputResult(
        exec_report=CadQueryExecutionReport(
            status=ExecutionStatus.REJECTED, executor_error="model.py was not found"
        )
    )
    ticket = _ticket(stages=("interpretation", "operations", "coding"))

    snapshot = _snapshot(
        ticket=ticket,
        last_completed_stage="coding",
        verification=report,
    )

    assert snapshot.program_source is None
    assert snapshot.verification is report


def test_an_uninitialized_verification_does_not_complete_coding() -> None:
    ticket = _ticket(stages=("interpretation", "operations", "coding"))

    with pytest.raises(ValidationError, match="must be completed"):
        _snapshot(
            ticket=ticket,
            last_completed_stage="coding",
            verification=VerifyOutputResult(
                exec_report=CadQueryExecutionReport(
                    status=ExecutionStatus.UNINITIALIZED
                )
            ),
        )


@pytest.mark.parametrize(
    ("completed_stage", "field", "value"),
    [
        (None, "interpretation", _interpretation()),
        ("interpretation", "operations", _operations()),
        ("operations", "program_source", "result = object()\n"),
        (
            "operations",
            "verification",
            VerifyOutputResult(
                exec_report=CadQueryExecutionReport(status=ExecutionStatus.REJECTED)
            ),
        ),
    ],
)
def test_snapshot_rejects_an_artifact_from_an_unfinished_stage(
    completed_stage: str | None,
    field: str,
    value,
) -> None:
    completed_stages = {
        None: (),
        "interpretation": ("interpretation",),
        "operations": ("interpretation", "operations"),
    }[completed_stage]
    snapshot = _snapshot(
        ticket=_ticket(stages=completed_stages),
        last_completed_stage=completed_stage,
    )
    data = snapshot.model_dump()
    data[field] = value

    with pytest.raises(ValidationError, match="unfinished stage artifacts"):
        ReconstructionSnapshot.model_validate(data)


@pytest.mark.parametrize("stage", ["interpretation", "operations", "coding"])
def test_snapshot_rejects_reports_from_unfinished_stages(stage):
    data = _snapshot().model_dump()
    data["stage_reports"] = {
        stage: {
            "concerns": {"concern_web": "A concern."},
            "dimension_checks": None,
            "unticketed_changes": {},
        }
    }
    with pytest.raises(ValidationError, match="unfinished stages.*stage_reports"):
        ReconstructionSnapshot.model_validate(data)


def test_old_completed_history_loads_without_claiming_a_stage_report():
    snapshot = _snapshot(
        ticket=_ticket(stages=("interpretation",)),
        last_completed_stage="interpretation",
    )
    run = ReconstructionHistory(
        run_id="run_legacy", input_drawings=drawing(), snapshots=[snapshot]
    )
    data = run.model_dump()
    del data["snapshots"][0]["stage_reports"]

    restored = ReconstructionHistory.model_validate(data)
    assert restored.snapshots[0].stage_reports == {}


def test_round_zero_requires_exactly_one_bootstrap_ticket() -> None:
    first = _snapshot()
    second_ticket = _ticket("ticket_second_bootstrap")
    invalid_first = first.model_copy(
        update={"open_tickets": [*first.open_tickets, second_ticket]}
    )

    with pytest.raises(ValidationError, match="exactly one bootstrap"):
        ReconstructionHistory(
            run_id="run_example",
            input_drawings=drawing(),
            snapshots=[invalid_first],
        )


def test_round_zero_rejects_a_finding_in_place_of_bootstrap_work() -> None:
    first = _snapshot(
        ticket=_ticket("ticket_wrong_base", subject=_finding()),
    )

    with pytest.raises(ValidationError, match="exactly one bootstrap"):
        ReconstructionHistory(
            run_id="run_example",
            input_drawings=drawing(),
            snapshots=[first],
        )


def test_later_rounds_reject_bootstrap_tickets() -> None:
    first = _snapshot(
        ticket=_ticket(stages=("interpretation", "operations", "coding")),
        last_completed_stage="coding",
        verification=VerifyOutputResult(
            exec_report=CadQueryExecutionReport(status=ExecutionStatus.REJECTED)
        ),
    )
    second = _snapshot(
        round=1,
        ticket=_ticket("ticket_late_bootstrap"),
    )

    with pytest.raises(ValidationError, match="only in round 0"):
        ReconstructionHistory(
            run_id="run_example",
            input_drawings=drawing(),
            snapshots=[first, second],
        )


def test_round_numbers_follow_snapshot_order() -> None:
    with pytest.raises(ValidationError, match="snapshot rounds"):
        ReconstructionHistory(
            run_id="run_example",
            input_drawings=drawing(),
            snapshots=[_snapshot(round=1)],
        )


def test_a_run_round_trips_bootstrap_findings_and_verification_as_json() -> None:
    first = _snapshot(
        ticket=_ticket(stages=("interpretation", "operations", "coding")),
        last_completed_stage="coding",
        verification=VerifyOutputResult(
            verification_id="000",
            exec_report=CadQueryExecutionReport(
                status=ExecutionStatus.VERIFIED,
                source="result = object()",
                returncode=0,
            ),
        ),
    )
    second = ReconstructionSnapshot(
        open_tickets=[
            _ticket(
                "ticket_wrong_base",
                subject=_finding(),
            )
        ],
        round=1,
        last_completed_stage=None,
        interpretation=None,
        operations=None,
        program_source=None,
        verification=None,
    )
    run = ReconstructionHistory(
        run_id="run_example",
        input_drawings=drawing(),
        snapshots=[first, second],
    )

    restored = ReconstructionHistory.model_validate_json(run.model_dump_json())

    assert restored == run
    assert isinstance(restored.snapshots[0].verification, VerifyOutputResult)
    assert restored.snapshots[0].verification is not None
    assert (
        restored.snapshots[0].verification.exec_report.status
        is ExecutionStatus.VERIFIED
    )
    assert isinstance(restored.snapshots[1].open_tickets[0].subject, AuditFinding)


def test_a_stage_with_no_ticket_of_its_own_answers_nothing() -> None:
    assert (
        TicketAnswers(
            stage_report=StageReport(
                concerns={}, dimension_checks=None, unticketed_changes={}
            ),
            responses={},
        ).responses
        == {}
    )
