"""Contextual validation shared by all reconstruction stages."""

import re
from collections.abc import Sequence

import pytest

from tests.zeroshot.contracts import drawing, interpretation, interpreted_feature
from zeroshot.pipeline.stages.coding.verify import VerifyOutputResult
from zeroshot.pipeline.stages.contracts import (
    ReconstructionHistory,
    ReconstructionSnapshot,
)
from zeroshot.pipeline.stages.interpretation.contracts import (
    Dimension,
    DrawingInterpretation,
    Region,
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
from zeroshot.pipeline.stages.types import (
    REASONING_STAGES,
    PipelineStage,
    ReasoningStage,
    next_stage,
)
from zeroshot.pipeline.stages.validate import (
    SubmissionValidationError,
    validate_submission,
)
from zeroshot.pipeline.verification import ExecutionStatus
from zeroshot.pipeline.workflow.lifecycle import (
    _commit_snapshot,
    advance_reconstruction,
    load_reconstruction,
    save_reconstruction,
    start_reconstruction,
)


def _interpretation() -> DrawingInterpretation:
    return interpretation("the base")


def _operations(*, semantics: list[str] | None = None) -> OperationPlan:
    return OperationPlan(
        proposal=[
            Operation(
                name="op_base",
                verb=OperationVerb.EXTRUDE,
                detail="Extrude the base.",
                semantics=(semantics if semantics is not None else ["sem_feature_1"]),
            )
        ],
        rationale="The base is one extrusion.",
    )


def _plan_for(
    semantics: list[str],
    *,
    detail: str = "Build the feature.",
) -> OperationPlan:
    return OperationPlan(
        proposal=[
            Operation(
                name="op_feature",
                verb=OperationVerb.EXTRUDE,
                detail=detail,
                semantics=semantics,
            )
        ],
        rationale="The operation constructs the named features.",
    )


def _response(ticket_id: str, stage: ReasoningStage) -> TicketResponse:
    return TicketResponse(
        ticket_id=ticket_id,
        stage=stage,
        summary=f"Reviewed {ticket_id} during {stage}.",
    )


def _answer_for(ticket_id: str, stage: ReasoningStage) -> dict[str, str]:
    """The submitted shape: answers keyed by ticket, with no stage to disagree."""
    return {ticket_id: f"Reviewed {ticket_id} during {stage}."}


def _ticket(
    ticket_id: str,
    *completed_stages: ReasoningStage,
    assigned: Sequence[ReasoningStage] = REASONING_STAGES,
) -> Ticket:
    return Ticket(
        ticket_id=ticket_id,
        subject=BootstrapWork(instruction="Reconstruct the part."),
        assigned_stages=list(assigned),
        responses=[_response(ticket_id, stage) for stage in completed_stages],
    )


def _snapshot(
    completed_stage: ReasoningStage | None,
    *,
    tickets: list[Ticket] | None = None,
    held: DrawingInterpretation | None = None,
) -> ReconstructionSnapshot:
    completed = (
        list(REASONING_STAGES[: REASONING_STAGES.index(completed_stage) + 1])
        if completed_stage
        else []
    )
    verification = (
        VerifyOutputResult(
            status=ExecutionStatus.VERIFIED,
            source="ret_base = object()\nresult = ret_base\n",
            returncode=0,
        )
        if completed_stage is PipelineStage.CODING
        else None
    )
    return ReconstructionSnapshot(
        open_tickets=tickets or [_ticket("ticket_initial", *completed)],
        round=0,
        last_completed_stage=completed_stage,
        interpretation=(held if held is not None else _interpretation())
        if completed
        else None,
        operations=_operations()
        if completed_stage in (PipelineStage.OPERATIONS, PipelineStage.CODING)
        else None,
        program_source=verification.source if verification is not None else None,
        verification=verification,
    )


def _verified_and_validate(output, snapshot, *, workspace_output=None):
    """Validate against the artifact this round's verifier would hand over."""
    deliverable = workspace_output
    if deliverable is None:
        deliverable = {
            PipelineStage.INTERPRETATION: _interpretation(),
            PipelineStage.OPERATIONS: _operations(),
        }.get(next_stage(snapshot.last_completed_stage))
    validate_submission(output, snapshot, deliverable=deliverable)


def test_every_reasoning_stage_accepts_its_expected_deliverable() -> None:
    _verified_and_validate(
        TicketAnswers(
            responses=_answer_for("ticket_initial", PipelineStage.INTERPRETATION),
        ),
        _snapshot(None),
    )
    _verified_and_validate(
        TicketAnswers(
            responses=_answer_for("ticket_initial", PipelineStage.OPERATIONS),
        ),
        _snapshot(PipelineStage.INTERPRETATION),
    )
    _verified_and_validate(
        TicketAnswers(
            responses=_answer_for("ticket_initial", PipelineStage.CODING),
            stage_report=StageReport(dimension_checks={}),
        ),
        _snapshot(PipelineStage.OPERATIONS),
        workspace_output=VerifyOutputResult(status=ExecutionStatus.REJECTED),
    )


@pytest.mark.parametrize(
    ("responses", "message"),
    [
        (
            _answer_for("ticket_one", PipelineStage.INTERPRETATION),
            "missing.*ticket_two",
        ),
        (
            _answer_for("ticket_one", PipelineStage.INTERPRETATION)
            | _answer_for("ticket_unknown", PipelineStage.INTERPRETATION),
            "unknown.*ticket_unknown",
        ),
        # GLM left the old field name behind as a key; the answer is a key
        # short, so the check that knows the open tickets names them.
        (
            _answer_for("ticket_one", PipelineStage.INTERPRETATION)
            | {"summary": "a leftover field name"},
            (
                r"unknown ticket responses: summary\. Open tickets for this "
                r"stage: ticket_one, ticket_two"
            ),
        ),
    ],
)
def test_ticket_responses_must_cover_the_current_snapshot_exactly_once(
    responses: dict[str, str],
    message: str,
) -> None:
    """Keying by ticket leaves only membership to check: a duplicate ticket or a
    response belonging to another stage cannot be written down."""
    snapshot = _snapshot(
        None,
        tickets=[
            _ticket("ticket_one"),
            _ticket("ticket_two"),
        ],
    )
    submission = TicketAnswers(
        responses=responses,
    )

    with pytest.raises(SubmissionValidationError, match=message):
        _verified_and_validate(submission, snapshot)


def test_a_stage_answers_its_assigned_tickets_and_only_those() -> None:
    snapshot = _snapshot(
        None,
        tickets=[
            _ticket("ticket_one"),
            _ticket("ticket_two", assigned=(PipelineStage.CODING,)),
        ],
    )

    _verified_and_validate(
        TicketAnswers(
            responses=_answer_for("ticket_one", PipelineStage.INTERPRETATION),
        ),
        snapshot,
    )

    with pytest.raises(SubmissionValidationError, match="not assigned.*ticket_two"):
        _verified_and_validate(
            TicketAnswers(
                responses=_answer_for("ticket_one", PipelineStage.INTERPRETATION)
                | _answer_for("ticket_two", PipelineStage.INTERPRETATION),
            ),
            snapshot,
        )


def test_a_stage_assigned_nothing_answers_nothing() -> None:
    snapshot = _snapshot(
        None,
        tickets=[_ticket("ticket_one", assigned=(PipelineStage.CODING,))],
    )

    _verified_and_validate(
        TicketAnswers(responses={}),
        snapshot,
    )


def test_the_current_snapshot_decides_which_deliverable_type_is_valid() -> None:
    answers = TicketAnswers(
        responses=_answer_for("ticket_initial", PipelineStage.INTERPRETATION),
    )

    with pytest.raises(
        SubmissionValidationError, match="verified DrawingInterpretation"
    ):
        _verified_and_validate(answers, _snapshot(None), workspace_output=_operations())


def test_operations_must_cover_only_current_semantic_features() -> None:
    submission = TicketAnswers(
        responses=_answer_for("ticket_initial", PipelineStage.OPERATIONS),
    )

    with pytest.raises(SubmissionValidationError, match="sem_feature_1"):
        _verified_and_validate(
            submission,
            _snapshot(PipelineStage.INTERPRETATION),
            workspace_output=_operations(semantics=["sem_absent"]),
        )


def test_interpretation_rejects_an_evidence_view_absent_from_the_artifact() -> None:
    from pydantic import ValidationError

    with pytest.raises(
        ValidationError, match=r"sem_bore.evidence\[0\].view: unknown view view_absent"
    ):
        interpretation(
            hypotheses=[
                interpreted_feature(
                    "sem_bore",
                    "bore",
                    evidence=[Region(view="view_absent", box_px=(0, 0, 1, 1))],
                )
            ]
        )


def _validate_plan(plan: OperationPlan, held: DrawingInterpretation) -> None:
    _verified_and_validate(
        TicketAnswers(
            responses=_answer_for("ticket_initial", PipelineStage.OPERATIONS),
        ),
        _snapshot(PipelineStage.INTERPRETATION, held=held),
        workspace_output=plan,
    )


def test_operation_validation_names_both_missing_and_invented_features() -> None:
    semantics = interpretation("base", "bore")
    plan = _plan_for(["sem_feature_1", "sem_absent"])
    plan.rationale = "The part is complete without sem_feature_2."

    with pytest.raises(SubmissionValidationError) as caught:
        _validate_plan(plan, semantics)

    message = str(caught.value)
    assert "no operation in the plan builds" in message
    assert "sem_feature_2" in message
    assert "the interpretation does not contain" in message
    assert "sem_absent" in message
    assert "rationale" not in message
    assert len(message.splitlines()) == 2


def test_a_plan_builds_the_adopted_candidate_not_the_one_rejected() -> None:
    boss = interpreted_feature("sem_boss", "boss", parameters={"radius": 4.0})
    hole = boss.candidates[0].model_copy(
        update={"name": "sem_hole", "description": "hole", "confidence": 0.2}
    )
    held = interpretation(
        hypotheses=[boss.model_copy(update={"candidates": [hole, *boss.candidates]})]
    )
    plan = _plan_for(["sem_hole"], detail="Cut sem_hole.radius deep.")

    with pytest.raises(SubmissionValidationError) as caught:
        _validate_plan(plan, held)

    message = str(caught.value)
    assert "sem_hole, a candidate the interpretation did not adopt" in message
    assert "op_feature: sem_hole is a candidate" in message
    assert "Use sem_boss" in message
    _validate_plan(_plan_for(["sem_boss"], detail="Add sem_boss.radius."), held)


def _measured_blend() -> DrawingInterpretation:
    return interpretation(
        hypotheses=[
            interpreted_feature(
                "sem_shoulder_blend",
                "shoulder blend",
                parameters={
                    "major_radius": 11.31245992416,
                    "tube_radius": 3.39440063713,
                },
            )
        ]
    )


@pytest.mark.parametrize(
    "literal",
    ["11.31245992416", "-11.31245992416", "11.31245992416e-6", "5.65622996208"],
)
def test_operation_validation_does_not_infer_copying_from_numeric_literals(
    literal: str,
) -> None:
    _validate_plan(
        _plan_for(["sem_shoulder_blend"], detail=f"Use an offset of {literal}."),
        _measured_blend(),
    )


def test_operation_validation_accepts_derived_and_short_numbers() -> None:
    _validate_plan(
        _plan_for(
            ["sem_shoulder_blend"],
            detail="Cut 5.65622996208 deep, half of sem_shoulder_blend.major_radius.",
        ),
        _measured_blend(),
    )
    _validate_plan(
        _plan_for(["sem_boss"], detail="Extrude 25 mm."),
        interpretation(
            hypotheses=[
                interpreted_feature("sem_boss", "boss", parameters={"radius": 25.0})
            ]
        ),
    )


def test_operation_validation_rejects_a_nonexistent_parameter_address() -> None:
    address = "sem_shoulder_blend.height"

    with pytest.raises(SubmissionValidationError, match=address):
        _validate_plan(
            _plan_for(
                ["sem_shoulder_blend"],
                detail=f"Sweep {address} along +z, offset by 11.31245992416.",
            ),
            _measured_blend(),
        )


def test_operation_validation_accepts_a_whole_position_parameter() -> None:
    held = interpretation(
        hypotheses=[
            interpreted_feature(
                "sem_main_bore", "main bore", parameters={"center": [1.5, 2.5, 0.0]}
            )
        ]
    )
    _validate_plan(
        _plan_for(["sem_main_bore"], detail="Cut from sem_main_bore.center."), held
    )


def test_operation_validation_rejects_a_coordinate_of_a_single_number() -> None:
    """A scalar parameter cannot be treated as a coordinate vector."""
    address = "sem_shoulder_blend.major_radius.x"

    with pytest.raises(SubmissionValidationError, match=re.escape(address)):
        _validate_plan(
            _plan_for(["sem_shoulder_blend"], detail=f"Sweep {address} along +z."),
            _measured_blend(),
        )


def test_operation_validation_accepts_a_printed_dimension_reference() -> None:
    held = interpretation(
        hypotheses=[
            interpreted_feature(
                "sem_main_bore", "main bore", dimension_refs=["dim_depth"]
            )
        ],
        views=[
            _interpretation()
            .views[0]
            .model_copy(
                update={
                    "dimensions": [
                        Dimension(
                            name="dim_depth",
                            kind="linear",
                            text="10",
                            nominal_value=10,
                            region=Region(view="view_front", box_px=(0, 0, 10, 10)),
                            quantity=1,
                            note=None,
                        )
                    ]
                }
            )
        ],
    )
    _validate_plan(
        _plan_for(["sem_main_bore"], detail="Extrude dim_depth.nominal_value."), held
    )


def test_operation_validation_accepts_a_reference_with_its_resolved_value() -> None:
    _validate_plan(
        _plan_for(
            ["sem_shoulder_blend"],
            detail=(
                "Sweep a blend of sem_shoulder_blend.major_radius (= 11.31245992416)."
            ),
        ),
        _measured_blend(),
    )


def test_only_coding_accepts_a_separate_terminal_verification() -> None:
    interpretation_submission = TicketAnswers(
        responses=_answer_for("ticket_initial", PipelineStage.INTERPRETATION),
    )
    coding_submission = TicketAnswers(
        responses=_answer_for("ticket_initial", PipelineStage.CODING),
        stage_report=StageReport(dimension_checks={}),
    )

    with pytest.raises(SubmissionValidationError, match="DrawingInterpretation"):
        _verified_and_validate(
            interpretation_submission,
            _snapshot(None),
            workspace_output=VerifyOutputResult(status=ExecutionStatus.REJECTED),
        )
    with pytest.raises(SubmissionValidationError, match="requires"):
        _verified_and_validate(coding_submission, _snapshot(PipelineStage.OPERATIONS))
    with pytest.raises(SubmissionValidationError, match="must be terminal"):
        _verified_and_validate(
            coding_submission,
            _snapshot(PipelineStage.OPERATIONS),
            workspace_output=VerifyOutputResult(),
        )


def test_coding_checks_the_submitted_program_against_current_round_operations() -> None:
    submission = TicketAnswers(
        responses=_answer_for("ticket_initial", PipelineStage.CODING),
        stage_report=StageReport(dimension_checks={}),
    )

    with pytest.raises(SubmissionValidationError, match="missing.*op_base"):
        _verified_and_validate(
            submission,
            _snapshot(PipelineStage.OPERATIONS),
            workspace_output=VerifyOutputResult(
                status=ExecutionStatus.REJECTED,
                source="ret_other = object()\nresult = ret_other\n",
            ),
        )


def test_coding_reports_dimension_and_program_faults_together() -> None:
    submission = TicketAnswers(
        responses=_answer_for("ticket_initial", PipelineStage.CODING),
        stage_report=StageReport(dimension_checks={"dim_r16": "from the leader"}),
    )

    with pytest.raises(
        SubmissionValidationError,
        match=r"(?s)unknown: \['dim_r16'\].*concerns.*op_base",
    ):
        _verified_and_validate(
            submission,
            _snapshot(PipelineStage.OPERATIONS),
            workspace_output=VerifyOutputResult(
                status=ExecutionStatus.REJECTED,
                source="ret_other = object()\nresult = ret_other\n",
            ),
        )


def test_coding_keeps_a_terminal_unreadable_program_auditable() -> None:
    submission = TicketAnswers(
        responses=_answer_for("ticket_initial", PipelineStage.CODING),
        stage_report=StageReport(dimension_checks={}),
    )

    _verified_and_validate(
        submission,
        _snapshot(PipelineStage.OPERATIONS),
        workspace_output=VerifyOutputResult(status=ExecutionStatus.REJECTED),
    )

    with pytest.raises(SubmissionValidationError, match="requires dimension_checks"):
        _verified_and_validate(
            submission.model_copy(
                update={"stage_report": StageReport(dimension_checks=None)}
            ),
            _snapshot(PipelineStage.OPERATIONS),
            workspace_output=VerifyOutputResult(status=ExecutionStatus.REJECTED),
        )


def _dimensioned_interpretation():
    held = _interpretation()
    first = held.views[0]
    first.dimensions = [
        Dimension(
            name="dim_width",
            kind="linear",
            text="10",
            nominal_value=10,
            region=first.region,
            quantity=1,
            note=None,
        )
    ]
    held.views.append(
        first.model_copy(
            update={
                "name": "view_top",
                "file": "top.png",
                "region": Region(view="view_top", box_px=(0, 0, 10, 10)),
                "dimensions": [
                    first.dimensions[0].model_copy(update={"name": "dim_equal"}),
                    first.dimensions[0].model_copy(
                        update={
                            "name": "dim_unreadable",
                            "nominal_value": None,
                            "text": "?",
                        }
                    ),
                ],
            }
        )
    )
    return held


@pytest.mark.parametrize("source", [None, "def broken("])
def test_coding_checks_all_dimensions_once_across_tickets_even_without_a_program(
    source,
):
    held = _dimensioned_interpretation()
    snapshot = _snapshot(
        PipelineStage.OPERATIONS,
        held=held,
        tickets=[
            _ticket(name, PipelineStage.INTERPRETATION, PipelineStage.OPERATIONS)
            for name in ("ticket_one", "ticket_two")
        ],
    )
    submission = TicketAnswers(
        responses={
            ticket.ticket_id: "Addressed during coding."
            for ticket in snapshot.open_tickets
        },
        stage_report=StageReport(
            concerns={
                "concern_no_solid": "The program failed before a solid was available."
            },
            dimension_checks={
                "dim_width": "Not established: ret_base has not produced a solid.",
                "dim_equal": "Not checked: same intended extent as dim_width, but no final solid.",
                "dim_unreadable": "Not established: the printed value is unreadable.",
            },
        ),
    )
    terminal = VerifyOutputResult(status=ExecutionStatus.REJECTED, source=source)
    validate_submission(submission, snapshot, deliverable=terminal)
    assert held.hypotheses[0].dimension_refs == []
    del submission.stage_report.dimension_checks["dim_unreadable"]
    with pytest.raises(SubmissionValidationError, match="missing.*dim_unreadable"):
        validate_submission(submission, snapshot, deliverable=terminal)


@pytest.mark.parametrize(
    ("names", "message"),
    [
        (None, "requires dimension_checks"),
        ([], "missing.*dim_equal.*dim_unreadable.*dim_width"),
        (["dim_width", "dim_unreadable"], "missing.*dim_equal"),
        (
            ["dim_width", "dim_equal", "dim_unreadable", "dim_invented"],
            "unknown.*dim_invented",
        ),
    ],
)
def test_coding_dimension_coverage_uses_ids_including_equal_and_unreadable_values(
    names, message
):
    submission = TicketAnswers(
        responses=_answer_for("ticket_initial", PipelineStage.CODING),
        stage_report=StageReport(
            dimension_checks=None
            if names is None
            else {name: "Not checked: no solid." for name in names}
        ),
    )
    with pytest.raises(SubmissionValidationError, match=message):
        validate_submission(
            submission,
            _snapshot(PipelineStage.OPERATIONS, held=_dimensioned_interpretation()),
            deliverable=VerifyOutputResult(status=ExecutionStatus.REJECTED),
        )


@pytest.mark.parametrize(
    "stage", [PipelineStage.INTERPRETATION, PipelineStage.OPERATIONS]
)
def test_non_coding_stages_cannot_submit_dimension_checks_even_when_empty(stage):
    snapshot = _snapshot(
        None if stage is PipelineStage.INTERPRETATION else PipelineStage.INTERPRETATION
    )
    with pytest.raises(
        SubmissionValidationError, match="dimension_checks must be null"
    ):
        _verified_and_validate(
            TicketAnswers(
                responses=_answer_for("ticket_initial", stage),
                stage_report=StageReport(dimension_checks={}),
            ),
            snapshot,
        )


def test_coding_saves_concerns_and_resolved_dimension_checks_together(tmp_path):
    run = ReconstructionHistory(
        run_id="run_dimension_checks",
        input_drawings=drawing(),
        snapshots=[
            _snapshot(PipelineStage.OPERATIONS, held=_dimensioned_interpretation())
        ],
    )
    submission = TicketAnswers(
        responses=_answer_for("ticket_initial", PipelineStage.CODING),
        stage_report=StageReport(
            concerns={
                "concern_width": "The adopted width remains dim_width.nominal_value."
            },
            dimension_checks={
                "dim_width": "Not established: ret_base was intended to span dim_width.nominal_value.",
                "dim_equal": "Not checked: no solid to compare against dim_equal.nominal_value.",
                "dim_unreadable": "Not established: dim_unreadable.nominal_value is unreadable.",
            },
        ),
    )
    committed = advance_reconstruction(
        run,
        submission,
        workspace_output=VerifyOutputResult(status=ExecutionStatus.REJECTED),
    )
    path = tmp_path / "reconstruction.json"
    save_reconstruction(path, committed)
    restored = load_reconstruction(path)
    report = restored.snapshots[-1].stage_reports[PipelineStage.CODING]
    assert "dim_width.nominal_value (= 10.0)" in report.concerns["concern_width"]
    assert "dim_width.nominal_value (= 10.0)" in report.dimension_checks["dim_width"]
    assert (
        "dim_unreadable.nominal_value (= null)"
        in report.dimension_checks["dim_unreadable"]
    )
    assert set(report.model_dump()) == {
        "concerns",
        "dimension_checks",
        "unticketed_changes",
    }
    assert len(restored.snapshots[-1].open_tickets[0].responses) == 3
    assert submission.stage_report.dimension_checks["dim_width"].endswith(
        "dim_width.nominal_value."
    )


def test_completed_coding_accepts_only_an_audit_report() -> None:
    submission = TicketAnswers(
        responses=_answer_for("ticket_initial", PipelineStage.CODING),
        stage_report=StageReport(dimension_checks={}),
    )

    with pytest.raises(SubmissionValidationError, match="only an AuditReport"):
        _verified_and_validate(
            submission,
            _snapshot(PipelineStage.CODING),
            workspace_output=VerifyOutputResult(status=ExecutionStatus.REJECTED),
        )


def test_interpretation_cannot_submit_ticket_answers_without_a_verified_artifact() -> (
    None
):
    submission = TicketAnswers(
        responses=_answer_for("ticket_initial", PipelineStage.INTERPRETATION)
    )
    with pytest.raises(
        SubmissionValidationError, match="verified DrawingInterpretation"
    ):
        validate_submission(submission, _snapshot(None))


def test_stage_reports_commit_with_artifacts_and_responses_and_survive_resume(tmp_path):
    run = start_reconstruction("run_reports", "Read the drawing.", drawing())
    held = _interpretation()
    held.features[0].parameters["width"] = 12.0
    submission = TicketAnswers(
        responses={"ticket_initial": "Established sem_feature_1.width."},
        stage_report=StageReport(
            concerns={
                "concern_height": "The height uses sem_feature_1.width provisionally."
            }
        ),
    )
    with pytest.raises(
        SubmissionValidationError, match="verified DrawingInterpretation"
    ):
        advance_reconstruction(run, submission)
    assert run.snapshots[-1].stage_reports == {}
    interpreted = advance_reconstruction(run, submission, workspace_output=held)
    snapshot = interpreted.snapshots[-1]
    report = snapshot.stage_reports[PipelineStage.INTERPRETATION]
    assert "sem_feature_1.width (= 12.0)" in report.concerns["concern_height"]
    assert (
        "sem_feature_1.width (= 12.0)" in snapshot.open_tickets[0].responses[0].summary
    )
    assert set(report.model_dump()) == {
        "concerns",
        "dimension_checks",
        "unticketed_changes",
    }
    assert type(report) is StageReport
    assert run.snapshots[-1].stage_reports == {}
    history = tmp_path / "reconstruction.json"
    save_reconstruction(history, interpreted)
    interpreted = load_reconstruction(history)
    assert interpreted.snapshots[-1].stage_reports == snapshot.stage_reports

    before = interpreted.model_dump_json()
    with pytest.raises(SubmissionValidationError, match="missing ticket responses"):
        advance_reconstruction(
            interpreted,
            TicketAnswers(
                responses={},
                stage_report=StageReport(
                    concerns={"concern_other": "This does not answer the ticket."}
                ),
            ),
            workspace_output=_operations(),
        )
    assert interpreted.model_dump_json() == before
    planned = advance_reconstruction(
        interpreted,
        TicketAnswers(
            responses=_answer_for("ticket_initial", PipelineStage.OPERATIONS)
        ),
        workspace_output=_operations(),
    )
    assert planned.snapshots[-1].stage_reports == {
        PipelineStage.INTERPRETATION: report,
        PipelineStage.OPERATIONS: StageReport(),
    }

    for reports in (
        {PipelineStage.OPERATIONS: StageReport()},
        {
            PipelineStage.INTERPRETATION: StageReport(
                concerns={"concern_upstream": "Rewritten upstream."}
            ),
            PipelineStage.OPERATIONS: StageReport(),
        },
    ):
        tampered = ReconstructionSnapshot.model_validate(
            {**dict(planned.snapshots[-1]), "stage_reports": reports}
        )
        with pytest.raises(
            ValueError, match="preserve the current interpretation stage report"
        ):
            _commit_snapshot(interpreted, tampered)
