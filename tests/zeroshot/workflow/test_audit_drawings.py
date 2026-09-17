"""Audit links through the interpretation, including direct missing-member tickets."""

import pytest

from tests.zeroshot.workflow.test_resolve_submission import interpretation
from zeroshot.pipeline.stages._base.validate import SubmissionValidationError
from zeroshot.pipeline.stages.audit.contracts import (
    AuditFinding,
    AuditReport,
    CausalHop,
    RevisionRequest,
    StageOutputRef,
)
from zeroshot.pipeline.stages.audit.validate import validate_audit_report
from zeroshot.pipeline.stages.contracts import ReconstructionSnapshot
from zeroshot.pipeline.stages.interpretation.contracts import DrawingInterpretation
from zeroshot.pipeline.stages.operations.contracts import Operation, OperationPlan
from zeroshot.pipeline.stages.tickets.contracts import (
    BootstrapWork,
    Ticket,
    TicketResponse,
)
from zeroshot.pipeline.stages.types import REASONING_STAGES, PipelineStage
from zeroshot.pipeline.verification import ExecutionStatus, VerifyOutputResult


def snapshot() -> ReconstructionSnapshot:
    return ReconstructionSnapshot(
        open_tickets=[
            Ticket(
                ticket_id="ticket_initial",
                subject=BootstrapWork(instruction="Reconstruct"),
                assigned_stages=list(REASONING_STAGES),
                responses=[
                    TicketResponse(
                        ticket_id="ticket_initial", stage=stage, summary="Completed."
                    )
                    for stage in REASONING_STAGES
                ],
            )
        ],
        round=0,
        last_completed_stage=PipelineStage.CODING,
        interpretation=interpretation(),
        operations=OperationPlan(
            proposal=[
                Operation(
                    name="op_bore",
                    verb="hole",
                    detail="Cut the bore.",
                    depends_on=[],
                    semantics=["sem_bore"],
                )
            ],
            rationale="One bore.",
        ),
        program_source="ret_bore = object()\nresult = ret_bore\n",
        verification=VerifyOutputResult(status=ExecutionStatus.VERIFIED, returncode=0),
    )


def ref(stage: str, name: str | None) -> StageOutputRef:
    return StageOutputRef(stage=stage, name=name)


def report(
    *hops: tuple[str, str, str, str], target: str | None = None, add: str | None = None
) -> AuditReport:
    backtrace = [
        CausalHop(
            effect=ref(es, en),
            cause=ref(cs, cn),
            rationale="Declared source of the mismatch.",
        )
        for es, en, cs, cn in hops
    ]
    root = backtrace[-1].cause if backtrace else ref("interpretation", target)
    return AuditReport(
        accepted=False,
        ticket_reviews=[],
        findings=[
            AuditFinding(
                name="find_bore",
                observation="The drawn bore is missing or incorrect.",
                evidence=["front.png"],
                backtrace=backtrace,
                revision_request=RevisionRequest(
                    action="add" if add else "modify",
                    targets=[root],
                    instruction="Correct the bore.",
                    proposed_names=[add] if add else [],
                ),
                related_ticket_ids=[],
            )
        ],
    )


@pytest.mark.parametrize("cause", ["view_front", "dim_diameter"])
def test_audit_traces_code_through_operation_feature_and_its_evidence(
    cause: str,
) -> None:
    validate_audit_report(
        report(
            ("coding", "ret_bore", "operations", "op_bore"),
            ("operations", "op_bore", "interpretation", "sem_bore"),
            ("interpretation", "sem_bore", "interpretation", cause),
        ),
        snapshot(),
    )


@pytest.mark.parametrize("name", ["view_missing", "dim_missing", "sem_missing"])
def test_missing_members_can_be_added_directly_without_inventing_a_chain(
    name: str,
) -> None:
    validate_audit_report(report(add=name), snapshot())
    with pytest.raises(SubmissionValidationError, match="does not exist"):
        validate_audit_report(report(target=name), snapshot())


def test_a_mistyped_member_suggests_a_close_existing_one() -> None:
    with pytest.raises(
        SubmissionValidationError, match=r"snapshot\. Maybe: sem_bore\?$"
    ):
        validate_audit_report(report(target="sem_bor"), snapshot())


def test_dimension_can_trace_to_its_source_view() -> None:
    validate_audit_report(
        report(("interpretation", "dim_diameter", "interpretation", "view_front")),
        snapshot(),
    )


def test_operation_cannot_skip_its_feature_link_to_a_dimension() -> None:
    with pytest.raises(SubmissionValidationError, match="op_bore.semantics"):
        validate_audit_report(
            report(("operations", "op_bore", "interpretation", "dim_diameter")),
            snapshot(),
        )


def test_interpretation_hops_require_an_explicit_evidence_link() -> None:
    with pytest.raises(SubmissionValidationError, match="not supported"):
        validate_audit_report(
            report(("interpretation", "view_front", "interpretation", "sem_bore")),
            snapshot(),
        )


def test_feature_can_trace_through_a_dimension_to_another_view() -> None:
    current = snapshot()
    data = current.interpretation.model_dump()
    front = data["views"][0]
    data["views"].append(
        {
            **front,
            "name": "view_top",
            "role": "top",
            "file": "top.png",
            "region": {**front["region"], "view": "view_top"},
            "dimensions": front["dimensions"],
        }
    )
    front["dimensions"] = []
    data["views"][1]["dimensions"][0]["region"]["view"] = "view_top"
    current.interpretation = DrawingInterpretation.model_validate(data)

    validate_audit_report(
        report(
            ("coding", "ret_bore", "operations", "op_bore"),
            ("operations", "op_bore", "interpretation", "sem_bore"),
            ("interpretation", "sem_bore", "interpretation", "dim_diameter"),
            ("interpretation", "dim_diameter", "interpretation", "view_top"),
        ),
        current,
    )
    # The two declared links must not become an invented direct reference.
    with pytest.raises(SubmissionValidationError, match="not supported"):
        validate_audit_report(
            report(("interpretation", "sem_bore", "interpretation", "view_top")),
            current,
        )
