"""Audit links through the interpretation, including direct missing-member tickets."""

from collections.abc import Iterator
from pathlib import Path

import ezdxf
import pytest
from PIL import Image

from tests.zeroshot.contracts import bootstrap_review, evidence
from tests.zeroshot.workflow.test_resolve_submission import interpretation
from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.stages._base.validate import SubmissionValidationError
from zeroshot.pipeline.stages.audit.contracts import (
    AuditFinding,
    AuditRegion,
    AuditReport,
    CausalHop,
    RevisionRequest,
    StageOutputRef,
)
from zeroshot.pipeline.stages.audit.validate import _region_error, validate_audit_report
from zeroshot.pipeline.stages.coding.verify import VerifyOutputResult
from zeroshot.pipeline.stages.contracts import ReconstructionSnapshot
from zeroshot.pipeline.stages.interpretation.contracts import DrawingInterpretation
from zeroshot.pipeline.stages.operations.contracts import Operation, OperationPlan
from zeroshot.pipeline.stages.tickets.contracts import (
    BootstrapWork,
    Ticket,
    TicketResponse,
)
from zeroshot.pipeline.stages.types import REASONING_STAGES, PipelineStage
from zeroshot.pipeline.verification import AttemptStore, ExecutionStatus


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
                    semantics=["sem_bore"],
                )
            ],
            rationale="One bore.",
        ),
        program_source="ret_bore = object()\nresult = ret_bore\n",
        verification=VerifyOutputResult(
            verification_id="000", status=ExecutionStatus.VERIFIED, returncode=0
        ),
    )


def ref(stage: str, name: str | None) -> StageOutputRef:
    return StageOutputRef(stage=stage, name=name)


def report(
    *hops: tuple[str, str, str, str],
    target: str | None = None,
    add: str | None = None,
    cites: list[AuditRegion] | None = None,
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
        concern_reviews={},
        ticket_reviews=bootstrap_review(),
        findings=[
            AuditFinding(
                name="find_bore",
                observation="The drawn bore is missing or incorrect.",
                evidence=cites or evidence("front.png"),
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
            "v_axis": "+y",
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


DRAWN = "attempts/round_000/coding/000/projection/front.dxf"


def _draw_square(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = ezdxf.new()
    document.modelspace().add_lwpolyline(
        [(0, 0), (20, 0), (20, 20), (0, 20)], close=True
    )
    document.saveas(path)


@pytest.fixture
def workspace() -> Iterator[AttemptStore]:
    """A 20x20 input raster, and a 20x20 projection of what this build drew."""
    with SandboxWorkdir() as workdir:
        Image.new("RGB", (20, 20), "white").save(workdir.host_bind_dir / "front.png")
        attempts = AttemptStore(workdir, lambda: 0)
        attempts.issue("coding")
        _draw_square(workdir.host_bind_dir / DRAWN)
        yield attempts


def cite(file: str, box: tuple[float, float, float, float]) -> AuditRegion:
    return AuditRegion(file=file, box=box)


def test_evidence_is_measured_on_the_files_it_names(
    workspace: AttemptStore,
) -> None:
    validate_audit_report(
        report(
            cites=[
                cite("front.png", (2, 2, 8, 8)),
                cite(DRAWN, (2.0, 2.0, 8.0, 8.0)),
            ],
            target="sem_bore",
        ),
        snapshot(),
        workspace,
    )


@pytest.mark.parametrize(
    ("cited", "message"),
    [
        (cite(DRAWN.replace("front", "absent"), (0, 0, 1, 1)), "does not exist"),
        (cite(DRAWN, (0, 0, 21, 1)), "lies outside"),
        (cite("../escaped/" + DRAWN, (0, 0, 1, 1)), "must not escape"),
        (cite(DRAWN.replace(".dxf", ".step"), (0, 0, 1, 1)), "is not a drawing"),
    ],
)
def test_evidence_that_cannot_be_opened_and_measured_is_refused(
    workspace: AttemptStore, cited: AuditRegion, message: str
) -> None:
    with pytest.raises(SubmissionValidationError, match=message):
        validate_audit_report(
            report(cites=[cited], target="sem_bore"), snapshot(), workspace
        )


def test_a_finding_measures_at_least_one_projection_dxf(
    workspace: AttemptStore,
) -> None:
    with pytest.raises(SubmissionValidationError, match="under a projection/"):
        validate_audit_report(
            report(cites=[cite("front.png", (0, 0, 10, 10))], target="sem_bore"),
            snapshot(),
            workspace,
        )


def test_an_audit_of_a_build_that_drew_nothing_still_reports_it() -> None:
    with SandboxWorkdir() as workdir:
        Image.new("RGB", (20, 20), "white").save(workdir.host_bind_dir / "front.png")
        attempts = AttemptStore(workdir, lambda: 0)
        attempts.issue("coding")
        validate_audit_report(
            report(cites=[cite("front.png", (0, 0, 10, 10))], target="sem_bore"),
            snapshot(),
            attempts,
        )


def test_a_projection_of_an_earlier_attempt_does_not_measure_this_build(
    workspace: AttemptStore,
) -> None:
    """Its shapes are a solid this audit is not reviewing."""
    stale = DRAWN.replace("coding/000", "coding/001")
    _draw_square(workspace.workdir.host_bind_dir / stale)
    with pytest.raises(SubmissionValidationError, match="coding/000"):
        validate_audit_report(
            report(cites=[cite(stale, (2.0, 2.0, 8.0, 8.0))], target="sem_bore"),
            snapshot(),
            workspace,
        )


def test_a_misplaced_box_does_not_also_demand_the_projection_it_cites(
    workspace: AttemptStore,
) -> None:
    """Citing the drawing and measuring it wrongly are two different mistakes."""
    with pytest.raises(SubmissionValidationError) as refusal:
        validate_audit_report(
            report(cites=[cite(DRAWN, (0.0, 0.0, 21.0, 1.0))], target="sem_bore"),
            snapshot(),
            workspace,
        )

    assert "lies outside" in str(refusal.value)
    assert "cite at least one" not in str(refusal.value)


@pytest.mark.parametrize("folder", ["projection", "render_3d"])
def test_generated_png_evidence_is_allowed_in_integer_pixels(folder):
    with SandboxWorkdir() as workdir:
        directory = workdir.host_bind_dir / folder
        directory.mkdir()
        Image.new("RGB", (1248, 480), "white").save(directory / "front.png")
        region = AuditRegion(file=f"{folder}/front.png", box=(0, 12, 52, 20))
        assert _region_error(region, workdir) is None
