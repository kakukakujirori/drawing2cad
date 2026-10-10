"""Audit targets and evidence, checked against the interpretation and the workspace."""

from collections.abc import Iterator
from dataclasses import replace
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
    ConcernReview,
)
from zeroshot.pipeline.stages.audit.validate import _region_error, validate_audit_report
from zeroshot.pipeline.stages.coding.verify import VerifyOutputResult
from zeroshot.pipeline.stages.contracts import ReconstructionSnapshot
from zeroshot.pipeline.stages.tickets.contracts import (
    BootstrapWork,
    Ticket,
    TicketResponse,
)
from zeroshot.pipeline.stages.types import REASONING_STAGES, PipelineStage
from zeroshot.pipeline.verification import AttemptStore, ExecutionStatus
from zeroshot.pipeline.verification.run_cadquery import CadQueryExecutionReport
from zeroshot.pipeline.verification.run_drawing_diff import DrawingDiffReport


def snapshot() -> ReconstructionSnapshot:
    return ReconstructionSnapshot(
        open_tickets=[
            Ticket(
                ticket_id="ticket_initial",
                subject=BootstrapWork(instruction="Reconstruct"),
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
        program_source="result = object()\n",
        verification=VerifyOutputResult(
            verification_id="000",
            exec_report=CadQueryExecutionReport(
                status=ExecutionStatus.VERIFIED, returncode=0
            ),
        ),
    )


def report(
    target: str = "sem_bore",
    cites: list[AuditRegion] | None = None,
) -> AuditReport:
    return AuditReport(
        concern_reviews={},
        ticket_reviews=bootstrap_review(),
        findings=[
            AuditFinding(
                name="find_bore",
                observation="The drawn bore is missing or incorrect.",
                evidence=cites or evidence("front.png"),
                cause="interpretation",
                targets=[target],
                revision_request="The bore is wrong.",
                related_ticket_ids=[],
            )
        ],
    )


@pytest.mark.parametrize("target", ["view_front", "dim_diameter", "sem_bore", "datum"])
def test_a_target_names_an_existing_member(target: str) -> None:
    validate_audit_report(report(target), snapshot())


@pytest.mark.parametrize("target", ["view_missing", "dim_missing", "sem_missing"])
def test_a_target_the_interpretation_lacks_is_refused(target: str) -> None:
    with pytest.raises(SubmissionValidationError, match="not an interpretation member"):
        validate_audit_report(report(target), snapshot())


def test_a_mistyped_member_suggests_a_close_existing_one() -> None:
    with pytest.raises(SubmissionValidationError, match=r"Maybe: sem_bore\?$"):
        validate_audit_report(report("sem_bor"), snapshot())


def test_every_listed_drawing_diff_group_needs_an_answer() -> None:
    group = {
        "direction": "missing",
        "kind": "lines",
        "size_px": 80,
        "box_px": [1, 2, 3, 4],
        "color": "red",
    }
    diff = DrawingDiffReport(
        drawing_path=Path("front.png"),
        projection_path=None,
        stats={"unmatched": [group]},
    )
    base = snapshot()
    assert base.verification is not None
    verification = replace(base.verification, drawing_diff_report={"view_front": diff})
    listed = base.model_copy(update={"verification": verification})
    unanswered = report()
    answer = ConcernReview(
        finding_name="find_bore", disposition="It is the missing bore."
    )
    answered = unanswered.model_copy(
        update={"concern_reviews": {"drawing_diff.view_front.1": answer}}
    )
    invented = unanswered.model_copy(
        update={"concern_reviews": {"drawing_diff.view_front.2": answer}}
    )

    with pytest.raises(
        SubmissionValidationError, match="missing: drawing_diff.view_front.1"
    ):
        validate_audit_report(unanswered, listed, require_drawing_diff_reviews=True)
    validate_audit_report(answered, listed, require_drawing_diff_reviews=True)
    # Without the duty, an answer is optional but must still name a listed item.
    validate_audit_report(unanswered, listed)
    validate_audit_report(answered, listed)
    with pytest.raises(SubmissionValidationError, match="does not raise"):
        validate_audit_report(invented, listed)


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
        validate_audit_report(report(cites=[cited]), snapshot(), workspace)


def test_a_finding_measures_at_least_one_projection_dxf(
    workspace: AttemptStore,
) -> None:
    with pytest.raises(SubmissionValidationError, match="under a projection/"):
        validate_audit_report(
            report(cites=[cite("front.png", (0, 0, 10, 10))]),
            snapshot(),
            workspace,
        )


def test_an_audit_of_a_build_that_drew_nothing_still_reports_it() -> None:
    with SandboxWorkdir() as workdir:
        Image.new("RGB", (20, 20), "white").save(workdir.host_bind_dir / "front.png")
        attempts = AttemptStore(workdir, lambda: 0)
        attempts.issue("coding")
        validate_audit_report(
            report(cites=[cite("front.png", (0, 0, 10, 10))]),
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
            report(cites=[cite(stale, (2.0, 2.0, 8.0, 8.0))]),
            snapshot(),
            workspace,
        )


def test_a_misplaced_box_does_not_also_demand_the_projection_it_cites(
    workspace: AttemptStore,
) -> None:
    """Citing the drawing and measuring it wrongly are two different mistakes."""
    with pytest.raises(SubmissionValidationError) as refusal:
        validate_audit_report(
            report(cites=[cite(DRAWN, (0.0, 0.0, 21.0, 1.0))]),
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
