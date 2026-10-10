"""Check an audit against committed outputs; report-only rules live in contracts."""

from pathlib import PurePosixPath
from typing import cast

from PIL import Image

from zeroshot.pipeline.messages.manifest import DRAWING_SUFFIXES, read_dxf_frame
from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.stages._base.validate import SubmissionValidationError
from zeroshot.pipeline.stages.audit.contracts import (
    AuditFinding,
    AuditRegion,
    AuditReport,
)
from zeroshot.pipeline.stages.coding.verify import unmatched_items
from zeroshot.pipeline.stages.contracts import ReconstructionSnapshot
from zeroshot.pipeline.stages.interpretation.contracts import DrawingInterpretation
from zeroshot.pipeline.stages.resolve_refs import close_names
from zeroshot.pipeline.stages.tickets.contracts import reported_concerns
from zeroshot.pipeline.stages.types import PipelineStage
from zeroshot.pipeline.verification import AttemptStore, ExecutionStatus


def validate_audit_report(
    report: AuditReport,
    snapshot: ReconstructionSnapshot,
    attempts: AttemptStore | None = None,
    *,
    require_drawing_diff_reviews: bool = False,
) -> None:
    """Check ticket and concern coverage, verification, targets and evidence.

    Evidence is checked against the files themselves, so it needs `attempts`.
    """
    # Acceptance requires a completed, successful build.
    if snapshot.last_completed_stage is not PipelineStage.CODING:
        raise SubmissionValidationError("audit requires a completed coding snapshot")
    _validate_ticket_coverage(report, snapshot)
    _validate_concern_coverage(report, snapshot, require_drawing_diff_reviews)
    if not report.findings and (  # i.e., accepted
        snapshot.verification is None
        or snapshot.verification.exec_report is None
        or snapshot.verification.exec_report.status is not ExecutionStatus.VERIFIED
        or snapshot.verification.exec_report.returncode != 0
    ):
        raise SubmissionValidationError(
            "audit cannot accept a reconstruction without a verified solid; "
            "report the verification failure and its cause"
        )
    # Snapshot validation guarantees an interpretation after coding.
    members = cast(DrawingInterpretation, snapshot.interpretation).members()
    errors: list[str] = []
    if attempts is not None:
        drawn = _drawings_of(snapshot, attempts)
        for finding in report.findings:
            errors.extend(_evidence_errors(finding, attempts.workdir, drawn))
    for finding in report.findings:
        for target in finding.targets:
            if target not in members:
                maybe = close_names(target, members)
                errors.append(
                    f"{finding.name}: target {target!r} is not an interpretation "
                    "member" + (f". Maybe: {', '.join(maybe)}?" if maybe else "")
                )
    if errors:
        raise SubmissionValidationError("\n".join(errors))


def _drawings_of(
    snapshot: ReconstructionSnapshot, attempts: AttemptStore
) -> PurePosixPath | None:
    """Where this verification drew, or None when it drew nothing to measure."""
    verification = snapshot.verification
    if verification is None or verification.verification_id is None:
        return None
    attempt = attempts.sandbox_attempt_dir(
        snapshot.round, "coding", verification.verification_id
    )
    host = attempts.workdir.sandbox_to_host_path(attempt)
    return attempt if any(host.glob("**/projection/*.dxf")) else None


def _evidence_errors(
    finding: AuditFinding, workdir: SandboxWorkdir, drawn: PurePosixPath | None
) -> list[str]:
    """Every region names a readable workspace drawing and lies inside it."""
    errors = []
    measured = False
    for index, region in enumerate(finding.evidence):
        # Whether the file was opened, which a misplaced box does not undo.
        measured |= drawn is not None and _measures(region.file, workdir, drawn)
        if (error := _region_error(region, workdir)) is not None:
            errors.append(f"{finding.name}.evidence[{index}]: {error}")
    if drawn is not None and not measured:
        errors.append(
            f"{finding.name}: cite at least one .dxf under a projection/ "
            f"directory of {drawn}, so the discrepancy is measured on what "
            "this build drew"
        )
    return errors


def _measures(file: str, workdir: SandboxWorkdir, drawn: PurePosixPath) -> bool:
    """A drawing this verification made, not an input or an older attempt."""
    cited = PurePosixPath(file)
    if not cited.is_absolute():
        cited = workdir.sandbox_bind_dir / cited
    return (
        cited.suffix.lower() == ".dxf"
        and cited.is_relative_to(drawn)
        and "projection" in cited.parts
    )


def _region_error(region: AuditRegion, workdir: SandboxWorkdir) -> str | None:
    file = region.file
    suffix = PurePosixPath(file).suffix.lower()
    if suffix not in DRAWING_SUFFIXES:
        return (
            f"{file} is not a drawing; evidence is measured on "
            f"{', '.join(sorted(DRAWING_SUFFIXES))}"
        )
    try:
        path = workdir.sandbox_to_host_path(file)
    except ValueError as error:
        return str(error)
    if not path.is_file():
        return f"{file} does not exist in the workspace"
    if suffix == ".dxf":
        # Its own coordinates: a projection carries the model's, not a sheet's.
        sheet = read_dxf_frame(path)["box_mm"]
    else:
        with Image.open(path) as image:
            sheet = (0, 0, *image.size)
    if any(
        edge < bound - 1e-7 for edge, bound in zip(region.box[:2], sheet[:2])
    ) or any(edge > bound + 1e-7 for edge, bound in zip(region.box[2:], sheet[2:])):
        return f"box {region.box} lies outside {file}, which spans {sheet}"
    return None


def _validate_concern_coverage(
    report: AuditReport,
    snapshot: ReconstructionSnapshot,
    require_drawing_diff_reviews: bool,
) -> None:
    """Every concern this round raised is answered; drawing_diff items when required."""
    verification = snapshot.verification
    concerns = set(reported_concerns(snapshot.stage_reports))
    diff_items = set(
        unmatched_items(verification.drawing_diff_report if verification else None)
    )
    required = concerns | diff_items if require_drawing_diff_reviews else concerns
    reviewed = set(report.concern_reviews)
    if missing := sorted(required - reviewed):
        raise SubmissionValidationError(
            "concern_reviews must answer every stage-report concern"
            + (" and drawing_diff item" if require_drawing_diff_reviews else "")
            + f" this round raises; missing: {', '.join(missing)}"
        )
    if unknown := sorted(reviewed - concerns - diff_items):
        raise SubmissionValidationError(
            "concern_reviews names keys this round does not raise: "
            f"{', '.join(unknown)}. This round raises: "
            f"{', '.join(sorted(required)) or 'none'}"
        )


def _validate_ticket_coverage(
    report: AuditReport,
    snapshot: ReconstructionSnapshot,
) -> None:
    """Every open ticket is reviewed, so no round's work closes unexamined.

    Report validation already links every unsolved review to current findings.
    Here we check those reviews against the snapshot, including on acceptance.
    """
    expected = {ticket.ticket_id for ticket in snapshot.open_tickets}
    reviewed = set(report.ticket_reviews)
    if expected != reviewed:
        raise SubmissionValidationError(
            "ticket_reviews must cover every open ticket exactly once: "
            f"missing={sorted(expected - reviewed)}, "
            f"unexpected={sorted(reviewed - expected)}"
        )
