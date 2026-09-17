"""Check an audit against committed outputs; report-only rules live in contracts."""

from collections.abc import Iterable, Iterator, Mapping
from typing import cast

from zeroshot.pipeline.stages._base.validate import SubmissionValidationError
from zeroshot.pipeline.stages.audit.contracts import (
    AuditFinding,
    AuditReport,
    CausalHop,
    StageOutputRef,
)
from zeroshot.pipeline.stages.contracts import ReconstructionSnapshot
from zeroshot.pipeline.stages.interpretation.contracts import DrawingInterpretation
from zeroshot.pipeline.stages.operations.contracts import Operation
from zeroshot.pipeline.stages.resolve_refs import close_names
from zeroshot.pipeline.stages.types import PipelineStage
from zeroshot.pipeline.verification import ExecutionStatus
from zeroshot.pipeline.verification.check_program import program_output_names


def validate_audit_report(
    report: AuditReport,
    snapshot: ReconstructionSnapshot,
) -> None:
    """Check ticket coverage, verification, members and declared causal links."""
    # Acceptance requires a completed, successful build.
    if snapshot.last_completed_stage is not PipelineStage.CODING:
        raise SubmissionValidationError("audit requires a completed coding snapshot")
    _validate_ticket_coverage(report, snapshot)
    if report.accepted and (
        snapshot.verification is None
        or snapshot.verification.status is not ExecutionStatus.VERIFIED
        or snapshot.verification.returncode != 0
    ):
        raise SubmissionValidationError(
            "audit cannot accept a reconstruction without a verified solid; "
            "report the verification failure and the root that must change"
        )
    # Index the committed members and the interpretation's explicit sources.
    # Snapshot validation guarantees interpretation and operations after coding.
    interpretation = cast(DrawingInterpretation, snapshot.interpretation)
    references = tuple(_iter_references(report.findings))
    operations_by_name = (
        {operation.name: operation for operation in snapshot.operations.proposal}
        if snapshot.operations is not None
        else {}
    )
    coding_names, coding_error = _inspect_coding_outputs(
        snapshot.program_source,
        references,
    )

    interpretation_links = {
        name: member.cites for name, member in interpretation.members().items()
    }
    known_members = {
        PipelineStage.INTERPRETATION: set(interpretation_links),
        PipelineStage.OPERATIONS: set(operations_by_name),
        PipelineStage.CODING: coding_names,
    }
    errors = [coding_error] if coding_error is not None else []

    # Existing references must resolve; proposed identities must not collide.
    errors.extend(_missing_reference_errors(references, known_members))
    errors.extend(_proposed_name_errors(report.findings, known_members))

    # Path shape is checked by AuditFinding; here each hop must match its source.
    for finding in report.findings:
        for hop in finding.backtrace:
            error = _causal_hop_error(
                hop,
                known_members=known_members,
                operations_by_name=operations_by_name,
                interpretation_links=interpretation_links,
            )
            if error is not None:
                errors.append(error)

    if errors:
        # A repeated reference or hop should not make the model repair the
        # same mechanical contradiction more than once.
        unique_errors = list(dict.fromkeys(errors))
        raise SubmissionValidationError("\n".join(unique_errors))


def _validate_ticket_coverage(
    report: AuditReport,
    snapshot: ReconstructionSnapshot,
) -> None:
    """Only current defect tickets are reviewed; bootstrap is a one-round order.

    Report validation already links every unsolved review to current findings.
    Here we check those reviews against the snapshot, including on acceptance.
    """
    expected = {
        ticket.ticket_id
        for ticket in snapshot.open_tickets
        if isinstance(ticket.subject, AuditFinding)
    }
    reviewed = {review.ticket_id for review in report.ticket_reviews}
    if expected != reviewed:
        raise SubmissionValidationError(
            "ticket_reviews must cover every current defect ticket exactly once "
            "and exclude bootstrap work: "
            f"missing={sorted(expected - reviewed)}, "
            f"unexpected={sorted(reviewed - expected)}"
        )


def _iter_references(
    findings: Iterable[AuditFinding],
) -> Iterator[StageOutputRef]:
    """Every existing output that an audit report claims to address."""
    for finding in findings:
        for hop in finding.backtrace:
            yield hop.effect
            yield hop.cause
        yield from finding.revision_request.targets


def _inspect_coding_outputs(
    program_source: str | None,
    references: Iterable[StageOutputRef],
) -> tuple[set[str], str | None]:
    """Return named code outputs and any syntax failure that hides them."""
    needs_named_outputs = any(
        reference.stage is PipelineStage.CODING and reference.name is not None
        for reference in references
    )
    if not needs_named_outputs or program_source is None:
        return set(), None

    try:
        return program_output_names(program_source), None
    except SyntaxError as error:
        location = (
            f"line {error.lineno}" if error.lineno is not None else "an unknown line"
        )
        return set(), (
            "named coding outputs cannot be checked because model.py has "
            f"invalid syntax at {location}"
        )


def _missing_reference_errors(
    references: Iterable[StageOutputRef],
    known_members: Mapping[PipelineStage, set[str]],
) -> list[str]:
    """Report absent named outputs; the report combines duplicate errors."""
    errors: list[str] = []
    for reference in references:
        if (
            reference.name is not None
            and reference.name not in known_members[reference.stage]
        ):
            maybe = close_names(reference.name, known_members[reference.stage])
            errors.append(
                f"{reference.stage} member {reference.name!r} does not exist "
                "in the audited snapshot"
                + (f". Maybe: {', '.join(maybe)}?" if maybe else "")
            )
    return errors


def _proposed_name_errors(
    findings: Iterable[AuditFinding],
    known_members: Mapping[PipelineStage, set[str]],
) -> list[str]:
    """Only split/merge may retain an existing identity among their own targets."""
    errors = []
    for finding in findings:
        request = finding.revision_request
        stage = request.targets[0].stage
        retained = (
            {target.name for target in request.targets}
            if request.action in {"split", "merge"}
            else set()
        )
        collisions = (set(request.proposed_names) & known_members[stage]) - retained
        for name in sorted(collisions):
            errors.append(
                f"{finding.name}: proposed {stage} name {name!r} already exists "
                "in the audited snapshot and is not a retained split/merge target"
            )
    return errors


def _causal_hop_error(
    hop: CausalHop,
    *,
    known_members: Mapping[PipelineStage, set[str]],
    operations_by_name: Mapping[str, Operation],
    interpretation_links: Mapping[str, frozenset[str]],
) -> str | None:
    """Validate only causal relations represented by an explicit contract."""
    effect = hop.effect
    cause = hop.cause

    # A whole-stage reference has no member identity with which to prove a
    # direct relation. Its existence was already checked above.
    if effect.name is None or cause.name is None:
        return None
    if (
        effect.name not in known_members[effect.stage]
        or cause.name not in known_members[cause.stage]
    ):
        return None

    if effect.stage is PipelineStage.CODING and cause.stage is PipelineStage.OPERATIONS:
        expected_return = f"ret_{cause.name.removeprefix('op_')}"
        if effect.name != expected_return:
            return (
                f"coding-to-operations hop {effect.name!r} -> "
                f"{cause.name!r} must use {expected_return!r}"
            )

    elif (
        effect.stage is PipelineStage.OPERATIONS
        and cause.stage is PipelineStage.OPERATIONS
    ):
        # Each operation changes what the earlier ones left, so any earlier
        # operation can be the cause and a later one cannot.
        order = list(operations_by_name)
        if order.index(cause.name) >= order.index(effect.name):
            return (
                f"operations hop {effect.name!r} -> {cause.name!r} must name "
                f"an operation listed before {effect.name}"
            )

    elif (
        effect.stage is PipelineStage.OPERATIONS
        and cause.stage is PipelineStage.INTERPRETATION
    ):
        operation = operations_by_name[effect.name]
        if cause.name not in operation.semantics:
            return (
                f"operations-to-interpretation hop {effect.name!r} -> "
                f"{cause.name!r} is not supported by {effect.name}.semantics"
            )

    elif (
        effect.stage is PipelineStage.INTERPRETATION
        and cause.stage is PipelineStage.INTERPRETATION
    ):
        if cause.name not in interpretation_links[effect.name]:
            return (
                f"interpretation hop {effect.name!r} -> {cause.name!r} "
                "is not supported by the member's evidence, dimension_refs or "
                "view region. For an omission with no existing citation, "
                "report the defect directly at its root."
            )

    # Coding-internal dependencies have no machine-readable contract here.
    return None
