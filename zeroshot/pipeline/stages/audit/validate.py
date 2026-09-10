from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from typing import cast

from zeroshot.pipeline.messages.contracts import (
    DrawingSource,
    Operation,
)
from zeroshot.pipeline.messages.contracts.audit import (
    AuditFinding,
    AuditReport,
    CausalHop,
    StageOutputRef,
)
from zeroshot.pipeline.messages.contracts.reconstruction import ReconstructionSnapshot
from zeroshot.pipeline.stages._base.validate import SubmissionValidationError
from zeroshot.pipeline.stages.types import (
    REASONING_STAGES,
    PipelineStage,
    ReasoningStage,
)
from zeroshot.pipeline.verification import ExecutionStatus
from zeroshot.pipeline.verification.check_program import program_output_names


def validate_audit_report(
    report: AuditReport,
    snapshot: ReconstructionSnapshot,
) -> None:
    """Reject an audit report that contradicts the audited stage outputs."""
    if snapshot.last_completed_stage is not PipelineStage.CODING:
        raise SubmissionValidationError("audit requires a completed coding snapshot")
    if report.accepted and (
        snapshot.verification is None
        or snapshot.verification.status is not ExecutionStatus.VERIFIED
        or snapshot.verification.returncode != 0
    ):
        raise SubmissionValidationError(
            "audit cannot accept a reconstruction without a verified solid; "
            "report the verification failure and the root that must change"
        )
    drawing = cast(DrawingSource, snapshot.drawings)

    references = tuple(_iter_references(report.findings))
    semantic_names = (
        {feature.name for feature in snapshot.semantics.proposal}
        if snapshot.semantics is not None
        else set()
    )
    operations_by_name = (
        {operation.name: operation for operation in snapshot.operations.proposal}
        if snapshot.operations is not None
        else {}
    )
    coding_names, coding_error = _inspect_coding_outputs(
        snapshot.program_source,
        references,
    )

    known_members = {
        PipelineStage.DRAWINGS: {sheet.name for sheet in drawing.sheets},
        PipelineStage.SEMANTICS: semantic_names,
        PipelineStage.OPERATIONS: set(operations_by_name),
        PipelineStage.CODING: coding_names,
    }
    errors = [coding_error] if coding_error is not None else []
    sheet_by_evidence = {
        entry.name: sheet.name
        for sheet in drawing.sheets
        for entry in (*sheet.evidence, *sheet.dimensions)
    }
    cited_sheets = {
        feature.name: {
            sheet_by_evidence[name]
            for name in feature.evidence
            if name in sheet_by_evidence
        }
        for feature in (snapshot.semantics.proposal if snapshot.semantics else [])
    }
    errors.extend(_missing_reference_errors(references, known_members))
    for finding in report.findings:
        error = _within_stage_walk_error(finding)
        if error is not None:
            errors.append(error)
    for hop in _iter_hops(report.findings):
        error = _causal_hop_error(
            hop,
            known_members=known_members,
            operations_by_name=operations_by_name,
            cited_sheets=cited_sheets,
        )
        if error is not None:
            errors.append(error)

    if errors:
        # A repeated reference or hop should not make the model repair the
        # same mechanical contradiction more than once.
        unique_errors = list(dict.fromkeys(errors))
        raise SubmissionValidationError("\n".join(unique_errors))


def _iter_references(
    findings: Iterable[AuditFinding],
) -> Iterator[StageOutputRef]:
    """Every existing output that an audit report claims to address."""
    for finding in findings:
        for hop in finding.backtrace:
            yield hop.effect
            yield hop.cause
        yield from finding.revision_request.targets


def _iter_hops(findings: Iterable[AuditFinding]) -> Iterator[CausalHop]:
    """Yield causal hops in report order."""
    for finding in findings:
        yield from finding.backtrace


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
            errors.append(
                f"{reference.stage} member {reference.name!r} does not exist "
                "in the audited snapshot"
            )
    return errors


def _within_stage_walk_error(finding: AuditFinding) -> str | None:
    """Refuse a backtrace that walks a stage instead of crossing out of it.

    A hop between two members of one stage claims the effect is wrong because
    the cause is, which for a chain of them restates the dependencies the
    artifact already declares. One run's auditor reached every revision target
    by starting at the program's last output and stepping back through all
    twenty in order, one rationale each; every hop passed, and the path said
    nothing the target had not. What a backtrace is for is the step out of the
    stage where the defect shows into the stage it comes from, so each stage
    gets one hop inside it: the one that names the member to blame.
    """
    walked: dict[ReasoningStage, list[str]] = defaultdict(list)
    for hop in finding.backtrace:
        if hop.effect.stage == hop.cause.stage:
            walked[hop.effect.stage].append(f"{hop.effect.name} -> {hop.cause.name}")

    # Counted per stage, not over the path: a backtrace that crosses all stages
    # is entitled to its one naming hop in each of them.
    overwalked = [
        f"{stage} {len(steps)} times ({', '.join(steps)})"
        for stage, steps in walked.items()
        if len(steps) > 1
    ]
    if not overwalked:
        return None
    return (
        f"{finding.name} steps between members of one stage more than once: "
        f"{'; '.join(overwalked)}. Name the member the defect comes from in one "
        "hop per stage and request the revision there; the operations it is "
        "consumed by afterwards are not separate causes."
    )


def _causal_hop_error(
    hop: CausalHop,
    *,
    known_members: Mapping[PipelineStage, set[str]],
    operations_by_name: Mapping[str, Operation],
    cited_sheets: Mapping[str, set[str]],
) -> str | None:
    """Validate only causal relations represented by an explicit contract."""
    effect = hop.effect
    cause = hop.cause

    distance = REASONING_STAGES.index(effect.stage) - REASONING_STAGES.index(
        cause.stage
    )
    if distance not in (0, 1):
        return (
            f"{effect.stage}-to-{cause.stage} hop must stay within one stage "
            "or move to the adjacent upstream stage: coding -> operations -> "
            "semantics -> drawings"
        )

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
        operation = operations_by_name[effect.name]
        if cause.name not in operation.depends_on:
            return (
                f"operations hop {effect.name!r} -> {cause.name!r} is "
                f"not supported by {effect.name}.depends_on"
            )

    elif (
        effect.stage is PipelineStage.OPERATIONS
        and cause.stage is PipelineStage.SEMANTICS
    ):
        operation = operations_by_name[effect.name]
        if cause.name not in operation.semantics:
            return (
                f"operations-to-semantics hop {effect.name!r} -> "
                f"{cause.name!r} is not supported by {effect.name}.semantics"
            )

    elif (
        effect.stage is PipelineStage.SEMANTICS
        and cause.stage is PipelineStage.DRAWINGS
    ):
        if cause.name not in cited_sheets[effect.name]:
            return (
                f"semantics-to-drawings hop {effect.name!r} -> {cause.name!r} "
                f"is not supported by {effect.name}.evidence: name the sheet "
                "owning a cited ev_ or dim_ entry. For an omission with no "
                "existing citation, report the defect directly at its root."
            )

    # No machine-readable relation currently exists for coding-internal or
    # semantics-internal reasoning. Such hops remain valid once both members
    # are known rather than being rejected on a guess.
    return None
