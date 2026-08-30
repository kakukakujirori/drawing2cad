"""Contextual validation of stage outputs against a reconstruction snapshot."""

import re
from collections.abc import Iterable, Iterator, Mapping, Sequence

from zeroshot.pipeline.messages.contracts import (
    Operation,
    OperationPlan,
    SemanticHypothesis,
)
from zeroshot.pipeline.messages.contracts.audit import (
    AuditReport,
    Backtrace,
    CausalHop,
    StageOutputRef,
)
from zeroshot.pipeline.messages.contracts.reconstruction import (
    CodingSubmission,
    OperationSubmission,
    ReconstructionSnapshot,
    SemanticSubmission,
    StageSubmission,
    Ticket,
    TicketResponse,
)
from zeroshot.pipeline.messages.contracts.stages import (
    REASONING_STAGES,
    PipelineStage,
    ReasoningStage,
    next_stage,
)
from zeroshot.pipeline.verification import (
    ExecutionStatus,
    VerifyOutputResult,
    check_program,
)

type Deliverable = (
    SemanticSubmission | OperationSubmission | CodingSubmission | AuditReport
)

_REFERENCE_LIKE = re.compile(r"\bsem_[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+\b")
_COPIED_DECIMALS = 4
_DECIMAL = re.compile(rf"\d+\.\d{{{_COPIED_DECIMALS},}}")
_NAMED_AT_MOST = 3


class DeliverableValidationError(ValueError):
    """A stage output contradicts the current reconstruction snapshot."""


def validate_deliverable(
    output: Deliverable,
    snapshot: ReconstructionSnapshot,
    *,
    verification: VerifyOutputResult | None = None,
) -> None:
    """Validate one stage output as the next artifact of the current round."""
    if isinstance(output, AuditReport):
        if verification is not None:
            raise DeliverableValidationError(
                "audit does not accept a separate verification"
            )
        _validate_audit_report(output, snapshot)
        return

    if not isinstance(output, StageSubmission):
        raise TypeError(f"unsupported deliverable type: {type(output).__name__}")

    stage = _next_reasoning_stage(snapshot.last_completed_stage)
    _validate_ticket_responses(
        output.responses,
        snapshot.open_tickets,
        expected_stage=stage,
    )
    if stage is not PipelineStage.CODING and verification is not None:
        raise DeliverableValidationError(f"{stage} must not submit verification")

    match stage:
        case PipelineStage.SEMANTICS:
            if not isinstance(output.deliverable, SemanticHypothesis):
                raise DeliverableValidationError(
                    "semantics must submit a SemanticHypothesis"
                )
        case PipelineStage.OPERATIONS:
            if not isinstance(output.deliverable, OperationPlan):
                raise DeliverableValidationError(
                    "operations must submit an OperationPlan"
                )
            _validate_operations(output.deliverable, snapshot)
        case PipelineStage.CODING:
            if output.deliverable is not None:
                raise DeliverableValidationError(
                    "coding must submit null because model.py is written "
                    "through the workspace"
                )
            _validate_coding(snapshot, verification)
        case _:
            raise DeliverableValidationError(f"unexpected reasoning stage: {stage}")


def _next_reasoning_stage(
    completed_stage: ReasoningStage | None,
) -> ReasoningStage:
    stage = next_stage(completed_stage)
    if stage not in REASONING_STAGES:
        raise DeliverableValidationError(
            "a completed coding snapshot accepts only an AuditReport"
        )
    return stage


################################################################


def _validate_ticket_responses(
    responses: Sequence[TicketResponse],
    tickets: Sequence[Ticket],
    *,
    expected_stage: ReasoningStage,
) -> None:
    """Require exactly one response for every current ticket, in any order."""
    expected_ids = {ticket.ticket_id for ticket in tickets}
    response_ids = [response.ticket_id for response in responses]
    submitted_ids = set(response_ids)

    duplicated = sorted(
        ticket_id for ticket_id in submitted_ids if response_ids.count(ticket_id) > 1
    )
    missing = sorted(expected_ids - submitted_ids)
    unknown = sorted(submitted_ids - expected_ids)
    wrong_stage = sorted(
        response.ticket_id for response in responses if response.stage != expected_stage
    )

    errors: list[str] = []
    if duplicated:
        errors.append("duplicate ticket responses: " + ", ".join(duplicated))
    if missing:
        errors.append("missing ticket responses: " + ", ".join(missing))
    if unknown:
        errors.append("unknown ticket responses: " + ", ".join(unknown))
    if wrong_stage:
        errors.append(
            f"ticket responses must belong to {expected_stage}: "
            + ", ".join(wrong_stage)
        )

    if errors:
        raise DeliverableValidationError("\n".join(errors))


################################################################


def _validate_operations(
    operations: OperationPlan,
    snapshot: ReconstructionSnapshot,
) -> None:
    if snapshot.semantics is None:
        raise DeliverableValidationError(
            "operations requires an integrated SemanticHypothesis"
        )

    # The submitted plan is the operations candidate; the snapshot contains
    # the semantics already integrated earlier in this same round.
    errors = _operation_plan_errors(operations, snapshot.semantics)
    if errors:
        raise DeliverableValidationError(" ".join(errors))


def _operation_plan_errors(
    plan: OperationPlan,
    hypothesis: SemanticHypothesis,
) -> list[str]:
    """Cross-stage contradictions that neither artifact can check alone."""
    established = {feature.name for feature in hypothesis.proposal}
    built = {
        semantic for operation in plan.proposal for semantic in operation.semantics
    }
    valid_references = _semantic_parameter_addresses(hypothesis)
    held_numbers = _hypothesis_numbers(hypothesis)
    errors: list[str] = []

    if uncovered := sorted(established - built):
        named = ", ".join(uncovered)
        errors.append(
            f"The hypothesis establishes {named}, and no operation in the plan "
            "builds them. Add the operations they take, or say in the rationale "
            "why the part is complete without them."
        )

    if unknown := sorted(built - established):
        named = ", ".join(unknown)
        errors.append(
            f"The plan cites {named}, which the hypothesis does not contain. "
            "Cite the features it does have."
        )

    for operation in sorted(plan.proposal, key=lambda item: item.name):
        copied = _transcribed_numbers(operation.detail, held_numbers)
        if not copied:
            continue
        named = ", ".join(copied[:_NAMED_AT_MOST])
        if len(copied) > _NAMED_AT_MOST:
            named += f" and {len(copied) - _NAMED_AT_MOST} more"
        errors.append(
            f"{operation.name} writes out {named}, which the hypothesis already "
            "holds. Cite it as sem_<feature>.geo_<claim>.<parameter> or "
            "sem_<feature>.ev_<reading>.<parameter> instead; the number is put "
            "in for you, and a number retyped is a number that can be mistyped."
        )

    for operation in sorted(plan.proposal, key=lambda item: item.name):
        unresolved = [
            match[0]
            for match in _REFERENCE_LIKE.finditer(operation.detail)
            if match[0] not in valid_references
        ]
        if unresolved:
            named = ", ".join(unresolved)
            errors.append(
                f"{operation.name} refers to {named}, which does not identify "
                "exactly one parameter in the hypothesis. Use the member name "
                "shown there, such as sem_main_bore.geo_cylinder.radius or "
                "sem_main_bore.ev_front_circle.center."
            )

    return errors


def _semantic_parameter_addresses(hypothesis: SemanticHypothesis) -> set[str]:
    return {
        f"{feature.name}.{member.name}.{parameter.name.value}"
        for feature in hypothesis.proposal
        for member in (*feature.geometry, *feature.evidence)
        for parameter in member.parameters
    }


def _hypothesis_numbers(hypothesis: SemanticHypothesis) -> list[float]:
    return [
        number
        for feature in hypothesis.proposal
        for member in (*feature.geometry, *feature.evidence)
        for parameter in member.parameters
        for number in parameter.values
    ]


def _transcribed_numbers(detail: str, held_numbers: Sequence[float]) -> list[str]:
    """Numbers copied at high precision from the semantic hypothesis."""
    copied: list[str] = []
    for literal in _DECIMAL.findall(detail):
        places = len(literal.split(".")[1])
        if any(round(number, places) == float(literal) for number in held_numbers):
            copied.append(literal)
    return copied


################################################################


def _validate_coding(
    snapshot: ReconstructionSnapshot,
    verification: VerifyOutputResult | None,
) -> None:
    if verification is None:
        raise DeliverableValidationError(
            "coding requires a terminal verification result"
        )
    if verification.status is ExecutionStatus.UNINITIALIZED:
        raise DeliverableValidationError("coding verification must be terminal")
    if snapshot.operations is None:
        raise DeliverableValidationError("coding requires an integrated OperationPlan")

    # A missing or syntactically invalid source is already represented by a
    # terminal verification failure and must remain auditable. When readable
    # source exists, reject plan-to-code identity drift as early as possible.
    if verification.source is None:
        return
    try:
        program_check = check_program(verification.source, snapshot.operations)
    except SyntaxError:
        return
    if not program_check.sound:
        raise DeliverableValidationError(
            "model.py does not match the current OperationPlan: "
            f"missing={program_check.missing_operations}, "
            f"unknown={program_check.unknown_operations}, "
            f"result_assigned={program_check.result_assigned}"
        )


################################################################


def _validate_audit_report(
    report: AuditReport,
    snapshot: ReconstructionSnapshot,
) -> None:
    """Reject an audit report that contradicts the audited stage outputs."""
    if snapshot.last_completed_stage is not PipelineStage.CODING:
        raise DeliverableValidationError("audit requires a completed coding snapshot")

    backtraces = tuple(
        backtrace for finding in report.findings for backtrace in finding.backtraces
    )
    references = tuple(_iter_references(backtraces))
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
        snapshot.operations,
        references,
    )

    known_members = {
        PipelineStage.SEMANTICS: semantic_names,
        PipelineStage.OPERATIONS: set(operations_by_name),
        PipelineStage.CODING: coding_names,
    }
    errors = [coding_error] if coding_error is not None else []
    errors.extend(_missing_reference_errors(references, known_members))
    for hop in _iter_hops(backtraces):
        error = _causal_hop_error(
            hop,
            known_members=known_members,
            operations_by_name=operations_by_name,
        )
        if error is not None:
            errors.append(error)

    if errors:
        # A repeated reference or hop should not make the model repair the
        # same mechanical contradiction more than once.
        unique_errors = list(dict.fromkeys(errors))
        raise DeliverableValidationError("\n".join(unique_errors))


def _iter_references(
    backtraces: Iterable[Backtrace],
) -> Iterator[StageOutputRef]:
    """Every existing output that an audit report claims to address."""
    for backtrace in backtraces:
        for hop in backtrace.hops:
            yield hop.effect
            yield hop.cause
        yield from backtrace.revision_request.targets


def _iter_hops(backtraces: Iterable[Backtrace]) -> Iterator[CausalHop]:
    """Yield causal hops in report order."""
    for backtrace in backtraces:
        yield from backtrace.hops


def _inspect_coding_outputs(
    program_source: str | None,
    operations: OperationPlan | None,
    references: Iterable[StageOutputRef],
) -> tuple[set[str], str | None]:
    """Return named code outputs and any syntax failure that hides them."""
    needs_named_outputs = any(
        reference.stage is PipelineStage.CODING and reference.name is not None
        for reference in references
    )
    if not needs_named_outputs or program_source is None or operations is None:
        return set(), None

    try:
        return _program_output_names(program_source, operations), None
    except SyntaxError as error:
        location = (
            f"line {error.lineno}" if error.lineno is not None else "an unknown line"
        )
        return set(), (
            "named coding outputs cannot be checked because model.py has "
            f"invalid syntax at {location}"
        )


def _program_output_names(
    source: str,
    operations: OperationPlan,
) -> set[str]:
    """The module-level ``ret_*`` and ``result`` names established by source."""
    program_check = check_program(source, operations)
    expected_operations = {operation.name for operation in operations.proposal}
    implemented_operations = (
        expected_operations - set(program_check.missing_operations)
    ) | set(program_check.unknown_operations)
    names = {f"ret_{name.removeprefix('op_')}" for name in implemented_operations}
    if program_check.result_assigned:
        names.add("result")
    return names


def _missing_reference_errors(
    references: Iterable[StageOutputRef],
    known_members: Mapping[PipelineStage, set[str]],
) -> list[str]:
    """Report each absent named output once, in first-reference order."""
    errors: list[str] = []
    seen: set[tuple[ReasoningStage, str | None]] = set()
    for reference in references:
        key = (reference.stage, reference.name)
        if key in seen:
            continue
        seen.add(key)

        if (
            reference.name is not None
            and reference.name not in known_members[reference.stage]
        ):
            errors.append(
                f"{reference.stage} member {reference.name!r} does not exist "
                "in the audited snapshot"
            )
    return errors


def _causal_hop_error(
    hop: CausalHop,
    *,
    known_members: Mapping[PipelineStage, set[str]],
    operations_by_name: Mapping[str, Operation],
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

    # No machine-readable relation currently exists for coding-internal or
    # semantics-internal reasoning. Such hops remain valid once both members
    # are known rather than being rejected on a guess.
    return None
