"""Validation, lifecycle transitions, and persistence for reconstruction runs."""

import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

from zeroshot.pipeline.messages.contracts import Operation, OperationPlan
from zeroshot.pipeline.messages.contracts.audit import (
    AuditFinding,
    AuditReport,
    Backtrace,
    CausalHop,
    ReasoningStage,
    StageOutputRef,
)
from zeroshot.pipeline.messages.contracts.reconstruction import (
    BootstrapWork,
    ReconstructionRun,
    ReconstructionSnapshot,
    Ticket,
)
from zeroshot.pipeline.verification import check_program

# ---------------------------------------------------------------------------
# Audit cross validation
# ---------------------------------------------------------------------------


class AuditCrossValidationError(ValueError):
    """An audit report refers to relationships absent from its snapshot."""


def check_audit_report(
    report: AuditReport,
    snapshot: ReconstructionSnapshot,
) -> None:
    """Raise when a structurally valid report contradicts the audited snapshot."""
    if snapshot.last_completed_stage != "coding":
        raise AuditCrossValidationError("audit requires a completed coding snapshot")

    backtraces = tuple(
        backtrace for finding in report.findings for backtrace in finding.backtraces
    )
    references = tuple(_iter_references(backtraces))

    semantics_by_name = (
        {feature.name: feature for feature in snapshot.semantics.proposal}
        if snapshot.semantics is not None
        else {}
    )
    operations_by_name = (
        {operation.name: operation for operation in snapshot.operations.proposal}
        if snapshot.operations is not None
        else {}
    )
    coding_names, coding_error = _inspect_coding_outputs(snapshot, references)

    known_members = {
        "semantics": set(semantics_by_name),
        "operations": set(operations_by_name),
        "coding": coding_names,
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
        raise AuditCrossValidationError("\n".join(unique_errors))


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
    snapshot: ReconstructionSnapshot,
    references: Iterable[StageOutputRef],
) -> tuple[set[str], str | None]:
    """Return named code outputs and any syntax failure that hides them."""
    needs_named_outputs = any(
        reference.stage == "coding" and reference.name is not None
        for reference in references
    )
    if (
        not needs_named_outputs
        or snapshot.program_source is None
        or snapshot.operations is None
    ):
        return set(), None

    try:
        return _program_output_names(
            snapshot.program_source,
            snapshot.operations,
        ), None
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
    known_members: Mapping[str, set[str]],
) -> list[str]:
    """Report each absent named output once, in first-reference order."""
    errors: list[str] = []
    seen: set[tuple[str, str | None]] = set()
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
    known_members: Mapping[str, set[str]],
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

    if effect.stage == "coding" and cause.stage == "operations":
        expected_return = f"ret_{cause.name.removeprefix('op_')}"
        if effect.name != expected_return:
            return (
                f"coding-to-operations hop {effect.name!r} -> "
                f"{cause.name!r} must use {expected_return!r}"
            )

    elif effect.stage == "operations" and cause.stage == "operations":
        operation = operations_by_name[effect.name]
        if cause.name not in operation.depends_on:
            return (
                f"operations hop {effect.name!r} -> {cause.name!r} is "
                f"not supported by {effect.name}.depends_on"
            )

    elif effect.stage == "operations" and cause.stage == "semantics":
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


# ---------------------------------------------------------------------------
# Pure lifecycle transitions
# ---------------------------------------------------------------------------


def start_reconstruction(
    run_id: str,
    instruction: str,
) -> ReconstructionRun:
    snapshot = ReconstructionSnapshot(
        open_tickets=[
            Ticket(
                ticket_id="ticket_initial",
                subject=BootstrapWork(instruction=instruction),
                responses=[],
            )
        ],
        round=0,
        last_completed_stage=None,
        semantics=None,
        operations=None,
        program_source=None,
        verification=None,
    )
    return ReconstructionRun(
        schema_version=1,
        run_id=run_id,
        snapshots=[snapshot],
    )


def replace_current_snapshot(
    run: ReconstructionRun,
    snapshot: ReconstructionSnapshot,
) -> ReconstructionRun:
    """Return a new run after one stage atomically advances the current round."""
    current = run.snapshots[-1]

    if current.last_completed_stage == "coding":
        raise ValueError("a completed coding snapshot is immutable")
    if snapshot.round != current.round:
        raise ValueError("replacement must belong to the current round")

    next_stage: dict[ReasoningStage | None, ReasoningStage] = {
        None: "semantics",
        "semantics": "operations",
        "operations": "coding",
    }
    expected_stage = next_stage[current.last_completed_stage]
    if snapshot.last_completed_stage != expected_stage:
        raise ValueError(
            f"current round must advance from {current.last_completed_stage!r} "
            f"to {expected_stage!r}"
        )

    _require_ticket_progress(current, snapshot)
    _require_only_stage_artifact_changed(current, snapshot, expected_stage)

    return ReconstructionRun(
        schema_version=run.schema_version,
        run_id=run.run_id,
        snapshots=[*run.snapshots[:-1], snapshot],
    )


def _require_ticket_progress(
    current: ReconstructionSnapshot,
    replacement: ReconstructionSnapshot,
) -> None:
    """Keep ticket identity and prior responses fixed within one round."""
    current_ids = [ticket.ticket_id for ticket in current.open_tickets]
    replacement_ids = [ticket.ticket_id for ticket in replacement.open_tickets]
    if replacement_ids != current_ids:
        raise ValueError("open tickets must not change within a round")

    for previous, updated in zip(
        current.open_tickets,
        replacement.open_tickets,
        strict=True,
    ):
        if updated.subject != previous.subject:
            raise ValueError(
                f"{previous.ticket_id} subject must not change within a round"
            )
        if (
            len(updated.responses) != len(previous.responses) + 1
            or updated.responses[:-1] != previous.responses
        ):
            raise ValueError(
                f"{previous.ticket_id} must append one response without "
                "rewriting prior responses"
            )


def _require_only_stage_artifact_changed(
    current: ReconstructionSnapshot,
    replacement: ReconstructionSnapshot,
    stage: ReasoningStage,
) -> None:
    """A stage may replace its own artifact but not an upstream/downstream one."""
    owned_artifact = {
        "semantics": "semantics",
        "operations": "operations",
        "coding": "program_source",
    }[stage]
    for artifact in ("semantics", "operations", "program_source"):
        if artifact == owned_artifact:
            continue
        if getattr(replacement, artifact) != getattr(current, artifact):
            raise ValueError(f"{stage} must preserve the current {artifact} artifact")


def open_next_round(
    run: ReconstructionRun,
    report: AuditReport,
) -> ReconstructionRun:
    """Create the next round from a rejected, cross-validated audit report."""
    current = run.snapshots[-1]
    check_audit_report(report, current)

    if report.accepted:
        raise ValueError("an accepted audit does not open another round")

    next_round = current.round + 1
    tickets = [_ticket_from_finding(next_round, finding) for finding in report.findings]
    snapshot = ReconstructionSnapshot(
        open_tickets=tickets,
        round=next_round,
        last_completed_stage=None,
        semantics=current.semantics,
        operations=current.operations,
        program_source=current.program_source,
        verification=None,
    )
    return ReconstructionRun(
        schema_version=run.schema_version,
        run_id=run.run_id,
        snapshots=[*run.snapshots, snapshot],
    )


def _ticket_from_finding(
    round_number: int,
    finding: AuditFinding,
) -> Ticket:
    suffix = finding.name.removeprefix("finding_")
    return Ticket(
        ticket_id=f"ticket_{round_number:03d}_{suffix}",
        subject=finding,
        responses=[],
    )


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------


def load_reconstruction(path: Path) -> ReconstructionRun:
    """Load and validate the complete reconstruction history at ``path``."""
    return ReconstructionRun.model_validate_json(path.read_text(encoding="utf-8"))


def save_reconstruction(path: Path, run: ReconstructionRun) -> None:
    """Atomically replace ``path`` with the complete reconstruction history."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = run.model_dump_json(indent=2) + "\n"

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)

    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
