"""Check a stage's ticket answers against its round and the round it revises."""

from collections import Counter
from collections.abc import Sequence
from functools import partial

from zeroshot.pipeline.stages._base.validate import (
    SubmissionValidationError,
    raise_together,
)
from zeroshot.pipeline.stages.audit.contracts import AuditFinding
from zeroshot.pipeline.stages.coding.validate import validate_dimension_checks
from zeroshot.pipeline.stages.contracts import (
    ReconstructionHistory,
    ReconstructionSnapshot,
)
from zeroshot.pipeline.stages.interpretation.contracts import DrawingInterpretation
from zeroshot.pipeline.stages.operations.contracts import OperationPlan
from zeroshot.pipeline.stages.tickets.contracts import (
    StageReport,
    Ticket,
    TicketAnswers,
    TicketResponse,
    tickets_assigned_to,
)
from zeroshot.pipeline.stages.types import (
    REASONING_STAGES,
    Member,
    PipelineStage,
    ReasoningStage,
    next_stage,
)
from zeroshot.pipeline.verification.check_program import program_members

type StageArtifact = DrawingInterpretation | OperationPlan | str


def validate_ticket_answers(
    answers: TicketAnswers, snapshot: ReconstructionSnapshot
) -> None:
    """Reject ticket responses or a stage report that contradict the round."""
    stage = next_stage(snapshot.last_completed_stage)
    if stage not in REASONING_STAGES:
        return  # This should not happen
    checks = answers.stage_report.dimension_checks

    def validate_report() -> None:
        if stage is PipelineStage.CODING:
            validate_dimension_checks(checks, snapshot.interpretation)
        elif checks is not None:
            raise SubmissionValidationError(f"{stage} dimension_checks must be null")

    raise_together(
        partial(
            _validate_responses,
            answers.responses,
            snapshot.open_tickets,
            expected_stage=stage,
        ),
        validate_report,
    )


def validate_revision_scope(
    report: StageReport,
    history: ReconstructionHistory,
    artifact: StageArtifact | None,
) -> None:
    """Reject member changes that no ticket covers and the report leaves unexplained."""
    # Round 0 has no previous artifact to compare with.
    if len(history.snapshots) < 2:
        return
    previous, current = history.snapshots[-2:]
    stage = next_stage(current.last_completed_stage)
    if stage not in REASONING_STAGES:
        return
    requests = [
        ticket.subject.revision_request
        for ticket in tickets_assigned_to(current.open_tickets, stage)
        if isinstance(ticket.subject, AuditFinding)
    ]
    # A whole-stage modify lets the stage change anything.
    if any(
        request.action == "modify"
        and request.targets[0].stage is stage
        and request.targets[0].name is None
        for request in requests
    ):
        return

    # 1. What this stage added, removed or changed since the previous round.
    baseline = _artifact(previous, stage)
    if baseline is None or not isinstance(artifact, type(baseline)):
        return
    before, after = _members(baseline), _members(artifact)
    if before is None or after is None:
        return
    changes = _changes(before, after)

    # 2. What justifies a change: the names the tickets give, what earlier stages
    #    changed this round, and the earlier members that cite either.
    named = {
        name
        for request in requests
        for name in (
            *(target.name for target in request.targets),
            *request.proposed_names,
        )
        if name is not None
    }
    upstream = [
        (_members(_artifact(previous, earlier)), _members(_artifact(current, earlier)))
        for earlier in REASONING_STAGES[: REASONING_STAGES.index(stage)]
    ]
    reasons = named | {
        name
        for earlier_before, earlier_after in upstream
        if earlier_before is not None and earlier_after is not None
        for name in _changes(earlier_before, earlier_after)
    }
    reasons |= {
        name
        for pair in upstream
        for members in pair
        if members is not None
        for name, member in members.items()
        if member.cites & reasons
    }

    # 3. A change is covered when its member is named, or cites a reason or
    #    another change of this stage. Otherwise the report must explain it.
    uncovered = sorted(
        f"{name} ({change})"
        for name, change in changes.items()
        if name not in named
        and not _cites(name, before, after) & (reasons | changes.keys())
        and name not in report.unticketed_changes
    )

    # 4. Report those, and any explanation given for a member that did not change.
    errors = []
    if uncovered:
        errors.append(
            f"These {stage} changes are outside the assigned tickets: "
            f"{', '.join(uncovered)}. Undo them, or give each a reason in "
            "stage_report.unticketed_changes."
        )
    if unchanged := sorted(report.unticketed_changes.keys() - changes.keys()):
        errors.append(
            "stage_report.unticketed_changes names members this stage did not "
            f"change: {', '.join(unchanged)}"
        )
    if errors:
        raise SubmissionValidationError("\n".join(errors))


def _artifact(
    snapshot: ReconstructionSnapshot, stage: ReasoningStage
) -> StageArtifact | None:
    return {
        PipelineStage.INTERPRETATION: snapshot.interpretation,
        PipelineStage.OPERATIONS: snapshot.operations,
        PipelineStage.CODING: snapshot.program_source,
    }[stage]


def _members(artifact: StageArtifact | None) -> dict[str, Member] | None:
    """None when there is nothing to compare, such as a program that does not parse."""
    if isinstance(artifact, DrawingInterpretation | OperationPlan):
        return artifact.members()
    if isinstance(artifact, str):
        try:
            return program_members(artifact)
        except SyntaxError:
            return None
    return None


def _changes(before: dict[str, Member], after: dict[str, Member]) -> dict[str, str]:
    return {
        **{name: "removed" for name in before.keys() - after.keys()},
        **{name: "added" for name in after.keys() - before.keys()},
        **{
            name: "changed"
            for name in before.keys() & after.keys()
            if before[name].content != after[name].content
        },
    }


def _cites(
    name: str, before: dict[str, Member], after: dict[str, Member]
) -> frozenset[str]:
    """What the member cites in either round, so a dropped citation still counts."""
    return frozenset[str]().union(
        *(members[name].cites for members in (before, after) if name in members)
    )


def _validate_responses(
    responses: Sequence[TicketResponse],
    tickets: Sequence[Ticket],
    *,
    expected_stage: ReasoningStage,
) -> None:
    """Require one response for every ticket assigned to this stage, and no other."""
    known_ids = {ticket.ticket_id for ticket in tickets}
    expected_ids = {
        ticket.ticket_id for ticket in tickets_assigned_to(tickets, expected_stage)
    }
    response_counts = Counter(response.ticket_id for response in responses)
    submitted_ids = set(response_counts)

    duplicated = sorted(
        ticket_id for ticket_id, count in response_counts.items() if count > 1
    )
    missing = sorted(expected_ids - submitted_ids)
    unassigned = sorted(submitted_ids & (known_ids - expected_ids))
    unknown = sorted(submitted_ids - known_ids)
    wrong_stage = sorted(
        response.ticket_id for response in responses if response.stage != expected_stage
    )

    errors: list[str] = []
    if duplicated:
        errors.append("duplicate ticket responses: " + ", ".join(duplicated))
    if missing:
        errors.append("missing ticket responses: " + ", ".join(missing))
    if unassigned:
        errors.append(
            f"{expected_stage} is not assigned to these tickets and must not "
            "respond to them: " + ", ".join(unassigned)
        )
    if unknown:
        errors.append("unknown ticket responses: " + ", ".join(unknown))
    if wrong_stage:
        errors.append(
            f"ticket responses must belong to {expected_stage}: "
            + ", ".join(wrong_stage)
        )

    if errors:
        raise SubmissionValidationError("\n".join(errors))
