"""Check a stage's ticket answers against its round."""

from collections.abc import Mapping, Sequence

from zeroshot.pipeline.stages._base.validate import SubmissionValidationError
from zeroshot.pipeline.stages.contracts import ReconstructionSnapshot
from zeroshot.pipeline.stages.tickets.contracts import (
    Ticket,
    TicketAnswers,
    tickets_assigned_to,
)
from zeroshot.pipeline.stages.types import (
    REASONING_STAGES,
    ReasoningStage,
    next_stage,
)


def validate_ticket_answers(
    answers: TicketAnswers, snapshot: ReconstructionSnapshot
) -> None:
    """Reject ticket responses that contradict the round."""
    stage = next_stage(snapshot.last_completed_stage)
    if stage not in REASONING_STAGES:
        return  # This should not happen
    _validate_responses(answers.responses, snapshot.open_tickets, expected_stage=stage)


def _validate_responses(
    responses: Mapping[str, str],
    tickets: Sequence[Ticket],
    *,
    expected_stage: ReasoningStage,
) -> None:
    """Require one answer for every ticket assigned to this stage, and no other.

    Keying the answers by ticket makes a duplicate or a stage disagreement
    unrepresentable, so only the three membership errors remain.
    """
    known_ids = {ticket.ticket_id for ticket in tickets}
    expected_ids = {
        ticket.ticket_id for ticket in tickets_assigned_to(tickets, expected_stage)
    }
    submitted_ids = set(responses)

    missing = sorted(expected_ids - submitted_ids)
    unassigned = sorted(submitted_ids & (known_ids - expected_ids))
    unknown = sorted(submitted_ids - known_ids)

    errors: list[str] = []
    if missing:
        errors.append("missing ticket responses: " + ", ".join(missing))
    if unassigned:
        errors.append(
            f"{expected_stage} is not assigned to these tickets and must not "
            "respond to them: " + ", ".join(unassigned)
        )
    if unknown:
        errors.append(
            "unknown ticket responses: "
            + ", ".join(unknown)
            + ". Open tickets for this stage: "
            + (", ".join(sorted(expected_ids)) or "none")
        )

    if errors:
        raise SubmissionValidationError("\n".join(errors))
