from collections import Counter
from collections.abc import Sequence

from zeroshot.pipeline.messages.tickets import (
    Ticket,
    TicketResponse,
    tickets_assigned_to,
)
from zeroshot.pipeline.stages.types import ReasoningStage


class SubmissionValidationError(ValueError):
    """A stage's submission contradicts the current reconstruction snapshot."""


def validate_ticket_responses(
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
