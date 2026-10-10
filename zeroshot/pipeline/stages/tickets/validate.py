"""Check coding's ticket answers against its round."""

from zeroshot.pipeline.stages._base.validate import SubmissionValidationError
from zeroshot.pipeline.stages.contracts import ReconstructionSnapshot
from zeroshot.pipeline.stages.tickets.contracts import TicketAnswers


def validate_ticket_answers(
    answers: TicketAnswers, snapshot: ReconstructionSnapshot
) -> None:
    """Require one answer for every open ticket, and no other.

    Keying the answers by ticket makes a duplicate unrepresentable, so only
    the two membership errors remain.
    """
    expected = {ticket.ticket_id for ticket in snapshot.open_tickets}
    submitted = set(answers.responses)
    errors: list[str] = []
    if missing := sorted(expected - submitted):
        errors.append("missing ticket responses: " + ", ".join(missing))
    if unknown := sorted(submitted - expected):
        errors.append(
            "unknown ticket responses: "
            + ", ".join(unknown)
            + ". Open tickets: "
            + (", ".join(sorted(expected)) or "none")
        )
    if errors:
        raise SubmissionValidationError("\n".join(errors))
