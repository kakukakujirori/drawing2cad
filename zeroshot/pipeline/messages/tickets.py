import re
from collections.abc import Sequence
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from zeroshot.pipeline.stages.audit.contracts import AuditFinding
from zeroshot.pipeline.stages.types import REASONING_STAGES, ReasoningStage

_TICKET_ID = re.compile(r"^ticket_[a-z0-9][a-z0-9_]*$")


class BootstrapWork(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instruction: str = Field(
        ...,
        description=(
            "The machine-owned initial reconstruction task used before any "
            "audited finding exists."
        ),
    )

    @field_validator("instruction")
    @classmethod
    def require_an_instruction(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("instruction must not be blank")
        return value


class TicketResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticket_id: str = Field(
        ...,
        description="The open ticket this response addresses.",
    )
    stage: ReasoningStage = Field(
        ...,
        description="The reasoning stage that produced this response.",
    )
    summary: str = Field(
        ...,
        description=(
            "Answer this assigned ticket: what changed, why no change was "
            "needed, or what prevented resolution. Include upstream concerns "
            "and provisional interpretations needed to explain this ticket's "
            "outcome; put additional concerns in remark. Do not restate the "
            "artifact's geometry or measurements: it remains authoritative. "
            "Cite the concrete stable names examined or changed: "
            "view_..., dim_..., or sem_... in interpretation, op_... in "
            "operations, and ret_... or result in coding."
        ),
    )

    @model_validator(mode="after")
    def require_valid_content(self) -> Self:
        if _TICKET_ID.fullmatch(self.ticket_id) is None:
            raise ValueError("ticket_id must be a ticket_... identifier")
        if not self.summary.strip():
            raise ValueError("summary must not be blank")
        return self


class Ticket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticket_id: str = Field(
        ...,
        description=(
            "The stable ticket_... identifier assigned by the pipeline for this round."
        ),
    )
    subject: BootstrapWork | AuditFinding = Field(
        ...,
        description=(
            "The initial reconstruction instruction or the audited defect "
            "that the assigned reasoning stages must address."
        ),
    )
    assigned_stages: list[ReasoningStage] = Field(
        ...,
        description=(
            "The stages that must answer this ticket, assigned by the pipeline "
            "from the revision roots the audit requested: the earliest root and "
            "every stage downstream of it, because a corrected artifact has to "
            "be carried through to the program. A stage that is not listed here "
            "must leave this ticket alone. No agent writes this field."
        ),
    )
    responses: list[TicketResponse] = Field(
        ...,
        description=(
            "One response from each assigned stage that has completed, kept in "
            "stage order. A newly opened ticket has an empty list."
        ),
    )

    @model_validator(mode="after")
    def require_consistent_responses(self) -> Self:
        if _TICKET_ID.fullmatch(self.ticket_id) is None:
            raise ValueError("ticket_id must be a ticket_... identifier")

        if not self.assigned_stages:
            raise ValueError("a ticket must be assigned to at least one stage")
        if (
            tuple(self.assigned_stages)
            != REASONING_STAGES[-len(self.assigned_stages) :]
        ):
            raise ValueError(
                "assigned_stages must run from one revision root through coding, "
                f"got {self.assigned_stages}"
            )

        stages: list[ReasoningStage] = []
        for response in self.responses:
            if response.ticket_id != self.ticket_id:
                raise ValueError("every response must refer to its containing ticket")
            if response.stage not in self.assigned_stages:
                raise ValueError(f"{response.stage} is not assigned to this ticket")
            stages.append(response.stage)

        if len(stages) != len(set(stages)):
            raise ValueError("a ticket may have only one response per stage")

        return self


def tickets_assigned_to(
    tickets: Sequence[Ticket],
    stage: ReasoningStage,
) -> list[Ticket]:
    return [ticket for ticket in tickets if stage in ticket.assigned_stages]


class StageReport(BaseModel):
    """Stage-wide observations stored once, separately from ticket responses."""

    model_config = ConfigDict(extra="forbid")

    remark: str = Field(
        default="",
        description=(
            "Additional concerns not covered by the assigned-ticket responses; "
            "empty if none. State the affected subject, reason and how you "
            "handled it. Include concerns about unassigned tickets here, naming "
            "their IDs when known. Do not repeat artifact details, ticket "
            "summaries or recorded questions. A shared explanation may appear "
            "once here, but each ticket summary must still state its outcome "
            "and the concern's effect on it."
        ),
    )
    dimension_checks: dict[str, str] | None = Field(
        default=None,
        description=(
            "Coding only: one entry for every dim_ name in the current "
            "interpretation, including unreadable values and equal values under "
            "different names. For each dimension identify where the final "
            "geometry realizes it and the supporting check, or explain why it "
            "is not established or not checked. Assigning a value to a variable "
            "alone does not establish the geometry. Use {} when there are no "
            "dimensions, and null in other stages. This is the coder's account, "
            "not independent proof that the dimensions are satisfied."
        ),
    )

    @field_validator("dimension_checks")
    @classmethod
    def require_dimension_explanations(
        cls, checks: dict[str, str] | None
    ) -> dict[str, str] | None:
        for name, explanation in (checks or {}).items():
            if not explanation.strip():
                raise ValueError(
                    f"{name}: dimension check explanation must not be blank"
                )
        return checks


class TicketAnswers(StageReport):
    """Your assigned-ticket answers and any additional stage-wide observations.

    Every reasoning stage revises its artifact in its workspace file, which
    the pipeline verifies and reads back, so no artifact belongs in here.
    """

    responses: list[TicketResponse] = Field(
        ...,
        description=(
            "Exactly one response for every open ticket this stage is assigned "
            "to, and none for the others. The pipeline validates the ticket IDs "
            "and the assignment against the current snapshot."
        ),
    )
