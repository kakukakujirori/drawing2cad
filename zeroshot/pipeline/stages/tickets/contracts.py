import re
from collections.abc import Mapping, Sequence
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from zeroshot.pipeline.stages._base.contracts import Submission
from zeroshot.pipeline.stages.audit.contracts import AuditFinding
from zeroshot.pipeline.stages.types import REASONING_STAGES, ReasoningStage

_TICKET_ID = re.compile(r"^ticket_[a-z0-9][a-z0-9_]*$")
_CONCERN_ID = re.compile(r"^concern_[a-z0-9][a-z0-9_]*$")


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


_ANSWER_A_TICKET = (
    "Say what changed, why no change was "
    "needed, or what prevented resolution. Include upstream concerns "
    "and provisional interpretations needed to explain this ticket's "
    "outcome; put the rest in stage_report.concerns, one entry each. "
    "Do not restate the artifact's geometry or measurements: it "
    "remains authoritative. "
    "Cite the concrete stable names examined or changed: "
    "view_..., dim_..., or sem_... in interpretation, op_... in "
    "operations, and ret_... or result in coding."
)


class TicketResponse(BaseModel):
    """One stage's answer to one ticket, as the snapshot keeps it.

    The stage is stamped by the pipeline, which already knows which one
    answered; a submission carries the answers keyed by ticket instead.
    """

    model_config = ConfigDict(extra="forbid")

    ticket_id: str = Field(
        ...,
        description="The open ticket this response addresses.",
    )
    stage: ReasoningStage = Field(
        ...,
        description="The reasoning stage that produced this response.",
    )
    summary: str = Field(..., description=_ANSWER_A_TICKET)

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
    evidence_crops: list[str] = Field(
        default_factory=list,
        description=(
            "One picture per region of the subject's evidence, in the same "
            "order, cut out by the pipeline. Open these to see what the audit "
            "measured. No agent writes this field."
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

    concerns: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Additional unresolved issues and important provisional choices "
            "your assigned-ticket responses do not already explain, one entry "
            "each, and {} if none. Keep a ticket's own doubts in its "
            "`responses` answer; do not repeat them here. The key is a stable "
            "identifier beginning concern_ and carrying on in lower_snake_case: "
            "concern_web_thickness. The value states the affected subject, the "
            "doubt and how you handled it. The audit answers each entry by its "
            "key, so give one concern its own entry rather than several in one. "
            "Keep an entry's key while the concern stands. Include concerns "
            "about unassigned tickets, naming their IDs when known. Do not "
            "repeat artifact details or ticket summaries; a ticket summary "
            "still states its own outcome."
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
    unticketed_changes: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Changes your tickets did not ask for, each with its reason. Use a "
            "member name as the key: datum, view_..., dim_..., sem_..., op_... or "
            "ret_.... The value describes the change and its justification. "
            "Leave this {} in round 0 and when your tickets asked for every change."
        ),
    )

    @field_validator("concerns", "dimension_checks", "unticketed_changes")
    @classmethod
    def require_explanations(
        cls, explanations: dict[str, str] | None, info: ValidationInfo
    ) -> dict[str, str] | None:
        for name, explanation in (explanations or {}).items():
            if info.field_name == "concerns" and _CONCERN_ID.fullmatch(name) is None:
                raise ValueError(
                    f"concerns key {name!r} must be a concern_... identifier in "
                    "lower_snake_case, naming the concern rather than describing it"
                )
            if not explanation.strip():
                raise ValueError(
                    f"{info.field_name}.{name}: explanation must not be blank"
                )
        return explanations


def reported_concerns(reports: Mapping[ReasoningStage, StageReport]) -> list[str]:
    """Every reported concern, as the audit answers it: stage.concern_id."""
    return [
        f"{stage.value}.{concern}"
        for stage in REASONING_STAGES
        if stage in reports
        for concern in reports[stage].concerns
    ]


class TicketAnswers(Submission):
    """Your assigned-ticket answers and stage-wide observations.

    Every reasoning stage revises its artifact in its workspace file, which
    the pipeline verifies and reads back, so no artifact belongs in here.
    """

    responses: dict[str, str] = Field(
        ...,
        description=(
            "Your answer to each open ticket this stage is assigned to, keyed "
            "by ticket ID: exactly those tickets and no others. " + _ANSWER_A_TICKET
        ),
    )
    stage_report: StageReport = Field(
        default_factory=StageReport,
        description="Stage-wide observations and dimension checks.",
    )

    @field_validator("responses")
    @classmethod
    def require_an_answer(cls, answers: dict[str, str]) -> dict[str, str]:
        """A key that is not a ticket is left to the round's own check, which
        knows the open tickets and can name them."""
        for ticket_id, summary in answers.items():
            if not summary.strip():
                raise ValueError(f"{ticket_id}: the answer must not be blank")
        return answers
