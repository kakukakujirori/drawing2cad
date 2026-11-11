import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
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
    "needed, or what prevented resolution. Include the doubts "
    "and provisional interpretations needed to explain this ticket's "
    "outcome; put the rest in stage_report.concerns, one entry each. "
    "Do not restate the artifact's geometry or measurements: it "
    "remains authoritative. "
    "Cite the concrete stable names examined or changed: "
    "view_..., dim_..., or sem_... in interpretation.json."
)


class StageTicketResponse(BaseModel):
    """A submitted TicketResponse as the snapshot keeps it, stamped with its stage.

    The pipeline stamps the stage rather than asking for it: it knows which
    stage ran.
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
            "that coding must address."
        ),
    )
    responses: list[StageTicketResponse] = Field(
        ...,
        description=(
            "Coding's response once it has completed; a newly opened ticket "
            "has an empty list."
        ),
    )
    evidence_renders: list[str] = Field(
        default_factory=list,
        description=(
            "Paths to images, one per region of the subject's evidence, "
            "in the same order. Regions are marked with red boxes or cropped "
            "in crop mode. Open these images to inspect the audit evidence. "
            "No agent writes this field."
        ),
    )

    @model_validator(mode="after")
    def require_consistent_responses(self) -> Self:
        if _TICKET_ID.fullmatch(self.ticket_id) is None:
            raise ValueError("ticket_id must be a ticket_... identifier")

        stages: list[ReasoningStage] = []
        for response in self.responses:
            if response.ticket_id != self.ticket_id:
                raise ValueError("every response must refer to its containing ticket")
            stages.append(response.stage)

        if len(stages) != len(set(stages)):
            raise ValueError("a ticket may have only one response per stage")

        return self


def _repeated(names: Iterable[str]) -> list[str]:
    counts = Counter(names)
    return sorted(name for name, count in counts.items() if count > 1)


class Concern(BaseModel):
    """One unresolved issue or provisional choice a stage reports."""

    model_config = ConfigDict(
        extra="ignore", json_schema_extra={"additionalProperties": False}
    )

    id: str = Field(
        ...,
        description=(
            "A stable identifier beginning concern_ and carrying on in "
            "lower_snake_case: concern_web_thickness. The audit answers the "
            "concern by it. Keep it while the concern stands."
        ),
    )
    explanation: str = Field(
        ...,
        description="The affected subject, the doubt and how you handled it.",
    )

    @model_validator(mode="after")
    def require_valid_content(self) -> Self:
        if _CONCERN_ID.fullmatch(self.id) is None:
            raise ValueError(
                f"concern id {self.id!r} must be a concern_... identifier in "
                "lower_snake_case, naming the concern rather than describing it"
            )
        if not self.explanation.strip():
            raise ValueError(f"{self.id}: explanation must not be blank")
        return self


class StageReport(BaseModel):
    """Stage-wide observations stored once, separately from ticket responses."""

    model_config = ConfigDict(
        extra="ignore", json_schema_extra={"additionalProperties": False}
    )

    concerns: list[Concern] = Field(
        ...,
        description=(
            "Additional unresolved issues and important provisional choices "
            "your ticket responses do not already explain, one entry "
            "each, and [] if none. Keep a ticket's own doubts in its "
            "`responses` answer; do not repeat them here. The audit answers "
            "each entry by its id, so give one concern its own entry rather "
            "than several in one. Do not repeat artifact details or ticket "
            "summaries; a ticket summary still states its own outcome."
        ),
    )

    @field_validator("concerns")
    @classmethod
    def require_distinct_ids(cls, concerns: list[Concern]) -> list[Concern]:
        if repeated := _repeated(concern.id for concern in concerns):
            raise ValueError("concerns repeat an id: " + ", ".join(repeated))
        return concerns


def reported_concerns(reports: Mapping[ReasoningStage, StageReport]) -> list[str]:
    """Every reported concern, as the audit answers it: stage.concern_id."""
    return [
        f"{stage.value}.{concern.id}"
        for stage in REASONING_STAGES
        if stage in reports
        for concern in reports[stage].concerns
    ]


class TicketResponse(BaseModel):
    """One open ticket's answer, as a stage submits it."""

    model_config = ConfigDict(
        extra="ignore", json_schema_extra={"additionalProperties": False}
    )

    ticket_id: str = Field(..., description="The open ticket this answers.")
    summary: str = Field(..., description=_ANSWER_A_TICKET)

    @model_validator(mode="after")
    def require_a_summary(self) -> Self:
        """An ID that is not a ticket is left to the round's own check, which
        knows the open tickets and can name them."""
        if not self.summary.strip():
            raise ValueError(f"{self.ticket_id}: summary must not be blank")
        return self


class TicketAnswers(Submission):
    """Your ticket answers and stage-wide observations.

    Coding revises its artifacts in workspace files, which the pipeline
    verifies and reads back, so no artifact belongs in here.
    """

    responses: list[TicketResponse] = Field(
        ...,
        description=(
            "Your answer to each open ticket, one entry per ticket: exactly "
            "those tickets and no others."
        ),
    )
    stage_report: StageReport = Field(
        ...,
        description=(
            "Stage-wide observations. Include concerns even when empty: concerns=[]."
        ),
    )

    @field_validator("responses")
    @classmethod
    def require_one_answer_per_ticket(
        cls, answers: list[TicketResponse]
    ) -> list[TicketResponse]:
        if repeated := _repeated(answer.ticket_id for answer in answers):
            raise ValueError("responses repeat a ticket: " + ", ".join(repeated))
        return answers
