"""The audit report file and its completion signal.

AuditReport
├── ticket_reviews: {ticket_id: TicketReview}
├── concern_reviews: {concern: ConcernReview}
└── findings: AuditFinding[]
    └── evidence: AuditRegion[]
"""

import re
from pathlib import PurePosixPath
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    model_validator,
)

from zeroshot.pipeline.stages._base.contracts import Submission

type FindingCause = Literal["interpretation", "coding"]

_FIND_NAME = re.compile(r"^find_[a-z0-9_]+$")
_MEMBER_NAME = re.compile(r"^(?:datum|(?:view|dim|sem)_[a-z0-9_]+)$")


class AuditRegion(BaseModel):
    """Where a defect is visible: one drawing in the workspace, and where to look."""

    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
        allow_inf_nan=False,
        json_schema_extra={"additionalProperties": False},
    )

    file: str = Field(
        ...,
        description=(
            "The workspace path of the drawing this was measured on. Use the "
            "input drawing, projection DXF or PNG, or perspective image of the "
            "built solid."
        ),
    )
    box: tuple[StrictInt | StrictFloat, ...] = Field(
        ...,
        min_length=4,
        max_length=4,
        description=(
            "For a DXF: u0, v0, u1, v1 in its own millimetre coordinates; "
            "decimals are allowed. For a raster (PNG/JPG etc.): x0, y0, x1, y1 "
            "as integer pixels from the top left, with y increasing downwards. "
            "Raster coordinates must be JSON integers, not decimals or strings."
        ),
    )

    @model_validator(mode="after")
    def require_an_ordered_box(self) -> Self:
        if PurePosixPath(self.file).suffix.lower() != ".dxf" and any(
            type(edge) is not int for edge in self.box
        ):
            raise ValueError("raster box must contain integer pixel coordinates")
        x0, y0, x1, y1 = self.box
        if x0 >= x1 or y0 >= y1:
            raise ValueError("box must satisfy x0 < x1 and y0 < y1")
        return self


class AuditSubmission(Submission):
    """Finish after reviewing the current report and its generated evidence."""

    accepted: bool = Field(
        ...,
        description=(
            "After reviewing AuditReport and its evidence crops, accept only when "
            "the reconstruction needs no correction. True exactly when the "
            "validated report's findings are empty."
        ),
    )


class AuditFinding(BaseModel):
    """One material defect, its evidence, and the revision it requires."""

    model_config = ConfigDict(
        extra="ignore", json_schema_extra={"additionalProperties": False}
    )

    name: str = Field(
        ...,
        description=(
            "A find_... lower_snake_case name unique within this report. "
            "Use find_ for every new audit output."
        ),
    )
    observation: str = Field(
        ...,
        description=(
            "The concrete mismatch or failure that was observed, without yet "
            "assigning it to a root cause."
        ),
    )
    evidence: list[AuditRegion] = Field(
        ...,
        description=(
            "Where the observation is visible, as files and boxes rather than "
            "prose. At least one must be a .dxf under a projection/ directory, "
            "so the mismatch is measured and not only seen."
        ),
    )
    cause: FindingCause = Field(
        ...,
        description=(
            "Where the defect originates. interpretation: interpretation.json "
            "misreads or omits something in the drawing, and the program "
            "faithfully builds that error. coding: interpretation.json is right "
            "for this defect, but the program builds something else."
        ),
    )
    targets: list[str] = Field(
        ...,
        description=(
            "Existing interpretation members this defect concerns: datum, "
            "view_..., dim_... or sem_.... For an interpretation cause, name at "
            "least one: the members that are wrong, or for something the "
            "interpretation omits, the view_ where the drawing shows it. For a "
            "coding cause, name the members the program builds wrongly, or [] "
            "when none applies."
        ),
    )
    revision_request: str = Field(
        ...,
        description=(
            "What is wrong at the cause: what the drawing shows and what the "
            "artifact has instead. Take values from printed dimensions where "
            "possible, and give directions as model axes with signs. State the "
            "defect, not its correction: the owning stage decides how to correct "
            "it."
        ),
    )
    related_ticket_ids: list[str] = Field(
        ...,
        description=(
            "If this finding describes the remaining problem of an open ticket "
            "you reviewed as unsolved, list that ticket's ID. Use [] for a new defect. "
            "Several tickets may share a finding, and "
            "one ticket may require several findings."
        ),
    )

    @model_validator(mode="after")
    def require_evidence_and_targets(self) -> Self:
        if _FIND_NAME.fullmatch(self.name) is None:
            raise ValueError("name must be a find_... lower_snake_case name")
        if not self.observation.strip():
            raise ValueError("observation must not be blank")
        if not self.revision_request.strip():
            raise ValueError("revision_request must not be blank")
        if not self.evidence:
            raise ValueError("evidence must not be empty")
        if len(set(self.evidence)) != len(self.evidence):
            raise ValueError("evidence regions must not contain duplicates")
        if len(set(self.related_ticket_ids)) != len(self.related_ticket_ids):
            raise ValueError("related_ticket_ids must not contain duplicates")
        if invalid := [t for t in self.targets if _MEMBER_NAME.fullmatch(t) is None]:
            raise ValueError(
                "targets must be datum, view_..., dim_... or sem_... names, got "
                + ", ".join(map(repr, invalid))
            )
        if len(set(self.targets)) != len(self.targets):
            raise ValueError("targets must not contain duplicates")
        if self.cause == "interpretation" and not self.targets:
            raise ValueError(
                "an interpretation cause needs at least one target; for an "
                "omission, name the view_ where the drawing shows it"
            )
        return self


# Report-wide consistency across otherwise independent findings.


class TicketReview(BaseModel):
    """Whether an open ticket's issue is settled in the current artifacts.

    Keyed by the ticket it checks, so the ticket cannot be left out.
    """

    model_config = ConfigDict(
        extra="ignore", json_schema_extra={"additionalProperties": False}
    )

    summary: str = Field(
        ...,
        description=(
            "The current check and why the ticket's issue is settled or remains. "
            "Do not repeat a finding's revision request here."
        ),
    )
    solved: bool = Field(
        ...,
        description=(
            "True only if the ticket's issue is resolved, not "
            "merely because the edit was attempted. If false, a current finding "
            "must include this ticket in related_ticket_ids."
        ),
    )

    @model_validator(mode="after")
    def require_a_check_summary(self) -> Self:
        if not self.summary.strip():
            raise ValueError("summary must not be blank")
        return self


class ConcernReview(BaseModel):
    """How one stage-report concern or drawing_diff item is disposed of.

    Keyed by what it answers, so nothing can be left out.
    """

    model_config = ConfigDict(
        extra="ignore", json_schema_extra={"additionalProperties": False}
    )

    finding_name: str | None = Field(
        ...,
        description=(
            "The finding that takes this concern over, once you judge the "
            "concern a real defect; null when it needs no correction. One "
            "concern has one root cause, and several concerns may name one "
            "finding."
        ),
    )
    disposition: str = Field(
        ...,
        description=(
            "How the concern is settled: what you checked, and why it needs no "
            "correction or how the named finding corroborates it."
        ),
    )

    @model_validator(mode="after")
    def require_a_disposition(self) -> Self:
        if not self.disposition.strip():
            raise ValueError("disposition must not be blank")
        return self


class AuditReport(Submission):
    """The audit.json artifact; the final decision belongs to AuditSubmission."""

    ticket_reviews: dict[str, TicketReview] = Field(
        ...,
        description=(
            "Your check of each open ticket, keyed by ticket ID: exactly those "
            "tickets and no others."
        ),
    )
    findings: list[AuditFinding] = Field(
        ...,
        description=(
            "Every material defect found; empty only when the reconstruction "
            "is accepted."
        ),
    )
    concern_reviews: dict[str, ConcernReview] = Field(
        ...,
        description=(
            "Your answer to every concern the current stage_reports raise, and "
            "to each drawing_diff item when the round instructions ask for them. "
            "Key a concern by <reporting_stage>.<concern_id> as it appears "
            "there: coding.concern_bore_diameter. The prefix names the stage "
            "that reported the concern, not the stage that must change. Key a "
            "drawing_diff item as listed: drawing_diff.view_top.1. For each "
            "entry, name the finding that takes it over, or the reason it needs "
            "none."
        ),
    )

    @model_validator(mode="after")
    def require_consistent_findings(self) -> Self:
        """Require unambiguous reviews and finding names."""
        unsolved = {
            ticket_id
            for ticket_id, review in self.ticket_reviews.items()
            if not review.solved
        }
        related = {
            ticket_id
            for finding in self.findings
            for ticket_id in finding.related_ticket_ids
        }
        if unsolved != related:
            raise ValueError(
                "unsolved ticket IDs must equal finding.related_ticket_ids: "
                f"without a finding={sorted(unsolved - related)}, "
                f"without an unsolved review={sorted(related - unsolved)}"
            )
        names = [finding.name for finding in self.findings]
        if len(set(names)) != len(names):
            raise ValueError("finding names must be unique within a report")
        for concern, review in self.concern_reviews.items():
            if review.finding_name is not None and review.finding_name not in names:
                raise ValueError(
                    f"concern_reviews for {concern} names a finding this "
                    f"report does not hold: {review.finding_name}"
                )
        return self
