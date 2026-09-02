"""Durable reconstruction snapshots shared across revision rounds."""

import re
from collections.abc import Sequence
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from zeroshot.pipeline.messages.contracts.audit import AuditFinding
from zeroshot.pipeline.messages.contracts.operations import Operation, OperationPlan
from zeroshot.pipeline.messages.contracts.semantics import (
    SemanticFeature,
    SemanticHypothesis,
)
from zeroshot.pipeline.messages.contracts.stages import (
    REASONING_STAGES,
    PipelineStage,
    ReasoningStage,
)
from zeroshot.pipeline.verification import ExecutionStatus, VerifyOutputResult

_TICKET_ID = re.compile(r"^ticket_[a-z0-9][a-z0-9_]*$")
_RUN_ID = re.compile(r"^run_[a-z0-9][a-z0-9_]*$")
_ARTIFACT_BY_STAGE = {
    PipelineStage.SEMANTICS: "semantics",
    PipelineStage.OPERATIONS: "operations",
    PipelineStage.CODING: "program_source",
}


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
            "What this stage changed, or why no change was needed, for this "
            "ticket. Cite the concrete stable names examined or changed: "
            "sem_... in semantics, op_... in operations, and ret_... or "
            "result in coding."
        ),
    )

    @model_validator(mode="after")
    def require_valid_content(self) -> Self:
        if _TICKET_ID.fullmatch(self.ticket_id) is None:
            raise ValueError("ticket_id must be a ticket_... identifier")
        if not self.summary.strip():
            raise ValueError("summary must not be blank")
        return self


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


class StageSubmission[M](BaseModel):
    """One stage's revision of its artifact, and its answers to its tickets.

    A model reads the concrete subclass, never this: pydantic takes a schema
    description from the class it is asked for, so the wording that names
    sem_, op_ or the workspace program belongs to the subclass that means it.
    """

    model_config = ConfigDict(extra="forbid")

    edits: list[M] = Field(
        ...,
        description=(
            "The members this stage changed, each complete and under its own "
            "stable name. A name the artifact already holds replaces that "
            "member, a new name adds one, and a member left out keeps what it "
            "had."
        ),
    )
    deleted: list[str] = Field(
        ...,
        description=(
            "The members this stage dropped, by the names the artifact holds "
            "them under. A name given here must not also appear in `edits`."
        ),
    )
    rationale: str | None = Field(
        ...,
        description=(
            "The artifact's rationale, rewritten when this revision changed "
            "the reasoning behind it, or null to keep the one it already has. "
            "The first round has none to keep, so it must be stated there."
        ),
    )
    responses: list[TicketResponse] = Field(
        ...,
        description=(
            "Exactly one response for every open ticket this stage is assigned "
            "to, and none for the others. The pipeline validates the ticket IDs "
            "and the assignment against the current snapshot."
        ),
    )

    @classmethod
    def unchanged(cls) -> Self:
        """A revision that leaves the preceding round's artifact as it is."""
        return cls(edits=[], deleted=[], rationale=None, responses=[])

    @model_validator(mode="after")
    def require_distinct_edited_and_deleted_names(self) -> Self:
        # `CodingSubmission` types its edits `list[None]`, so a model that
        # sends `[null]` passes the field check and arrives here; reading
        # `.name` off it would raise AttributeError instead of the rejection
        # the retry loop knows how to feed back.
        named = [
            edit.name  # type: ignore[attr-defined]
            for edit in self.edits
            if edit is not None
        ]
        duplicated = sorted({name for name in named if named.count(name) > 1})
        if duplicated:
            raise ValueError(f"edits name {', '.join(duplicated)} more than once")
        if len(set(self.deleted)) != len(self.deleted):
            raise ValueError("deleted must not name the same address twice")
        both = sorted(set(named) & set(self.deleted))
        if both:
            raise ValueError(f"{', '.join(both)} is both edited and deleted")
        return self


class SemanticSubmission(StageSubmission[SemanticFeature]):
    """Your revision of the semantic hypothesis, and your ticket responses.

    This ends the semantics stage: give it once, after the analysis
    behind it is complete.
    """

    edits: list[SemanticFeature] = Field(
        ...,
        description=(
            "Every feature you changed, each complete and under its stable "
            "sem_ name: a name the hypothesis already holds replaces that "
            "feature, and a new name adds one. A feature is itself a list of "
            "named geo_ and ev_ members, and the same rule holds within it -- "
            "give the members you changed and leave the rest out, and the ones "
            "you leave out keep what they had. `description` and "
            "`open_question` carry no name of their own, so state them "
            "whenever you give a feature."
        ),
    )
    deleted: list[str] = Field(
        ...,
        description=(
            "Every feature or member you dropped: a whole feature as "
            "sem_main_bore, and one member of a feature as "
            "sem_main_bore.geo_cylinder or sem_main_bore.ev_front_circle. A "
            "member may be dropped from a feature you are not otherwise "
            "changing. A name given here must not also appear in `edits`."
        ),
    )


class OperationSubmission(StageSubmission[Operation]):
    """Your revision of the operation plan, and your ticket responses.

    This ends the operations stage: give it once, after the analysis
    behind it is complete.
    """

    edits: list[Operation] = Field(
        ...,
        description=(
            "Every operation you changed, each complete and under its stable "
            "op_ name: a name the plan already holds replaces that operation, "
            "and a new name adds one. An operation you leave out keeps what it "
            "had."
        ),
    )
    deleted: list[str] = Field(
        ...,
        description=(
            "Every operation you dropped, by its own op_ name. An operation "
            "holds no named members, so nothing finer can be addressed here. A "
            "name given here must not also appear in `edits`."
        ),
    )


class CodingSubmission(StageSubmission[None]):
    """Your ticket responses. The program itself is read from the workspace.

    This ends the coding stage: give it once, after the program in the
    workspace is the one you mean to submit.
    """

    edits: list[None] = Field(
        ...,
        description="Always empty: you revise the program in the workspace.",
    )
    deleted: list[str] = Field(..., description="Always empty.")
    rationale: str | None = Field(..., description="Always null.")

    @model_validator(mode="after")
    def require_an_empty_revision(self) -> Self:
        """Reject a revision, and say which member carried one.

        `rationale` counts as absent when it is blank as well as when it is
        null: a model that means "nothing to say" writes `""` as readily as
        `null`, and rejecting that spends a retry on a distinction the contract
        does not care about. The message names the member that was not empty,
        because it reaches the model as the correction it has to act on -- one
        that only restates the rule reads as a description of what the model
        just did, and it sends the same answer again.
        """
        carried = []
        if self.edits:
            carried.append(f"edits has {len(self.edits)} entries, expected []")
        if self.deleted:
            carried.append(f"deleted names {', '.join(self.deleted)}, expected []")
        if self.rationale is not None and self.rationale.strip():
            carried.append("rationale is a string, expected null")
        if carried:
            raise ValueError(
                f"coding carries no revision of its own ({'; '.join(carried)}): "
                "the program is captured from the workspace through "
                "verification, so only `responses` is yours to fill"
            )
        return self


class ReconstructionSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    open_tickets: list[Ticket] = Field(
        ...,
        min_length=1,
        description=(
            "Every ticket raised for this round, each naming the stages "
            "assigned to answer it."
        ),
    )
    round: int = Field(
        ...,
        ge=0,
        description=(
            "The zero-based round number, equal to this snapshot's position "
            "in ReconstructionRun.snapshots."
        ),
    )
    last_completed_stage: ReasoningStage | None = Field(
        ...,
        description=(
            "The last reasoning stage atomically integrated into this "
            "snapshot, or null before semantics completes. Coding includes "
            "a completed verification attempt, whether it succeeded or failed."
        ),
    )
    semantics: SemanticHypothesis | None = Field(
        ...,
        description=(
            "The complete semantic hypothesis produced in this round, or "
            "null until this round's semantics stage completes. Earlier "
            "hypotheses remain available in preceding snapshots."
        ),
    )
    operations: OperationPlan | None = Field(
        ...,
        description=(
            "The complete operation DAG produced in this round, or null "
            "until this round's operations stage completes. Earlier plans "
            "remain available in preceding snapshots."
        ),
    )
    program_source: str | None = Field(
        ...,
        description=(
            "The complete readable model.py source produced in this round, "
            "or null until coding completes or when no readable program was "
            "produced. Earlier programs remain available in preceding "
            "snapshots."
        ),
    )
    verification: VerifyOutputResult | None = Field(
        ...,
        description=(
            "The terminal result of the verification attempt that closes this "
            "round's coding stage. It is null before that attempt completes. "
            "Its own `source` is always null because the program is kept once, "
            "in `program_source`, and long logs are clipped."
        ),
    )

    @model_validator(mode="after")
    def require_a_consistent_stage_checkpoint(self) -> Self:
        """Keep the artifact checkpoint and every ticket's response prefix aligned."""

        ticket_ids = [ticket.ticket_id for ticket in self.open_tickets]
        if len(ticket_ids) != len(set(ticket_ids)):
            raise ValueError("ticket IDs must be unique within a round")

        # A stage is complete only after it has answered every open ticket.
        # Responses are a chronological list, so no stage may be skipped or
        # appear before its predecessor.
        completed_count = (
            0
            if self.last_completed_stage is None
            else REASONING_STAGES.index(self.last_completed_stage) + 1
        )
        completed_stages = REASONING_STAGES[:completed_count]
        for ticket in self.open_tickets:
            expected_stages = tuple(
                stage for stage in completed_stages if stage in ticket.assigned_stages
            )
            actual_stages = tuple(response.stage for response in ticket.responses)
            if actual_stages != expected_stages:
                raise ValueError(
                    f"{ticket.ticket_id} responses must be "
                    f"{expected_stages}, got {actual_stages}"
                )

        # Previous-round artifacts remain in ReconstructionRun and must not
        # fill a stage that has not completed in this round.
        premature = [
            _ARTIFACT_BY_STAGE[stage]
            for stage in REASONING_STAGES[completed_count:]
            if getattr(self, _ARTIFACT_BY_STAGE[stage]) is not None
        ]
        if premature:
            raise ValueError(
                "unfinished stage artifacts must be null in the current round: "
                + ", ".join(premature)
            )

        # Stage integrity checks
        if PipelineStage.SEMANTICS in completed_stages and self.semantics is None:
            raise ValueError("semantics must exist after semantics")

        if PipelineStage.OPERATIONS in completed_stages and self.operations is None:
            raise ValueError("operations must exist after operations")

        if self.last_completed_stage is PipelineStage.CODING:
            if self.verification is None:
                raise ValueError("verification must exist after coding")
            if self.verification.status is ExecutionStatus.UNINITIALIZED:
                raise ValueError("verification must be completed after coding")
        elif self.verification is not None:
            raise ValueError("verification must be null until coding completes")

        return self


class ReconstructionRun(BaseModel):
    """The complete durable history stored in reconstruction.json."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = Field(
        ...,
        description="The reconstruction.json schema version.",
    )
    run_id: str = Field(
        ...,
        description=(
            "The pipeline-assigned run_... identifier shared by every "
            "snapshot in this reconstruction."
        ),
    )
    snapshots: list[ReconstructionSnapshot] = Field(
        ...,
        min_length=1,
        description=(
            "All round snapshots in chronological order. Only the final "
            "snapshot may still be in progress."
        ),
    )

    @model_validator(mode="after")
    def require_a_consistent_history(self) -> Self:
        if _RUN_ID.fullmatch(self.run_id) is None:
            raise ValueError("run_id must be a run_... identifier")

        rounds = [snapshot.round for snapshot in self.snapshots]
        expected_rounds = list(range(len(self.snapshots)))
        if rounds != expected_rounds:
            raise ValueError(f"snapshot rounds must be {expected_rounds}, got {rounds}")

        incomplete_history = [
            snapshot.round
            for snapshot in self.snapshots[:-1]
            if snapshot.last_completed_stage is not PipelineStage.CODING
        ]
        if incomplete_history:
            raise ValueError(
                "only the latest snapshot may be incomplete; incomplete "
                f"historical rounds: {incomplete_history}"
            )

        ticket_ids = [
            ticket.ticket_id
            for snapshot in self.snapshots
            for ticket in snapshot.open_tickets
        ]
        if len(ticket_ids) != len(set(ticket_ids)):
            raise ValueError("ticket IDs must be unique across the run")

        first_tickets = self.snapshots[0].open_tickets
        if len(first_tickets) != 1 or not isinstance(
            first_tickets[0].subject, BootstrapWork
        ):
            raise ValueError("round 0 must contain exactly one bootstrap ticket")

        later_bootstrap_tickets = [
            ticket.ticket_id
            for snapshot in self.snapshots[1:]
            for ticket in snapshot.open_tickets
            if isinstance(ticket.subject, BootstrapWork)
        ]
        if later_bootstrap_tickets:
            raise ValueError(
                "bootstrap tickets are allowed only in round 0: "
                + ", ".join(later_bootstrap_tickets)
            )

        return self
