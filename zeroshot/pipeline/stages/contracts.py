"""Durable reconstruction snapshots shared across revision rounds."""

import re
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from zeroshot.pipeline.messages.tickets import BootstrapWork, Ticket
from zeroshot.pipeline.stages.drawings.contracts import DrawingSource
from zeroshot.pipeline.stages.operations.contracts import OperationPlan
from zeroshot.pipeline.stages.semantics.contracts import SemanticHypothesis
from zeroshot.pipeline.stages.types import (
    REASONING_STAGES,
    STAGE_ARTIFACT_FIELDS,
    PipelineStage,
    ReasoningStage,
)
from zeroshot.pipeline.verification import ExecutionStatus, VerifyOutputResult

_RUN_ID = re.compile(r"^run_[a-z0-9][a-z0-9_]*$")


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
            "snapshot, or null before the drawing is read. Coding includes "
            "a completed verification attempt, whether it succeeded or failed."
        ),
    )
    drawings: DrawingSource | None = Field(
        ...,
        description=(
            "The complete DrawingSource produced in this round, or null until "
            "this round's drawing stage completes. Earlier readings remain "
            "available in preceding snapshots."
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
            artifact
            for stage, artifacts in STAGE_ARTIFACT_FIELDS.items()
            if stage not in completed_stages
            for artifact in artifacts
            if getattr(self, artifact) is not None
        ]
        if premature:
            raise ValueError(
                "unfinished stage artifacts must be null in the current round: "
                + ", ".join(premature)
            )

        # Stage integrity checks
        if PipelineStage.DRAWINGS in completed_stages and self.drawings is None:
            raise ValueError("drawings must exist after drawings")

        if PipelineStage.SEMANTICS in completed_stages and self.semantics is None:
            raise ValueError("semantics must exist after semantics")

        if PipelineStage.OPERATIONS in completed_stages and self.operations is None:
            raise ValueError("operations must exist after operations")

        if self.last_completed_stage is PipelineStage.CODING:
            if self.verification is None:
                raise ValueError("verification must exist after coding")
            if self.verification.status is ExecutionStatus.UNINITIALIZED:
                raise ValueError("verification must be completed after coding")
        return self


class ReconstructionRun(BaseModel):
    """The complete durable history stored in reconstruction.json."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(
        ...,
        description=(
            "The pipeline-assigned run_... identifier shared by every "
            "snapshot in this reconstruction."
        ),
    )
    input_drawings: DrawingSource = Field(
        ...,
        description=(
            "The immutable drawing supplied to the run, before the drawing "
            "stage separates or transcribes it."
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
