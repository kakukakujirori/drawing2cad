"""The audit stage's answer.

AuditReport
├── ticket_reviews: TicketReview[]
├── concern_reviews: ConcernReview[]
└── findings: AuditFinding[]
    ├── backtrace: CausalHop[]
    │   ├── effect: StageOutputRef
    │   └── cause: StageOutputRef
    └── revision_request: RevisionRequest
        └── targets: StageOutputRef[]
"""

import re
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from zeroshot.pipeline.stages.types import (
    REASONING_STAGES,
    PipelineStage,
    ReasoningStage,
)

type RevisionAction = Literal[
    "add",
    "delete",
    "modify",
    "split",
    "merge",
    "rename",
]


_FIND_NAME = re.compile(r"^find_[a-z0-9_]+$")
_INTERPRETATION_NAME = re.compile(r"^(?:view|dim|sem)_[a-z0-9_]+$")
_OPERATION_NAME = re.compile(r"^op_[a-z0-9_]+$")
_CODE_NAME = re.compile(r"^ret_[a-z0-9_]+$")


# References and revision actions: checks independent of the audited snapshot.


def _valid_member_name(stage: ReasoningStage, name: str) -> bool:
    pattern = {
        PipelineStage.INTERPRETATION: _INTERPRETATION_NAME,
        PipelineStage.OPERATIONS: _OPERATION_NAME,
        PipelineStage.CODING: _CODE_NAME,
    }[stage]
    return pattern.fullmatch(name) is not None


class StageOutputRef(BaseModel):
    """A whole reasoning-stage output or one stable named member within it."""

    model_config = ConfigDict(extra="forbid")

    stage: ReasoningStage = Field(
        ...,
        description="The reasoning stage that owns the referenced output.",
    )
    name: str | None = Field(
        ...,
        description=(
            "The stable member name: view_..., dim_... or sem_... for "
            "interpretation, op_... for operations, and ret_... for coding. The "
            "terminal result variable is not a causal member; use null to "
            "refer to the stage's complete output."
        ),
    )

    @model_validator(mode="after")
    def require_a_name_owned_by_the_stage(self) -> Self:
        if self.name is not None and not _valid_member_name(self.stage, self.name):
            raise ValueError(
                f"{self.name!r} is not a valid member name for {self.stage}"
            )
        return self


class RevisionRequest(BaseModel):
    """The change one finding requires at the root its backtrace reaches."""

    model_config = ConfigDict(extra="forbid")

    action: RevisionAction = Field(
        ...,
        description=(
            "The structural change requested. Use rename only when the stable "
            "identity itself must change. Coding accepts only modify; changing "
            "operation/return identities requires an operations revision."
        ),
    )
    targets: list[StageOutputRef] = Field(
        ...,
        description=(
            "The existing outputs affected by this request. For add, give one "
            "whole-stage reference whose name is null. Modify and delete take "
            "every named member that shares the defect; modify may instead take "
            "one whole-stage reference, but not both at once. Split and rename "
            "take exactly one named member, and merge at least two. Every "
            "target must belong to the same stage. Without an explanation, the "
            "owning stage changes only these targets, the proposed names and "
            "members that cite them, so list every member that must change."
        ),
    )
    instruction: str = Field(
        ...,
        description=(
            "What is wrong with the target and what its owning stage must "
            "correct, without supplying a replacement artifact."
        ),
    )
    proposed_names: list[str] = Field(
        ...,
        description=(
            "Stable names proposed for the result of the action. Add requires "
            "one or more names, split requires at least two, and merge and "
            "rename require exactly one. Modify and delete require an empty "
            "list. Proposed names must be unique across this report. Add and "
            "rename require new names; split and merge may retain their own "
            "target names. Every proposed name must follow the naming convention of "
            "the target stage: view_..., dim_... or sem_... for interpretation, "
            "op_... for operations. Coding only permits modify, so its list is empty."
        ),
    )

    @model_validator(mode="after")
    def require_targets_and_names_appropriate_for_the_action(self) -> Self:
        """Check action shape; existing-name collisions need the snapshot."""
        if not self.instruction.strip():
            raise ValueError("instruction must not be blank")
        if not self.targets:
            raise ValueError("targets must not be empty")
        target_keys = [(target.stage, target.name) for target in self.targets]
        if len(set(target_keys)) != len(target_keys):
            raise ValueError("targets must not contain duplicates")

        stages = {target.stage for target in self.targets}
        if len(stages) != 1:
            raise ValueError("all targets must belong to the same stage")
        stage = self.targets[0].stage
        if stage is PipelineStage.CODING and self.action != "modify":
            raise ValueError(
                "coding accepts only modify; request structural changes at operations"
            )

        if len(set(self.proposed_names)) != len(self.proposed_names):
            raise ValueError("proposed_names must not contain duplicates")
        invalid_names = [
            name for name in self.proposed_names if not _valid_member_name(stage, name)
        ]
        if invalid_names:
            raise ValueError(
                f"invalid proposed names for {stage}: {', '.join(invalid_names)}"
            )

        named_targets = [target for target in self.targets if target.name is not None]
        if self.action == "add":
            if len(self.targets) != 1 or named_targets:
                raise ValueError("add requires one whole-stage target")
            if not self.proposed_names:
                raise ValueError("add requires at least one proposed name")
        elif self.action == "modify":
            if named_targets and len(named_targets) != len(self.targets):
                raise ValueError(
                    "modify requires either named targets or one whole-stage "
                    "target, not both"
                )
            if self.proposed_names:
                raise ValueError("modify does not accept proposed names")
        elif self.action == "delete":
            if len(named_targets) != len(self.targets):
                raise ValueError("delete requires named targets")
            if self.proposed_names:
                raise ValueError("delete does not accept proposed names")
        elif self.action == "split":
            if len(self.targets) != 1 or len(named_targets) != 1:
                raise ValueError("split requires exactly one named target")
            if len(self.proposed_names) < 2:
                raise ValueError("split requires at least two proposed names")
        elif self.action == "merge":
            if len(self.targets) < 2 or len(named_targets) != len(self.targets):
                raise ValueError("merge requires at least two named targets")
            if len(self.proposed_names) != 1:
                raise ValueError("merge requires exactly one proposed name")
        elif self.action == "rename":
            if len(self.targets) != 1 or len(named_targets) != 1:
                raise ValueError("rename requires exactly one named target")
            if len(self.proposed_names) != 1:
                raise ValueError("rename requires exactly one proposed name")
            if self.proposed_names[0] == self.targets[0].name:
                raise ValueError("rename requires a different proposed name")

        return self


# Causal paths: topology is local; declared artifact links need the snapshot.


class CausalHop(BaseModel):
    """One reverse step from an observed downstream effect to its cause."""

    model_config = ConfigDict(extra="forbid")

    effect: StageOutputRef = Field(
        ...,
        description="The downstream stage output in which the problem appears.",
    )
    cause: StageOutputRef = Field(
        ...,
        description="The adjacent output claimed to have caused the effect.",
    )
    rationale: str = Field(
        ...,
        description="Why this cause explains this effect.",
    )

    @model_validator(mode="after")
    def require_a_meaningful_step(self) -> Self:
        """Move within a stage or to its adjacent upstream stage."""
        if self.effect == self.cause:
            raise ValueError("a causal hop must move to a different output")
        distance = REASONING_STAGES.index(self.effect.stage) - REASONING_STAGES.index(
            self.cause.stage
        )
        if distance not in (0, 1):
            raise ValueError(
                "a causal hop must stay within one stage or move to the adjacent "
                "upstream stage: coding -> operations -> interpretation"
            )
        if not self.rationale.strip():
            raise ValueError("rationale must not be blank")
        return self


class AuditFinding(BaseModel):
    """One material defect, its evidence, and the revision it requires."""

    model_config = ConfigDict(extra="forbid")

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
    evidence: list[str] = Field(
        ...,
        description=(
            "Exact locators for the evidence supporting the observation, such as "
            "an original artifact path, view_ name and pixel/UV region, "
            "dim_ name, sem_ parameter reference, operation name or "
            "field, code result variable, or verification-report field. These are "
            "references only, not explanations."
        ),
    )
    backtrace: list[CausalHop] = Field(
        ...,
        description=(
            "The causal path from the observed effect to the revision root, as "
            "adjacent effect-to-cause steps in traversal order. Each hop's cause "
            "moves within a stage or one step upstream along coding -> "
            "operations -> interpretation. An interpretation-internal hop "
            "may name a feature's cited view or dimension. "
            "Each hop's cause "
            "must equal the next hop's effect, and the last cause must be one of "
            "the revision targets. Leave it empty when the defect is already at "
            "its root. Do not revisit an output. Take at most one named-to-named "
            "hop within each prefix (ret_, op_, sem_, dim_, view_); crossing "
            "prefixes, such as sem_ -> dim_ -> view_, is allowed. Whole-stage "
            "references have no prefix and do not count toward that limit."
        ),
    )
    revision_request: RevisionRequest = Field(
        ...,
        description=(
            "The revision this defect requires, at the root its backtrace reaches."
        ),
    )
    related_ticket_ids: list[str] = Field(
        ...,
        description=(
            "If this finding describes the remaining problem of a current defect ticket "
            "you reviewed as unsolved, list that ticket's ID. Use [] for a new defect. "
            "Several tickets may share a finding, and "
            "one ticket may require several findings. These are review links, "
            "not causal backtrace edges."
        ),
    )

    @model_validator(mode="after")
    def require_evidence_and_a_revision_path(self) -> Self:
        """Require evidence and a contiguous, acyclic path ending at the target."""
        if _FIND_NAME.fullmatch(self.name) is None:
            raise ValueError("name must be a find_... lower_snake_case name")
        if not self.observation.strip():
            raise ValueError("observation must not be blank")
        if not self.evidence:
            raise ValueError("evidence must not be empty")
        if any(not locator.strip() for locator in self.evidence):
            raise ValueError("evidence locators must not be blank")
        if len(set(self.evidence)) != len(self.evidence):
            raise ValueError("evidence locators must not contain duplicates")
        if len(set(self.related_ticket_ids)) != len(self.related_ticket_ids):
            raise ValueError("related_ticket_ids must not contain duplicates")

        # A path must be continuous and may not revisit an output.
        for current, following in zip(self.backtrace, self.backtrace[1:], strict=False):
            if current.cause != following.effect:
                raise ValueError(
                    "each causal hop's cause must equal the next hop's effect"
                )
        visited = (
            {(self.backtrace[0].effect.stage, self.backtrace[0].effect.name)}
            if self.backtrace
            else set()
        )
        for hop in self.backtrace:
            key = (hop.cause.stage, hop.cause.name)
            if key in visited:
                raise ValueError("a causal path must not contain a cycle")
            visited.add(key)

        # Limit walks within one member kind, allowing sem -> dim -> view.
        walked_prefixes: set[str] = set()
        for hop in self.backtrace:
            if hop.effect.name is None or hop.cause.name is None:
                continue
            effect_prefix = hop.effect.name.partition("_")[0]
            cause_prefix = hop.cause.name.partition("_")[0]
            if effect_prefix != cause_prefix:
                continue
            if effect_prefix in walked_prefixes:
                raise ValueError(
                    f"a causal path must not step within the {effect_prefix}_ "
                    "prefix more than once"
                )
            walked_prefixes.add(effect_prefix)

        # The requested change must include the root reached by the path.
        if (
            self.backtrace
            and self.backtrace[-1].cause not in self.revision_request.targets
        ):
            raise ValueError(
                "the final causal cause must be one of the revision targets"
            )
        return self


# Report-wide consistency across otherwise independent findings.


class TicketReview(BaseModel):
    """Whether a previously observed defect is resolved in the current artifacts."""

    model_config = ConfigDict(extra="forbid")

    ticket_id: str = Field(
        ...,
        pattern=r"^ticket_[a-z0-9][a-z0-9_]*$",
        description="The open ticket being checked.",
    )
    summary: str = Field(
        ...,
        description=(
            "The current check and why the old defect is resolved or remains. "
            "Do not repeat a finding's backtrace or revision request here."
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
    """How one concern a reasoning stage reported is disposed of."""

    model_config = ConfigDict(extra="forbid")

    concern: str = Field(
        ...,
        description=(
            "The concern being answered, as <reporting_stage>.<concern_id> in "
            "the current stage_reports: coding.concern_bore_diameter. The "
            "prefix names the stage that reported the concern, not the stage "
            "that must change; the finding you link may target another stage."
        ),
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


class AuditReport(BaseModel):
    """The auditor's complete acceptance decision and defect analysis.

    This ends the audit: give it once, after the analysis behind it is
    complete.
    """

    model_config = ConfigDict(extra="forbid")

    accepted: bool = Field(
        ...,
        description=("True only when no reasoning-stage output requires correction."),
    )
    ticket_reviews: list[TicketReview] = Field(
        ...,
        description=("Exactly one review per open ticket."),
    )
    findings: list[AuditFinding] = Field(
        ...,
        description=(
            "Every material defect found; empty only when the reconstruction "
            "is accepted."
        ),
    )
    concern_reviews: list[ConcernReview] = Field(
        ...,
        description=(
            "One review for every concern the current stage_reports raise, and "
            "no others. A concern cannot be left out: give each its own entry, "
            "naming the finding that takes it over or the reason it needs none."
        ),
    )

    @model_validator(mode="after")
    def require_the_decision_to_match_the_findings(self) -> Self:
        """Require one decision and unambiguous finding and proposed identities."""
        if self.accepted == bool(self.findings):
            raise ValueError("accepted must be true exactly when findings is empty")
        review_ids = [review.ticket_id for review in self.ticket_reviews]
        if len(set(review_ids)) != len(review_ids):
            raise ValueError("ticket_reviews must not contain duplicate ticket IDs")
        unsolved = {
            review.ticket_id for review in self.ticket_reviews if not review.solved
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
        concerns = [review.concern for review in self.concern_reviews]
        if len(set(concerns)) != len(concerns):
            raise ValueError("concern_reviews must review each concern once")
        for review in self.concern_reviews:
            if review.finding_name is not None and review.finding_name not in names:
                raise ValueError(
                    f"concern_reviews for {review.concern} names a finding this "
                    f"report does not hold: {review.finding_name}"
                )
        proposed = [
            (finding.revision_request.targets[0].stage, name)
            for finding in self.findings
            for name in finding.revision_request.proposed_names
        ]
        if len(set(proposed)) != len(proposed):
            raise ValueError(
                "proposed_names must be unique across findings for each stage"
            )
        return self
