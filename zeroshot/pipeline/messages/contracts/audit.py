"""The audit stage's answer.

AuditReport
└── findings: AuditFinding[]
    └── backtraces: Backtrace[]
        ├── hops: CausalHop[]
        │   ├── effect: StageOutputRef
        │   └── cause: StageOutputRef
        └── revision_request: RevisionRequest
            └── targets: StageOutputRef[]
"""

import re
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from zeroshot.pipeline.messages.contracts.stages import (
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


_FINDING_NAME = re.compile(r"^finding_[a-z0-9_]+$")
_SEMANTIC_NAME = re.compile(r"^sem_[a-z0-9_]+$")
_OPERATION_NAME = re.compile(r"^op_[a-z0-9_]+$")
_CODE_NAME = re.compile(r"^ret_[a-z0-9_]+$")


def _valid_member_name(stage: ReasoningStage, name: str) -> bool:
    pattern = {
        PipelineStage.SEMANTICS: _SEMANTIC_NAME,
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
            "The stable member name: sem_... for semantics, op_... for "
            "operations, and ret_... for coding. The terminal result variable "
            "is not a causal member; use null to refer to the stage's complete "
            "output."
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
    """One revision intent issued at the root reached by a backtrace."""

    model_config = ConfigDict(extra="forbid")

    action: RevisionAction = Field(
        ...,
        description=(
            "The structural change requested. Use rename only when the stable "
            "identity itself must change."
        ),
    )
    targets: list[StageOutputRef] = Field(
        ...,
        description=(
            "The existing outputs affected by this request. For add, give one "
            "whole-stage reference whose name is null. For modify, delete, "
            "split, and rename, give exactly one target; split, delete, and "
            "rename require a named member, while modify may target either a "
            "member or the complete stage output. For merge, give at least two "
            "named members. Every target must belong to the same stage."
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
            "list. Every proposed name must follow the naming convention of "
            "the target stage."
        ),
    )

    @model_validator(mode="after")
    def require_targets_and_names_appropriate_for_the_action(self) -> Self:
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
            if len(self.targets) != 1:
                raise ValueError("modify requires exactly one target")
            if self.proposed_names:
                raise ValueError("modify does not accept proposed names")
        elif self.action == "delete":
            if len(self.targets) != 1 or len(named_targets) != 1:
                raise ValueError("delete requires exactly one named target")
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

        return self


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
        if self.effect == self.cause:
            raise ValueError("a causal hop must move to a different output")
        if not self.rationale.strip():
            raise ValueError("rationale must not be blank")
        return self


class Backtrace(BaseModel):
    """One ordered causal path from a symptom to one revision root."""

    model_config = ConfigDict(extra="forbid")

    hops: list[CausalHop] = Field(
        ...,
        description=(
            "Adjacent effect-to-cause steps in traversal order. Each hop's "
            "cause must equal the next hop's effect."
        ),
    )
    revision_request: RevisionRequest = Field(
        ...,
        description=("The revision requested at the final cause reached by this path."),
    )

    @model_validator(mode="after")
    def require_a_contiguous_path_to_the_revision_target(self) -> Self:
        for current, following in zip(self.hops, self.hops[1:], strict=False):
            if current.cause != following.effect:
                raise ValueError(
                    "each causal hop's cause must equal the next hop's effect"
                )
        if self.hops and self.hops[-1].cause not in self.revision_request.targets:
            raise ValueError(
                "the final causal cause must be one of the revision targets"
            )
        return self


class AuditFinding(BaseModel):
    """One material defect, its evidence, and its possible causal roots."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        description=("A finding_... lower_snake_case name unique within this report."),
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
            "an artifact path, canonical semantic reference, operation name or "
            "field, code result variable, or verification-report field. These are "
            "references only, not explanations."
        ),
    )
    backtraces: list[Backtrace] = Field(
        ...,
        description=(
            "One causal path for each independently identified revision root."
        ),
    )

    @model_validator(mode="after")
    def require_evidence_and_a_revision_path(self) -> Self:
        if _FINDING_NAME.fullmatch(self.name) is None:
            raise ValueError("name must be a finding_... lower_snake_case name")
        if not self.observation.strip():
            raise ValueError("observation must not be blank")
        if not self.evidence:
            raise ValueError("evidence must not be empty")
        if any(not locator.strip() for locator in self.evidence):
            raise ValueError("evidence locators must not be blank")
        if len(set(self.evidence)) != len(self.evidence):
            raise ValueError("evidence locators must not contain duplicates")
        if not self.backtraces:
            raise ValueError("backtraces must not be empty")
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
    findings: list[AuditFinding] = Field(
        ...,
        description=(
            "Every material defect found; empty only when the reconstruction "
            "is accepted."
        ),
    )

    @model_validator(mode="after")
    def require_the_decision_to_match_the_findings(self) -> Self:
        if self.accepted == bool(self.findings):
            raise ValueError("accepted must be true exactly when findings is empty")
        names = [finding.name for finding in self.findings]
        if len(set(names)) != len(names):
            raise ValueError("finding names must be unique within a report")
        return self
