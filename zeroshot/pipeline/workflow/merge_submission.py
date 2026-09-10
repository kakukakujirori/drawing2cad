"""Merge one reasoning stage's revision onto the artifact it revises."""

from collections.abc import Collection, Mapping, Sequence
from typing import Protocol

from pydantic import ValidationError

from zeroshot.pipeline.messages.contracts import (
    Operation,
    OperationPlan,
    SemanticFeature,
    SemanticHypothesis,
)
from zeroshot.pipeline.messages.contracts.reconstruction import (
    OperationSubmission,
    ProposalSubmission,
    ReconstructionSnapshot,
    SemanticSubmission,
    StageSubmission,
    TicketAnswers,
)
from zeroshot.pipeline.stages._base.validate import SubmissionValidationError
from zeroshot.pipeline.stages.types import PipelineStage, ReasoningStage

type StageArtifact = SemanticHypothesis | OperationPlan


class _Named(Protocol):
    name: str


# Only proposal stages revise through structured diffs. Drawing and coding are
# captured from workspace files before this function is called.
_REVISION_BY_STAGE: Mapping[ReasoningStage, type[StageSubmission]] = {
    PipelineStage.SEMANTICS: SemanticSubmission,
    PipelineStage.OPERATIONS: OperationSubmission,
}


def merge_submission(
    submission: TicketAnswers,
    previous: ReconstructionSnapshot,
    stage: ReasoningStage,
) -> StageArtifact:
    """The complete artifact this stage's edits and deletions produce.

    `previous` is the snapshot this round revises, which is the preceding
    round's, or the snapshot a first round started from. Drawing and coding
    outputs come from workspace verification and do not pass through here.
    """
    if stage not in _REVISION_BY_STAGE:
        raise SubmissionValidationError(
            f"{stage} uses a workspace output, not a structured revision"
        )

    expected = _REVISION_BY_STAGE[stage]
    if not isinstance(submission, expected):
        raise SubmissionValidationError(f"{stage} must submit a {expected.__name__}")

    # Branched on the submission rather than the stage: the check above ties
    # the two together, and only the type says which artifact is being built.
    try:
        if isinstance(submission, SemanticSubmission):
            return SemanticHypothesis(
                proposal=_merged_features(_features(previous.semantics), submission),
                rationale=_rationale(submission, previous.semantics),
            )
        if isinstance(submission, OperationSubmission):
            return OperationPlan(
                proposal=_merged_operations(
                    _operations(previous.operations), submission
                ),
                rationale=_rationale(submission, previous.operations),
            )
        raise NotImplementedError(f"no merge for {type(submission).__name__}")
    except ValidationError as error:
        raise SubmissionValidationError(
            f"the revised artifact is not valid: {error}"
        ) from error


def _rationale(
    submission: ProposalSubmission,
    previous: SemanticHypothesis | OperationPlan | None,
) -> str:
    if submission.rationale is not None:
        return submission.rationale
    if previous is None:
        raise SubmissionValidationError(
            "the first round has no rationale to keep, so state one"
        )
    return previous.rationale


def _features(previous: SemanticHypothesis | None) -> Sequence[SemanticFeature]:
    return previous.proposal if previous is not None else []


def _operations(previous: OperationPlan | None) -> Sequence[Operation]:
    return previous.proposal if previous is not None else []


def _merged_operations(
    previous: Sequence[Operation],
    submission: StageSubmission,
) -> list[Operation]:
    if addressed := sorted(a for a in submission.deleted if "." in a):
        raise SubmissionValidationError(
            f"{', '.join(addressed)} is not an address in the current plan: an "
            "operation has no members, so delete it by its own op_ name"
        )
    dropped = _dropped_entries(submission.deleted, previous, "operation")
    return _merged_list(previous, _by_name(submission.edits), dropped)


def _merged_features(
    previous: Sequence[SemanticFeature],
    submission: StageSubmission,
) -> list[SemanticFeature]:
    dropped = _dropped_entries(submission.deleted, previous, "feature")
    dropped_members = _dropped_members(submission.deleted, previous)

    known = _by_name(previous)
    revised = {
        edit.name: (
            edit
            if edit.name not in known
            else _revised_feature(
                known[edit.name], edit, dropped_members.get(edit.name, ())
            )
        )
        for edit in submission.edits
    }
    # A feature whose members were dropped but which the edits left alone
    # still has to be rebuilt without them.
    revised |= {
        name: _revised_feature(known[name], known[name], members)
        for name, members in dropped_members.items()
        if name not in revised
    }
    return _merged_list(previous, revised, dropped)


def _revised_feature(
    base: SemanticFeature,
    edit: SemanticFeature,
    dropped: Collection[str],
) -> SemanticFeature:
    return SemanticFeature(
        name=edit.name,
        description=edit.description,
        open_question=edit.open_question,
        geometry=_merged_list(base.geometry, _by_name(edit.geometry), dropped),
        # Not merged: `evidence` is a citation list rather than named members,
        # so a feature given again says what it now rests on, whole.
        evidence=list(edit.evidence),
    )


def _by_name[T: _Named](members: Sequence[T]) -> dict[str, T]:
    return {member.name: member for member in members}


def _merged_list[T: _Named](
    previous: Sequence[T],
    revised: Mapping[str, T],
    dropped: Collection[str],
) -> list[T]:
    """Revised members keep their place; new ones follow in the order given."""
    existing = {member.name for member in previous}
    kept = [
        revised.get(member.name, member)
        for member in previous
        if member.name not in dropped
    ]
    added = [member for name, member in revised.items() if name not in existing]
    return [*kept, *added]


def _dropped_entries(
    deleted: Sequence[str],
    previous: Sequence[_Named],
    what: str,
) -> set[str]:
    known = {member.name for member in previous}
    entries = {address for address in deleted if "." not in address}
    if unknown := sorted(entries - known):
        raise SubmissionValidationError(
            f"cannot delete {', '.join(unknown)}: the current artifact holds "
            f"no such {what}"
        )
    return entries


def _dropped_members(
    deleted: Sequence[str],
    previous: Sequence[SemanticFeature],
) -> dict[str, set[str]]:
    """The geo_ claims dropped from each feature, checked against it."""
    known = _by_name(previous)
    dropped: dict[str, set[str]] = {}
    for address in deleted:
        if "." not in address:
            continue
        feature_name, _, member_name = address.partition(".")
        feature = known.get(feature_name)
        if feature is None or "." in member_name:
            raise SubmissionValidationError(
                f"{address} is not an address in the current hypothesis: name a "
                "whole feature as sem_main_bore, or one of its claims as "
                "sem_main_bore.geo_cylinder. A citation is dropped by giving "
                "the feature again without it."
            )
        if member_name not in _by_name(feature.geometry):
            raise SubmissionValidationError(
                f"cannot delete {address}: {feature_name} has no claim "
                f"called {member_name}"
            )
        dropped.setdefault(feature_name, set()).add(member_name)
    return dropped
