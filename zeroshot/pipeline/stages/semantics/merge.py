from zeroshot.pipeline.stages._base.merge import merge_lists
from zeroshot.pipeline.stages._base.validate import SubmissionValidationError
from zeroshot.pipeline.stages.semantics.contracts import SemanticHypothesis
from zeroshot.pipeline.stages.semantics.submission import SemanticSubmission


def merge_semantics(
    submission: SemanticSubmission,
    previous: SemanticHypothesis | None,
) -> SemanticHypothesis:
    previous_features = previous.proposal if previous is not None else []

    existing_names = {feature.name for feature in previous_features}
    unknown = set(submission.deleted) - existing_names
    if unknown:
        raise SubmissionValidationError(
            "Cannot delete features absent from the current hypothesis: "
            + ", ".join(sorted(unknown))
        )

    rationale = submission.rationale
    if rationale is None and previous is not None:
        rationale = previous.rationale
    if rationale is None:
        raise SubmissionValidationError(
            "the first round has no rationale to keep, so state one"
        )

    return SemanticHypothesis(
        proposal=merge_lists(
            previous_features,
            submission.edits,
            submission.deleted,
        ),
        rationale=rationale,
    )
