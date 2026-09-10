from zeroshot.pipeline.messages.contracts.operations import OperationPlan
from zeroshot.pipeline.messages.contracts.reconstruction import OperationSubmission
from zeroshot.pipeline.stages._base.merge import merge_lists
from zeroshot.pipeline.stages._base.validate import SubmissionValidationError


def merge_operations(
    submission: OperationSubmission,
    previous: OperationPlan | None,
) -> OperationPlan:
    previous_operations = previous.proposal if previous is not None else []

    existing_names = {operation.name for operation in previous_operations}
    unknown = set(submission.deleted) - existing_names
    if unknown:
        raise SubmissionValidationError(
            "Cannot delete operations absent from the current plan: "
            + ", ".join(sorted(unknown))
        )

    rationale = submission.rationale
    if rationale is None and previous is not None:
        rationale = previous.rationale
    if rationale is None:
        raise SubmissionValidationError(
            "the first round has no rationale to keep, so state one"
        )

    return OperationPlan(
        proposal=merge_lists(
            previous_operations,
            submission.edits,
            submission.deleted,
        ),
        rationale=rationale,
    )
