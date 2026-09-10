from typing import cast

from zeroshot.pipeline.stages._base.validate import SubmissionValidationError
from zeroshot.pipeline.stages.contracts import ReconstructionSnapshot
from zeroshot.pipeline.stages.drawings.contracts import DrawingSource
from zeroshot.pipeline.stages.operations.contracts import OperationPlan
from zeroshot.pipeline.stages.resolve_refs import unresolved_references
from zeroshot.pipeline.stages.semantics.contracts import SemanticHypothesis


def validate_operations(
    operations: OperationPlan,
    snapshot: ReconstructionSnapshot,
) -> None:
    if snapshot.semantics is None:
        raise SubmissionValidationError(
            "operations requires an integrated SemanticHypothesis"
        )
    drawing = cast(DrawingSource, snapshot.drawings)

    # The submitted plan is the operations candidate; the snapshot contains
    # the drawing and the semantics already integrated earlier in this round.
    errors = _operation_plan_errors(operations, snapshot.semantics, drawing)
    if errors:
        raise SubmissionValidationError("\n".join(errors))


def _operation_plan_errors(
    plan: OperationPlan,
    hypothesis: SemanticHypothesis,
    drawing: DrawingSource,
) -> list[str]:
    """Cross-stage contradictions that neither artifact can check alone."""
    established = {feature.name for feature in hypothesis.proposal}
    built = {
        semantic for operation in plan.proposal for semantic in operation.semantics
    }
    errors: list[str] = []

    if uncovered := sorted(established - built):
        named = ", ".join(uncovered)
        errors.append(
            f"The hypothesis establishes {named}, and no operation in the plan "
            "builds them. Add the operations that build them."
        )

    if unknown := sorted(built - established):
        named = ", ".join(unknown)
        errors.append(
            f"The plan cites {named}, which the hypothesis does not contain. "
            "Cite the features it does have."
        )

    for operation in sorted(plan.proposal, key=lambda item: item.name):
        if unresolved := unresolved_references(operation.detail, hypothesis, drawing):
            named = ", ".join(unresolved)
            errors.append(
                f"{operation.name} refers to {named}, which names nothing the "
                "round holds. A claim of a feature is sem_main_bore.geo_cylinder"
                ".radius; an entry or a figure of the drawing stands alone and "
                "must name its parameter, as ev_front_circle.center and "
                "dim_bore_diameter.nominal do. For one number of a point, add "
                ".x or .y: ev_front_circle.center.x."
            )

    return errors
