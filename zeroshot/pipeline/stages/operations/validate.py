from zeroshot.pipeline.stages._base.validate import SubmissionValidationError
from zeroshot.pipeline.stages.interpretation.contracts import DrawingInterpretation
from zeroshot.pipeline.stages.operations.contracts import OperationPlan
from zeroshot.pipeline.stages.resolve_refs import unresolved_references


def validate_operations(
    operations: OperationPlan,
    interpretation: DrawingInterpretation | None,
) -> None:
    if interpretation is None:
        raise SubmissionValidationError(
            "operations requires an integrated DrawingInterpretation"
        )
    errors = _operation_plan_errors(operations, interpretation)
    if errors:
        raise SubmissionValidationError("\n".join(errors))


def _operation_plan_errors(
    plan: OperationPlan,
    interpretation: DrawingInterpretation,
) -> list[str]:
    """Cross-stage contradictions that neither artifact can check alone."""
    established = {feature.name for feature in interpretation.features}
    built = {
        semantic for operation in plan.proposal for semantic in operation.semantics
    }
    errors: list[str] = []

    if uncovered := sorted(established - built):
        named = ", ".join(uncovered)
        errors.append(
            f"The interpretation establishes {named}, and no operation in the plan "
            "builds them. Add the operations that build them."
        )

    if unknown := sorted(built - established):
        named = ", ".join(unknown)
        errors.append(
            f"The plan cites {named}, which the interpretation does not contain. "
            "Cite the features it does have."
        )

    for operation in sorted(plan.proposal, key=lambda item: item.name):
        if unresolved := unresolved_references(operation.detail, interpretation):
            named = ", ".join(unresolved)
            errors.append(
                f"{operation.name} refers to {named}, which names nothing the "
                "round holds. Cite a feature parameter as sem_main_bore.radius "
                "or a printed figure as dim_bore_diameter.nominal_value. "
                "Use the parameter names recorded in the interpretation."
            )

    return errors
