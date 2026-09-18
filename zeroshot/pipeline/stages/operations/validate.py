from zeroshot.pipeline.stages._base.validate import (
    KeyLocation,
    LocatedError,
    SubmissionValidationError,
)
from zeroshot.pipeline.stages.interpretation.contracts import DrawingInterpretation
from zeroshot.pipeline.stages.operations.contracts import OperationPlan
from zeroshot.pipeline.stages.resolve_refs import (
    reference_suggestions,
    unresolved_references,
)


def validate_operations(
    operations: OperationPlan,
    interpretation: DrawingInterpretation | None,
) -> None:
    if interpretation is None:
        raise SubmissionValidationError(
            "operations requires an integrated DrawingInterpretation"
        )
    if errors := _operation_plan_errors(operations, interpretation):
        raise LocatedError(errors)


def _operation_plan_errors(
    plan: OperationPlan,
    interpretation: DrawingInterpretation,
) -> list[tuple[KeyLocation, str]]:
    """Cross-stage contradictions that neither artifact can check alone."""
    established = {feature.name for feature in interpretation.features}
    at = {operation.name: index for index, operation in enumerate(plan.proposal)}
    built = {
        semantic: ("proposal", at[operation.name], "semantics", position)
        for operation in plan.proposal
        for position, semantic in enumerate(operation.semantics)
    }
    errors: list[tuple[KeyLocation, str]] = []

    # A feature no operation builds is missing from the plan, so it has no place in it.
    if uncovered := sorted(established - built.keys()):
        named = ", ".join(uncovered)
        missing = (
            f"The interpretation establishes {named}, and no operation in the plan "
            "builds them. Add the operations that build them."
        )
        errors.append((("proposal",), missing))

    for unknown in sorted(built.keys() - established):
        uncited = (
            f"The plan cites {unknown}, which the interpretation does not contain. "
            "Cite the features it does have."
        )
        errors.append((built[unknown], uncited))

    for operation in sorted(plan.proposal, key=lambda item: item.name):
        for address in dict.fromkeys(
            unresolved_references(operation.detail, interpretation)
        ):
            maybe = reference_suggestions(address, interpretation)
            unknown_reference = f"{operation.name}: unknown reference {address}." + (
                f" Maybe: {', '.join(maybe)}?" if maybe else ""
            )
            errors.append(
                (("proposal", at[operation.name], "detail"), unknown_reference)
            )

    return errors
