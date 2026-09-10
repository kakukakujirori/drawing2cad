from zeroshot.pipeline.messages.contracts import (
    DrawingSource,
    SemanticHypothesis,
)
from zeroshot.pipeline.stages._base.validate import SubmissionValidationError


def validate_semantics(
    hypothesis: SemanticHypothesis,
    drawing: DrawingSource,
) -> None:
    """Require each feature's citations to exist in the current drawing."""
    known = drawing.cited_names()
    errors = [
        f"{feature.name} cites {', '.join(missing)}, which the drawing does not hold"
        for feature in hypothesis.proposal
        if (missing := sorted(set(feature.evidence) - known))
    ]
    if errors:
        raise SubmissionValidationError(
            "\n".join(errors)
            + "\nCite an entry or printed figure by the name the drawing gives it, "
            "and report a "
            "reading you needed and could not find in `open_question`."
        )
