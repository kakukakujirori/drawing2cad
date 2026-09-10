from pydantic import Field, field_validator

from zeroshot.pipeline.stages._base.parameters import require_name
from zeroshot.pipeline.stages._base.submission import RevisionSubmission
from zeroshot.pipeline.stages.semantics.contracts import SemanticFeature


class SemanticSubmission(RevisionSubmission[SemanticFeature]):
    """Your revision of the semantic hypothesis, and your ticket responses.

    This ends the semantics stage: give it once, after the analysis
    behind it is complete.
    """

    edits: list[SemanticFeature] = Field(
        ...,
        description=(
            "Every feature you changed, each complete and under its stable "
            "sem_ name. A name the hypothesis already holds replaces that "
            "entire feature, and a new name adds one. Include its complete "
            "geometry and evidence lists, description, and open_question. "
            "To remove a claim or citation, submit the complete feature "
            "without it. A feature omitted from edits keeps what it had."
        ),
    )
    deleted: list[str] = Field(
        ...,
        description=(
            "Whole features to delete, by their existing sem_ names, such as "
            "sem_main_bore. Nested addresses such as "
            "sem_main_bore.geo_cylinder are not allowed here; remove a claim "
            "by submitting the complete feature without it in edits. A name "
            "given here must not also appear in edits."
        ),
    )

    @field_validator("deleted")
    @classmethod
    def require_feature_names(cls, names: list[str]) -> list[str]:
        for name in names:
            require_name(name, "sem_")
        return names
