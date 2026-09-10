from pydantic import Field, field_validator

from zeroshot.pipeline.stages._base.parameters import require_name
from zeroshot.pipeline.stages._base.submission import RevisionSubmission
from zeroshot.pipeline.stages.operations.contracts import Operation


class OperationSubmission(RevisionSubmission[Operation]):
    """Your revision of the operation plan, and your ticket responses.

    This ends the operations stage: give it once, after the analysis
    behind it is complete.
    """

    edits: list[Operation] = Field(
        ...,
        description=(
            "Every operation you changed, each complete and under its stable "
            "op_ name: a name the plan already holds replaces that operation, "
            "and a new name adds one. An operation you leave out keeps what it "
            "had."
        ),
    )
    deleted: list[str] = Field(
        ...,
        description=(
            "Whole operations to delete, by their existing op_ names. Nested "
            "addresses such as op_bore.detail are not allowed here; change "
            "fields by submitting the complete operation in edits. A name "
            "given here must not also appear in edits."
        ),
    )

    @field_validator("deleted")
    @classmethod
    def require_operation_names(cls, names: list[str]) -> list[str]:
        for name in names:
            require_name(name, "op_")
        return names
