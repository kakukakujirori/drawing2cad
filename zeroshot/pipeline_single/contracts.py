"""Answers of the single-stage baseline: no tickets, no backtrace."""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CodingReport(BaseModel):
    """The coder's closing answer. This ends the coding stage."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(
        ...,
        description=(
            "What the final program builds, which audit findings it addressed, "
            "and what remains doubtful or incomplete."
        ),
    )


class Finding(BaseModel):
    """One material defect and what the coder must change."""

    model_config = ConfigDict(extra="forbid")

    observation: str = Field(..., description="The concrete mismatch observed.")
    evidence: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "Exact locators: an input or render path with a pixel region, a "
            "printed dimension, or a line of the program."
        ),
    )
    revision_request: str = Field(
        ..., description="What the coder must correct, without writing the code."
    )


class SingleAuditReport(BaseModel):
    """The auditor's decision. This ends the audit."""

    model_config = ConfigDict(extra="forbid")

    accepted: bool = Field(
        ..., description="True only when the solid matches the drawing."
    )
    findings: list[Finding] = Field(
        ...,
        description="Every material defect, largest first; empty only when accepted.",
    )

    @model_validator(mode="after")
    def require_the_decision_to_match_the_findings(self) -> "SingleAuditReport":
        if self.accepted == bool(self.findings):
            raise ValueError("accepted must be true exactly when findings is empty")
        return self
