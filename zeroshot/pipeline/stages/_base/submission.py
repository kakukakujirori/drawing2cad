from typing import Protocol, Self, runtime_checkable

from pydantic import ConfigDict, Field, model_validator

from zeroshot.pipeline.messages.tickets import TicketAnswers


@runtime_checkable
class _Named(Protocol):
    name: str


class RevisionSubmission[M: _Named](TicketAnswers):
    """One stage's revision of its artifact, and its answers to its tickets.

    A model reads the concrete subclass, never this: pydantic takes a schema
    description from the class it is asked for, so the wording that names
    sheet_, sem_, op_ or the workspace program belongs to the subclass that
    means it.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    edits: list[M] = Field(
        ...,
        description=(
            "The members this stage changed, each complete and under its own "
            "stable name. A name the artifact already holds replaces that "
            "member, a new name adds one, and a member left out keeps what it "
            "had."
        ),
    )
    deleted: list[str] = Field(
        ...,
        description=(
            "The members this stage dropped, by the names the artifact holds "
            "them under. A name given here must not also appear in `edits`."
        ),
    )
    rationale: str | None = Field(
        ...,
        description=(
            "The artifact's rationale, rewritten when this revision changed "
            "the reasoning behind it, or null to keep the one it already has. "
            "The first round has none to keep, so it must be stated there."
        ),
    )

    @classmethod
    def unchanged(cls) -> Self:
        """A revision that leaves the preceding round's artifact as it is."""
        return cls(edits=[], deleted=[], rationale=None, responses=[])

    @model_validator(mode="after")
    def require_distinct_edited_and_deleted_names(self) -> Self:
        named = [edit.name for edit in self.edits]
        duplicated = sorted({name for name in named if named.count(name) > 1})
        if duplicated:
            raise ValueError(f"edits name {', '.join(duplicated)} more than once")
        if len(set(self.deleted)) != len(self.deleted):
            raise ValueError("deleted must not name the same address twice")
        both = sorted(set(named) & set(self.deleted))
        if both:
            raise ValueError(f"{', '.join(both)} is both edited and deleted")
        return self
