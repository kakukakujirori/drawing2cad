from collections.abc import Collection, Iterable
from typing import Protocol


class _Named(Protocol):
    name: str


def index_by_name[T: _Named](members: Iterable[T]) -> dict[str, T]:
    return {member.name: member for member in members}


def merge_lists[T: _Named](
    previous: Iterable[T],
    edits: Iterable[T],
    deleted_names: Collection[str] = (),
) -> list[T]:
    edits_by_name = index_by_name(edits)
    assert edits_by_name.keys().isdisjoint(deleted_names), (
        "cannot edit and delete the same member."
    )
    merged = index_by_name(previous) | edits_by_name
    return [item for name, item in merged.items() if name not in deleted_names]
