"""Annotate references to interpretation values without copying measurements."""

import difflib
import re
from collections.abc import Iterable
from typing import cast

from pydantic import BaseModel

from zeroshot.pipeline.stages.interpretation.contracts import DrawingInterpretation

_PARAMETER = r"[a-z_][a-z0-9_]*"
# Stable member names by themselves are identities, not value references.
# Consume our prior annotation so re-resolution refreshes rather than duplicates it.
_REFERENCE = re.compile(
    r"\b(?P<member>(?:sem|dim)_[a-z0-9_]+)"
    rf"\.(?P<parameter>{_PARAMETER})\b(?!\.[a-z0-9_])"
    r"(?:\s*\(= [^()]*\))?"
)
_REFERENCE_LIKE = re.compile(r"\b(?:sem|ev|dim)_[a-z0-9_]+(?:\.[a-z0-9_]+)+\b")
_MISSING = object()


def resolve_references[M: BaseModel](
    answer: M, interpretation: DrawingInterpretation | None
) -> M:
    """Copy an answer, annotating each address with its current value."""
    return cast(M, _references_resolved_within(answer, interpretation))


def without_annotations(text: str) -> str:
    """Drop the "(= value)" the pipeline appended after each reference.

    `_REFERENCE` matches an address together with its annotation, and this
    writes the address back alone: "sem_bore.radius (= 3.0)" -> "sem_bore.radius".
    """
    return _REFERENCE.sub(
        lambda address: f"{address['member']}.{address['parameter']}", text
    )


def unresolved_references(
    text: str, interpretation: DrawingInterpretation | None
) -> list[str]:
    """Return unknown addresses; a declared null parameter is still known."""
    return [
        match[0]
        for match in _REFERENCE_LIKE.finditer(text)
        if _value_named(_REFERENCE.fullmatch(match[0]), interpretation) is _MISSING
    ]


def reference_suggestions(
    address: str, interpretation: DrawingInterpretation
) -> list[str]:
    """Up to three legal addresses like an unknown one, in its member if it exists."""
    member, _, parameter = address.partition(".")
    values = _values_by_member(interpretation)
    suggestions: list[str] = []
    for name in [member] if member in values else close_names(member, values):
        # A corrected member may hold the parameter exactly as written.
        known = values[name]
        matches = [parameter] if parameter in known else _similar(parameter, known)
        suggestions += [f"{name}.{match}" for match in matches]
    return suggestions[:3]


def close_names(name: str, known: Iterable[str]) -> list[str]:
    """Up to three known names like a mistyped one, sharing its sem_/op_/... prefix."""
    prefix = name.partition("_")[0] + "_"
    return _similar(name, [other for other in known if other.startswith(prefix)])


def _similar(word: str, known: Iterable[str]) -> list[str]:
    """Close spellings first, then names holding every _-separated part of the word.

    Parts catch `center` in `hole_center_x_mm`, which spelling similarity misses.
    """
    known = list(known)
    parts = set(word.split("_"))
    spelled = difflib.get_close_matches(word, known, cutoff=0.7)
    containing = [other for other in known if parts <= set(other.split("_"))]
    return list(dict.fromkeys([*spelled, *containing]))[:3]


def _references_resolved_within(
    value: object, interpretation: DrawingInterpretation | None
) -> object:
    if isinstance(value, str):
        resolved = _references_resolved_in_prose(value, interpretation)
        # Preserve StrEnum instances when their text did not change.
        return value if resolved == value else resolved
    if isinstance(value, BaseModel):
        return value.model_copy(
            update={
                name: _references_resolved_within(getattr(value, name), interpretation)
                for name in type(value).model_fields
            }
        )
    if isinstance(value, list):
        return [_references_resolved_within(item, interpretation) for item in value]
    if isinstance(value, tuple):
        return tuple(
            _references_resolved_within(item, interpretation) for item in value
        )
    if isinstance(value, dict):
        return {
            key: _references_resolved_within(item, interpretation)
            for key, item in value.items()
        }
    return value


def _references_resolved_in_prose(
    text: str, interpretation: DrawingInterpretation | None
) -> str:
    def substitute(address: re.Match[str]) -> str:
        written = f"{address['member']}.{address['parameter']}"
        value = _value_named(address, interpretation)
        if value is _MISSING:
            return written
        return f"{written} (= {_rendered(value)})"

    return _REFERENCE.sub(substitute, text)


def _value_named(
    address: re.Match[str] | None, interpretation: DrawingInterpretation | None
) -> object:
    if address is None or interpretation is None:
        return _MISSING
    member = _values_by_member(interpretation).get(address["member"], {})
    return member.get(address["parameter"], _MISSING)


def _values_by_member(
    interpretation: DrawingInterpretation,
) -> dict[str, dict[str, object]]:
    """Every legal address: feature parameters and printed figures' values.

    Parameter keys are free-form, but only reference-shaped ones are addressable.
    """
    values: dict[str, dict[str, object]] = {
        candidate.name: {
            key: value
            for key, value in candidate.parameters.items()
            if re.fullmatch(_PARAMETER, key)
        }
        for _, candidate in interpretation.candidates
    }
    for view in interpretation.views:
        for dimension in view.dimensions:
            values[dimension.name] = {
                "nominal_value": dimension.nominal_value,
                "quantity": dimension.quantity,
            }
    return values


def _rendered(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, list):
        return "[" + " ".join(_rendered(item) for item in value) + "]"
    return repr(value)
