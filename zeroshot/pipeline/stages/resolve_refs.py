"""Annotate references to interpretation values without copying measurements."""

import re
from typing import cast

from pydantic import BaseModel

from zeroshot.pipeline.stages.interpretation.contracts import DrawingInterpretation

# Stable member names by themselves are identities, not value references.
# Consume our prior annotation so re-resolution refreshes rather than duplicates it.
_REFERENCE = re.compile(
    r"\b(?P<member>(?:sem|dim)_[a-z0-9_]+)"
    r"\.(?P<parameter>[a-z_][a-z0-9_]*)\b(?!\.[a-z0-9_])"
    r"(?:\s*\(= [^()]*\))?"
)
_REFERENCE_LIKE = re.compile(r"\b(?:sem|ev|dim)_[a-z0-9_]+(?:\.[a-z0-9_]+)+\b")
_MISSING = object()


def resolve_references[M: BaseModel](
    answer: M, interpretation: DrawingInterpretation | None
) -> M:
    """Copy an answer, annotating each address with its current value."""
    return cast(M, _references_resolved_within(answer, interpretation))


def unresolved_references(
    text: str, interpretation: DrawingInterpretation | None
) -> list[str]:
    """Return unknown addresses; a declared null parameter is still known."""
    return [
        match[0]
        for match in _REFERENCE_LIKE.finditer(text)
        if _value_named(_REFERENCE.fullmatch(match[0]), interpretation) is _MISSING
    ]


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
    name, parameter = address["member"], address["parameter"]
    if name.startswith("sem_"):
        for feature in interpretation.features:
            if feature.name == name:
                return feature.parameters.get(parameter, _MISSING)
    else:
        for view in interpretation.views:
            for dimension in view.dimensions:
                if dimension.name == name:
                    return {
                        "nominal_value": dimension.nominal_value,
                        "quantity": dimension.quantity,
                    }.get(parameter, _MISSING)
    return _MISSING


def _rendered(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, list):
        return "[" + " ".join(_rendered(item) for item in value) + "]"
    return repr(value)
