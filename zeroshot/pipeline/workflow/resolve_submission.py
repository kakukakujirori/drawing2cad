"""Turn the addresses in one stage's answer into the numbers they name.

A stage writes prose, and the geometry it needs is already written down once
in the hypothesis. An address points at it instead of repeating it, and the
value is put in here, so no number is ever retyped and none arrives mistyped.

Which addresses are sound is settled here too. `validate_submission` refuses
whatever `unresolved_references` reports, so one definition serves both the
refusing and the resolving, and the two cannot drift apart.
"""

import re
from typing import cast

from pydantic import BaseModel

from zeroshot.pipeline.messages.contracts.semantics import (
    Parameter,
    SemanticFeature,
    SemanticHypothesis,
    render_parameter_values,
)

# An address names the feature, the claim or reading within it, and the
# parameter: `sem_main_bore.geo_cylinder.radius`. The parameter takes `.x` or
# `.y` for one number of a point, and may be left off to mean the whole
# reading. The trailing group takes back a value an earlier pass wrote in, so
# resolving again rewrites that value rather than appending a second one;
# `(= ` is what tells one of them from a parenthesis of the model's own.
_REFERENCE = re.compile(
    r"\b(sem_[a-z0-9_]+)\.((?:geo|ev)_[a-z0-9_]+)"
    r"(?:\.([a-z_]+)(?:\.([xy]))?)?"
    # A dot carrying on into a name continues the address; one ending a
    # sentence does not.
    r"\b(?!\.[a-z_])"
    r"(?:\s*\(= [^()]*\))?"
)
# Anything written as though it were an address, so that a near miss is
# refused rather than passed through as ordinary prose.
_REFERENCE_LIKE = re.compile(r"\bsem_[a-z0-9_]+(?:\.[a-z0-9_]+)+\b")
_RESOLVED_VALUE = re.compile(r"\s*\(= [^()]*\)")


def without_resolved_values(text: str) -> str:
    """`text` with the values this module wrote into it taken back out."""
    return _RESOLVED_VALUE.sub("", text)


def resolve_references[M: BaseModel](answer: M, hypothesis: SemanticHypothesis) -> M:
    """`answer` with the value written beside every address it holds.

    A whole answer, because a ticket summary carries geometry as much as an
    operation detail does. Resolving an answer twice changes nothing, and a
    stored one comes back as the next round's input.
    """
    return cast(M, _references_resolved_within(answer, hypothesis))


def unresolved_references(text: str, hypothesis: SemanticHypothesis) -> list[str]:
    """The addresses in `text` that name nothing the hypothesis holds."""
    features = {feature.name: feature for feature in hypothesis.proposal}
    unresolved: list[str] = []
    for candidate in _REFERENCE_LIKE.finditer(text):
        address = _REFERENCE.fullmatch(candidate[0])
        feature = features.get(address[1]) if address is not None else None
        if (
            address is None
            or feature is None
            or _parameters_named(feature, address) is None
        ):
            unresolved.append(candidate[0])
    return unresolved


def _references_resolved_within(
    value: object, hypothesis: SemanticHypothesis
) -> object:
    """The same value, with every string anywhere inside it resolved."""
    if isinstance(value, str):
        resolved = _references_resolved_in_prose(value, hypothesis)
        # `re.sub` hands back a plain `str` even when it changed nothing,
        # and `model_copy` does not validate, so returning that would leave
        # a StrEnum field holding a string that no longer has a `.value`.
        return value if resolved == value else resolved
    if isinstance(value, BaseModel):
        return value.model_copy(
            update={
                name: _references_resolved_within(getattr(value, name), hypothesis)
                for name in type(value).model_fields
            }
        )
    if isinstance(value, list):
        return [_references_resolved_within(item, hypothesis) for item in value]
    if isinstance(value, tuple):
        return tuple(_references_resolved_within(item, hypothesis) for item in value)
    if isinstance(value, dict):
        return {
            key: _references_resolved_within(item, hypothesis)
            for key, item in value.items()
        }
    return value


def _references_resolved_in_prose(text: str, hypothesis: SemanticHypothesis) -> str:
    """One string, with a value written beside each address it names."""
    features = {feature.name: feature for feature in hypothesis.proposal}

    def substitute(address: re.Match[str]) -> str:
        # Rebuilt from the groups rather than kept whole, so that a value which
        # no longer holds goes even when the address it sat on names nothing.
        written = ".".join(part for part in address.groups() if part is not None)
        feature = features.get(address[1])
        named = _parameters_named(feature, address) if feature is not None else None
        stated = _rendered_values(named, address) if named else ""
        return f"{written} (= {stated})" if stated else written

    return _REFERENCE.sub(substitute, text)


def _parameters_named(
    feature: SemanticFeature, address: re.Match[str]
) -> list[Parameter] | None:
    """The parameters an address names, or None if it names nothing.

    An empty list is a real answer rather than a failure: a plane states no
    size, so naming one is sound and there is no number to write beside it.
    """
    member_name = address[2]
    members = feature.geometry if member_name.startswith("geo_") else feature.evidence
    named = [member for member in members if member.name == member_name]
    if len(named) != 1:
        return None

    parameters = named[0].parameters
    if address[3] is None:
        return list(parameters)

    wanted = [
        parameter for parameter in parameters if parameter.name.value == address[3]
    ]
    # `.x` and `.y` name one number each, so the parameter has to hold two.
    if len(wanted) != 1 or (address[4] and len(wanted[0].values) != 2):
        return None
    return wanted


def _rendered_values(parameters: list[Parameter], address: re.Match[str]) -> str:
    """The numbers those parameters hold, written as the next stage reads them."""
    if address[3] is None:
        return " ".join(
            f"{parameter.name.value}={render_parameter_values(parameter.values)}"
            for parameter in parameters
        )
    values = parameters[0].values
    if coordinate := address[4]:
        values = [values["xy".index(coordinate)]]
    return render_parameter_values(values)
