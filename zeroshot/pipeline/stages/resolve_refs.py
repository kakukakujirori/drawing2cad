"""Turn the addresses in one stage's answer into the numbers they name.

A stage writes prose, and the numbers it needs are written down once already:
in the drawing that was read, or in the hypothesis built on it. An address
points at one instead of repeating it, and the value is put in here, so no
number is ever retyped and none arrives mistyped.

Which addresses are sound is settled here too. `validate_submission` refuses
whatever `unresolved_references` reports, so one definition serves both the
refusing and the resolving, and the two cannot drift apart.
"""

import re
from collections.abc import Mapping, Sequence
from typing import cast

from pydantic import BaseModel

from zeroshot.pipeline.stages._base.parameters import Parameter
from zeroshot.pipeline.stages.drawings.contracts import DrawingSource
from zeroshot.pipeline.stages.semantics.contracts import (
    SemanticHypothesis,
    render_parameter_values,
)

# Every address this resolves, and what each one is worth:
#
#   sem_main_bore.geo_cylinder.radius   one number of a claim
#   sem_main_bore.geo_cylinder          every number that claim states
#   ev_front_circle.center              one parameter of a drawing entry
#   ev_front_circle.center.x            one number of that point
#   dim_bore_diameter.nominal           what a printed figure states
#
# A claim is scoped by its feature; an entry and a figure are not, their names
# being unique across the whole drawing.
# A drawing name without a parameter is prose: `ev_front_circle` alone is left
# alone, because the drawing stores that very string as an entry's own `name`.
# The address ends where a dot stops carrying on into a name, so one ending a
# sentence is not read as part of it.
# The trailing group takes back the value an earlier pass wrote in, so
# resolving twice rewrites rather than appends; `(= ` marks one as ours.
_REFERENCE = re.compile(
    r"\b(?:(?P<feature>sem_[a-z0-9_]+)\.(?P<claim>geo_[a-z0-9_]+)"
    r"|(?P<read>(?:ev|dim)_[a-z0-9_]+)(?=\.[a-z_]))"
    r"(?:\.(?P<parameter>[a-z_]+)(?:\.(?P<coordinate>[xy]))?)?"
    r"\b(?!\.[a-z_])"
    r"(?:\s*\(= [^()]*\))?"
)
# Anything written as though it were an address, so that a near miss is
# refused rather than passed through as ordinary prose.
_REFERENCE_LIKE = re.compile(r"\b(?:sem|ev|dim)_[a-z0-9_]+(?:\.[a-z0-9_]+)+\b")

type _Held = Mapping[str, Sequence[float]]


def resolve_references[M: BaseModel](
    answer: M,
    hypothesis: SemanticHypothesis | None,
    drawing: DrawingSource,
) -> M:
    """`answer` with the value written beside every address it holds.

    A whole answer, because a ticket summary carries geometry as much as an
    operation detail does. `hypothesis` is null before the round has one, and
    only drawing addresses resolve until it does. Resolving an answer twice
    changes nothing, and a stored one comes back as the next round's input.
    """
    return cast(M, _references_resolved_within(answer, hypothesis, drawing))


def unresolved_references(
    text: str,
    hypothesis: SemanticHypothesis | None,
    drawing: DrawingSource,
) -> list[str]:
    """The addresses in `text` that name nothing the round holds."""
    return [
        candidate[0]
        for candidate in _REFERENCE_LIKE.finditer(text)
        if _numbers_named(_REFERENCE.fullmatch(candidate[0]), hypothesis, drawing)
        is None
    ]


def _references_resolved_within(
    value: object,
    hypothesis: SemanticHypothesis | None,
    drawing: DrawingSource,
) -> object:
    """The same value, with every string anywhere inside it resolved."""
    if isinstance(value, str):
        resolved = _references_resolved_in_prose(value, hypothesis, drawing)
        # `re.sub` hands back a plain `str` even when it changed nothing,
        # and `model_copy` does not validate, so returning that would leave
        # a StrEnum field holding a string that no longer has a `.value`.
        return value if resolved == value else resolved
    if isinstance(value, BaseModel):
        return value.model_copy(
            update={
                name: _references_resolved_within(
                    getattr(value, name), hypothesis, drawing
                )
                for name in type(value).model_fields
            }
        )
    if isinstance(value, list):
        return [
            _references_resolved_within(item, hypothesis, drawing) for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _references_resolved_within(item, hypothesis, drawing) for item in value
        )
    if isinstance(value, dict):
        return {
            key: _references_resolved_within(item, hypothesis, drawing)
            for key, item in value.items()
        }
    return value


def _references_resolved_in_prose(
    text: str,
    hypothesis: SemanticHypothesis | None,
    drawing: DrawingSource,
) -> str:
    """One string, with a value written beside each address it names."""

    def substitute(address: re.Match[str]) -> str:
        # Rebuilt from the groups rather than kept whole, so that a value which
        # no longer holds goes even when the address it sat on names nothing.
        written = ".".join(part for part in address.groups() if part is not None)
        named = _numbers_named(address, hypothesis, drawing)
        stated = _rendered(named, address) if named else ""
        return f"{written} (= {stated})" if stated else written

    return _REFERENCE.sub(substitute, text)


def _numbers_named(
    address: re.Match[str] | None,
    hypothesis: SemanticHypothesis | None,
    drawing: DrawingSource,
) -> dict[str, list[float]] | None:
    """The numbers an address names, or None if it names nothing.

    An empty mapping is a real answer rather than a failure: a plane states no
    size, so naming one is sound and there is no number to write beside it.
    """
    if address is None:
        return None
    held = _held_by(address, hypothesis, drawing)
    if held is None:
        return None
    if (wanted := address["parameter"]) is None:
        return {name: list(values) for name, values in held.items()}
    if wanted not in held:
        return None
    values = list(held[wanted])
    if coordinate := address["coordinate"]:
        # `.x` and `.y` name one number each, so the parameter has to hold two.
        if len(values) != 2:
            return None
        values = [values["xy".index(coordinate)]]
    return {wanted: values}


def _held_by(
    address: re.Match[str],
    hypothesis: SemanticHypothesis | None,
    drawing: DrawingSource,
) -> _Held | None:
    """Every number the member an address opens with holds, by parameter name."""
    if (feature_name := address["feature"]) is not None:
        if hypothesis is None:
            return None
        held = [
            claim
            for feature in hypothesis.proposal
            if feature.name == feature_name
            for claim in feature.geometry
            if claim.name == address["claim"]
        ]
        return _by_name(held[0].parameters) if len(held) == 1 else None

    name = address["read"]
    if name.startswith("ev_"):
        entries = [entry for entry in drawing.evidence() if entry.name == name]
        return _by_name(entries[0].parameters) if len(entries) == 1 else None

    figures = [figure for figure in drawing.dimensions() if figure.name == name]
    if len(figures) != 1:
        return None
    # A printed figure holds no parameter list; these are what it does state.
    return {
        "nominal": [figures[0].nominal],
        "quantity": [float(figures[0].quantity)],
    }


def _by_name(parameters: Sequence[Parameter]) -> _Held:
    return {parameter.name.value: parameter.values for parameter in parameters}


def _rendered(named: Mapping[str, Sequence[float]], address: re.Match[str]) -> str:
    """The numbers those names hold, written as the next stage reads them."""
    if address["parameter"] is None:
        return " ".join(
            f"{name}={render_parameter_values(list(values))}"
            for name, values in named.items()
        )
    (values,) = named.values()
    return render_parameter_values(list(values))
