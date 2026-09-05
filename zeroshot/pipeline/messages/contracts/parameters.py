"""Numeric parameters shared by 2D drawing entities and 3D shape claims."""

from collections.abc import Mapping, Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

# Zero denotes a variable-length numeric list.
_ARITY = {
    "start": 2,
    "end": 2,
    "center": 2,
    "major_axis": 2,
    "radius": 1,
    "major_radius": 1,
    "minor_radius": 1,
    "tube_radius": 1,
    "base_radius": 1,
    "top_radius": 1,
    "height": 1,
    "degree": 1,
    # Below is a variable-length list. 0 is just a placeholder.
    "control_points": 0,
    "vertices": 0,
    "knots": 0,
}

_POINT_LISTS = frozenset({"control_points", "vertices"})

_VECTORS = frozenset({"major_axis"})


ParameterName = StrEnum("ParameterName", {name.upper(): name for name in _ARITY})


class Parameter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: ParameterName = Field(
        ..., description="Parameter name from the set required by its entity or claim."
    )
    values: list[float] = Field(
        ...,
        description=(
            "One number for a scalar; x, y for a point or vector; flattened "
            "x, y pairs for control_points and vertices; the full knot vector "
            "for knots. Supply every required parameter exactly once."
        ),
    )


def rows[K: StrEnum](table: Mapping[K, tuple[str, ...]]) -> str:
    """Render the parameter sets enforced by the validator."""
    return "\n".join(
        f"  {kind.value} takes {', '.join(names) or 'nothing'}"
        for kind, names in table.items()
    )


def require_parameters(
    subject: str, expected: tuple[str, ...], parameters: Sequence[Parameter]
) -> dict[str, list[float]]:
    """Validate names and arity, then index the values by name."""
    given = {parameter.name.value: parameter.values for parameter in parameters}
    if len(given) != len(parameters):
        raise ValueError("a parameter is given twice")
    missing = [name for name in expected if name not in given]
    unknown = [name for name in given if name not in expected]
    if missing or unknown:
        raise ValueError(
            f"{subject} takes {list(expected)}"
            + (f"; missing {missing}" if missing else "")
            + (f"; unknown {unknown}" if unknown else "")
        )
    for name, values in given.items():
        arity = _ARITY[name]
        if arity and len(values) != arity:
            raise ValueError(f"{name} takes {arity} number(s), got {len(values)}")
        if not arity and not values:
            raise ValueError(f"{name} cannot be empty")
        if name in _POINT_LISTS and len(values) % 2:
            raise ValueError(f"{name} must contain x, y pairs")
        if name in _VECTORS and not any(values):
            raise ValueError(f"{name} must be a non-zero vector")
    return given
