"""The drawing contract, and the two things it exists to guarantee.

Mechanically: that the schema stays inside what a provider's strict JSON-schema
mode accepts, because the failure is a 400 at run time rather than anything a
type checker would catch.

Semantically: that a curve reading cannot be made without the numbers that
define it. That is the whole point of the contract -- a stage that could say
"arc" and move on is the stage that was handing on prose.
"""

import pytest
from pydantic import BaseModel, ValidationError

from tests.zeroshot.contracts import evidence
from zeroshot.pipeline.stages._base.parameters import _ARITY, Parameter
from zeroshot.pipeline.stages.drawings.contracts import (
    _DRAWN_PARAMETERS,
    DrawingEvidence,
    DrawnEntity,
)

CONTRACTS = [
    DrawingEvidence,
    Parameter,
]

# Emitted by `Field(ge=...)`, `min_length=`, a default, or a tuple annotation.
# OpenAI's strict mode rejects every one of them.
_UNSUPPORTED_KEYWORDS = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minItems",
        "maxItems",
        "minLength",
        "maxLength",
        "pattern",
        "format",
        "default",
        "prefixItems",
        "oneOf",
        "not",
    }
)


def _object_schemas(node: object) -> list[dict]:
    """Every object schema in the document, `$defs` included."""
    if isinstance(node, dict):
        found = [node] if node.get("type") == "object" else []
        for held in node.values():
            found.extend(_object_schemas(held))
        return found
    if isinstance(node, list):
        return [found for held in node for found in _object_schemas(held)]
    return []


def _keywords(node: object) -> set[str]:
    if isinstance(node, dict):
        return set(node) | {key for held in node.values() for key in _keywords(held)}
    if isinstance(node, list):
        return {key for held in node for key in _keywords(held)}
    return set()


@pytest.mark.parametrize("contract", CONTRACTS)
def test_every_property_is_required_so_strict_output_accepts_the_schema(
    contract: type[BaseModel],
) -> None:
    """Strict mode requires every property to appear in `required`. A field
    written `x: T | None = None` silently drops out of it, so this is checked
    mechanically rather than left to whoever adds the next field."""
    for schema in _object_schemas(contract.model_json_schema()):
        assert set(schema.get("required", [])) == set(schema.get("properties", {}))
        assert schema.get("additionalProperties") is False


@pytest.mark.parametrize("contract", CONTRACTS)
def test_the_schema_avoids_keywords_strict_output_rejects(
    contract: type[BaseModel],
) -> None:
    """Constraints belong in a `model_validator`, whose message reaches the
    model as a correction, not in `Field`, where they become schema keywords
    the provider refuses."""
    assert not _UNSUPPORTED_KEYWORDS & _keywords(contract.model_json_schema())


@pytest.mark.parametrize("entity", list(DrawnEntity))
def test_every_drawn_entity_states_the_parameters_it_carries(
    entity: DrawnEntity,
) -> None:
    """Evidence is transcription: every number is a field of the DXF entity, so
    it can be checked against the file. That is why the exact geometry lives
    here rather than in the 3D claim."""
    required = _DRAWN_PARAMETERS[entity]
    assert required, f"{entity} states no parameters"
    evidence(entity)

    for name in required:
        thinned = [p for p in evidence(entity).parameters if p.name.value != name]
        with pytest.raises(ValidationError, match=name):
            DrawingEvidence(
                name=f"ev_{entity.value}",
                entity=entity,
                edge_style="visible",
                source=[],
                parameters=thinned,
            )


def test_a_reading_is_in_sheet_coordinates_not_model_coordinates() -> None:
    """Two numbers, not three. A reading that carried a z has stopped being a
    transcription of the drawing and become a claim about the solid."""
    evidence("circle")
    with pytest.raises(ValidationError, match="takes 2"):
        evidence("circle", center=[0.0, 0.0, 0.0])


def test_a_spline_reading_must_carry_its_poles() -> None:
    """The measured failure in one sentence: a spline nobody parameterised came
    back as a sampled polyline, right volume and wrong faces."""
    evidence("spline")
    with pytest.raises(ValidationError, match="control_points"):
        DrawingEvidence(
            name="ev_spline",
            entity="spline",
            edge_style="visible",
            source=[],
            parameters=[Parameter(name="degree", values=[3.0])],
        )
    with pytest.raises(ValidationError, match="x, y pairs"):
        evidence("spline", control_points=[0.0, 1.0, 2.0])


def test_a_spline_reading_must_carry_its_knot_vector() -> None:
    """Control points and a degree do not determine a spline: 197 of the
    corpus's 287 have a non-uniform knot vector, and rebuilding those poles on
    a uniform one gives a different curve. Carrying the poles but not the knots
    is the polyline failure again, one step further along -- a curve that is
    smooth, exact-looking, and not the one on the drawing."""
    with pytest.raises(ValidationError, match="knots"):
        DrawingEvidence(
            name="ev_spline",
            entity="spline",
            edge_style="visible",
            source=[],
            parameters=[
                Parameter(name="control_points", values=[0.0, 0.0, 1.0, 1.0]),
                Parameter(name="degree", values=[3.0]),
            ],
        )


def test_a_knot_vector_is_not_held_to_the_shape_of_a_point_list() -> None:
    """Its length is one per control point plus the degree plus one, so it is
    odd as often as it is even -- the `x, y` pairing the other lists are held
    to would reject half of them."""
    reading = evidence("spline", knots=[0.0, 0.0, 0.0, 1.0, 1.0])

    assert len(reading.parameters[-1].values) == 5

    with pytest.raises(ValidationError, match="cannot be empty"):
        evidence("spline", knots=[])


def test_an_arc_is_its_own_kind() -> None:
    """OCC stores an arc as a bounded circle, but the contract is what the
    model reasons in, and the two read differently off a drawing."""
    assert "start" in _DRAWN_PARAMETERS[DrawnEntity.ARC]
    assert "start" not in _DRAWN_PARAMETERS[DrawnEntity.CIRCLE]


def test_an_arc_is_bounded_the_way_the_file_bounds_it() -> None:
    """The drawing contract is format-independent page geometry. Its DXF
    reader converts native angles to endpoints, and its writer converts the
    endpoints back, so downstream stages use the same coordinates as every
    other drawn entity."""
    assert _ARITY["start"] == 2
    assert _ARITY["end"] == 2
    assert "start_angle" not in _DRAWN_PARAMETERS[DrawnEntity.ARC]
