"""The semantics contract, and the two things it exists to guarantee.

Mechanically: that the schema stays inside what a provider's strict JSON-schema
mode accepts, because the failure is a 400 at run time rather than anything a
type checker would catch.

Semantically: that a curve claim cannot be made without the numbers that define
it. That is the whole point of the contract -- a stage that could say "arc" and
move on is the stage that was handing on prose.
"""

import pytest
from pydantic import BaseModel, ValidationError

from tests.zeroshot.contracts import evidence, feature, geometry, hypothesis
from zeroshot.pipeline.messages.contracts.semantics import (
    _ARITY,
    _CLAIMED_PARAMETERS,
    _DRAWN_PARAMETERS,
    _EXCLUDED_GEOMETRY,
    DrawnEntity,
    FeatureGeometry,
    GeometryKind,
    Parameter,
    SemanticFeature,
    SemanticHypothesis,
    ViewEvidence,
)

CONTRACTS = [
    SemanticHypothesis,
    SemanticFeature,
    FeatureGeometry,
    ViewEvidence,
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


@pytest.mark.parametrize("kind", list(GeometryKind))
def test_every_kind_states_the_sizes_it_is_measured_by(kind: GeometryKind) -> None:
    """Dropping a size the kind is measured by has to fail, or a claim can be
    made as a bare label -- which is the prose this replaces."""
    geometry(kind)
    for name in _CLAIMED_PARAMETERS[kind]:
        thinned = [p for p in geometry(kind).parameters if p.name.value != name]
        with pytest.raises(ValidationError, match=name):
            FeatureGeometry(
                name=f"geo_{kind.value}",
                kind=kind,
                source="exact",
                axis="z",
                parameters=thinned,
            )


@pytest.mark.parametrize("kind", list(GeometryKind))
def test_no_kind_accepts_a_size_it_is_not_measured_by(kind: GeometryKind) -> None:
    extra = Parameter(name="tube_radius", values=[1.0])
    if extra.name.value in _CLAIMED_PARAMETERS[kind]:
        pytest.skip(f"{kind} is measured by {extra.name.value}")
    with pytest.raises(ValidationError, match="unknown"):
        FeatureGeometry(
            name=f"geo_{kind.value}",
            kind=kind,
            source="exact",
            axis="z",
            parameters=[*geometry(kind).parameters, extra],
        )


def test_a_3d_claim_carries_no_position() -> None:
    """The line this contract draws. A size is read off one view; a position in
    3D is the cross-view correspondence already solved, which is the hard half
    of the task -- and a wrong one is inherited in silence by every stage after
    it, where a wrong radius shows up in the next render."""
    positional = {"center", "start", "end", "point", "base_center", "origin"}
    claimed = {name for row in _CLAIMED_PARAMETERS.values() for name in row}
    assert not positional & claimed
    # The names live in one table now, so the split is only real while the
    # claim side never reaches for a positional one.
    assert positional & {name for row in _DRAWN_PARAMETERS.values() for name in row}


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
            ViewEvidence(
                name=f"ev_{entity.value}",
                view="front",
                entity=entity,
                edge_style="visible",
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
        ViewEvidence(
            name="ev_spline",
            view="front",
            entity="spline",
            edge_style="visible",
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
        ViewEvidence(
            name="ev_spline",
            view="front",
            entity="spline",
            edge_style="visible",
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
    assert GeometryKind.ARC in _CLAIMED_PARAMETERS
    assert "start_angle" in _DRAWN_PARAMETERS[DrawnEntity.ARC]
    assert "start_angle" not in _DRAWN_PARAMETERS[DrawnEntity.CIRCLE]


def test_an_arc_is_bounded_the_way_the_file_bounds_it() -> None:
    """A DXF arc stores `start_angle` and `end_angle`; `start_point` is
    something ezdxf computes from them. Asking for the point asked a reader to
    derive a value and report it as a reading, and it answered with the angle
    it had -- one number where a point takes two, which cost a retry on every
    drawing with an arc in it."""
    assert _ARITY["start_angle"] == 1
    assert _ARITY["end_angle"] == 1
    assert "start" not in _DRAWN_PARAMETERS[DrawnEntity.ARC]


def test_a_feature_may_declare_no_geometry() -> None:
    """A body of straight edges has nothing whose type is at risk. Forcing an
    entry only produced labels -- a live run answered a rectangular plate with
    four `line` claims carrying no parameters at all."""
    assert feature(1, "a rectangular plate", geometry=[]).geometry == []


@pytest.mark.parametrize("kind", [GeometryKind.LINE, GeometryKind.PLANE])
def test_the_kinds_with_no_size_carry_their_claim_in_the_axis(
    kind: GeometryKind,
) -> None:
    """A line and a plane have no size. They are named anyway, because the
    field says what must be present in the built solid and both are: `axis` --
    the direction it runs or the normal it faces -- is the whole claim."""
    assert not _CLAIMED_PARAMETERS[kind]
    claim = geometry(kind)
    assert claim.parameters == []
    assert claim.axis is not None


def test_naming_a_kind_and_excluding_it_are_exclusive() -> None:
    """The two sets together are the decision; overlapping would make it two
    decisions that disagree."""
    named = {kind.value.replace("_", "") for kind in GeometryKind}
    assert not {name.lower() for name in _EXCLUDED_GEOMETRY} & named


@pytest.mark.parametrize("name", ["main_bore", "sem-Main", "sem_"])
def test_a_feature_name_must_be_a_semantic_identity(name: str) -> None:
    with pytest.raises(ValidationError, match="usable sem name"):
        feature(name, "a boss")


def test_a_name_may_carry_a_digit_anywhere_after_its_prefix() -> None:
    """The prefix already keeps a name off anything Python binds, so what
    follows it needs no second rule about where a digit may sit."""
    assert feature("sem_2d_bore", "a boss").name == "sem_2d_bore"
    assert geometry("sphere", name="geo_5radius").name == "geo_5radius"


def test_feature_names_are_unique() -> None:
    with pytest.raises(ValidationError, match="feature names must be unique"):
        hypothesis(
            proposal=[
                feature("sem_main_bore", "a boss"),
                feature("sem_main_bore", "a hole"),
            ]
        )


@pytest.mark.parametrize("member", ["geometry", "evidence"])
def test_member_names_are_unique_within_their_own_group(member: str) -> None:
    """A member name is an address, not a label or list position, so one
    address may not silently name two claims or two readings."""
    overrides = {
        member: [
            geometry("sphere", name="geo_round_end"),
            geometry("cylinder", name="geo_round_end"),
        ]
        if member == "geometry"
        else [
            evidence("circle", name="ev_front_edge"),
            evidence("line", name="ev_front_edge"),
        ]
    }

    with pytest.raises(ValidationError, match=rf"duplicate {member} names"):
        feature(1, "a boss", **overrides)


def test_geometry_and_evidence_names_mark_their_namespace() -> None:
    with pytest.raises(ValidationError, match="usable geo name"):
        geometry("sphere", name="round_end")
    with pytest.raises(ValidationError, match="usable ev name"):
        evidence("circle", name="front_circle")


def test_a_rejected_name_names_the_characters_to_remove() -> None:
    """`lower_snake_case` does not tell a model that wrote geo_baseradius5.79
    that the period alone is the problem."""
    with pytest.raises(ValidationError, match=r"Remove '\.'"):
        geometry("sphere", name="geo_baseradius5.79")


def test_a_feature_is_named_by_identity_and_not_by_its_place_in_the_list() -> None:
    """A stable name survives a revision that drops or reorders a feature, so
    nothing may resolve a feature by list index."""
    revised = hypothesis(
        proposal=[
            feature("sem_main_hole", "a hole"),
            feature("sem_outer_boss", "a boss"),
        ]
    )

    assert [held.name for held in revised.proposal] == [
        "sem_main_hole",
        "sem_outer_boss",
    ]


def test_a_hypothesis_holds_at_least_one_feature() -> None:
    with pytest.raises(ValidationError, match="at least one"):
        hypothesis()


def test_the_2d_evidence_and_the_3d_claim_may_disagree() -> None:
    """A spline in a view is usually the silhouette of a blend, not a spline
    surface. Recording the drawing's entity and the claimed face separately is
    what keeps the stage from building a swept spline where a torus belongs."""
    blend = feature(
        1,
        "shoulder blend",
        geometry=[geometry("torus", major_radius=11.312, tube_radius=2.0)],
        evidence=[evidence("spline")],
    )
    assert blend.geometry[0].kind == "torus"
    assert blend.evidence[0].entity == "spline"


def test_the_rendered_hypothesis_drops_the_parameters_a_kind_does_not_use() -> None:
    """Every field being required means a geometry entry spells out a dozen
    nulls. They are dropped from the rendering, not from the contract, which is
    what keeps the downstream prompts readable."""
    rendered = hypothesis(
        proposal=[
            feature(
                1,
                "bore",
                geometry=[geometry("circle", radius=9.276)],
            )
        ]
    ).model_dump_json(exclude_none=True)
    assert '"values":[9.276]' in rendered
    assert "null" not in rendered


def test_a_hypothesis_feature_must_cite_evidence() -> None:
    """The split puts the exact numbers in the evidence, so a feature with none
    hands the stages after it a type and a size and nothing to place them by.

    The requirement holds of the hypothesis rather than of the feature: a
    feature travels between rounds carrying only the members that changed."""
    partial = feature(1, "a boss", evidence=[])

    with pytest.raises(ValidationError, match="cites no evidence"):
        SemanticHypothesis(proposal=[partial], rationale="the views agree")


def test_the_two_tables_agree_on_every_name_they_share() -> None:
    """One arity table serves the reading and the claim, which is only sound
    while a name they both use means the same shape on either side."""
    drawn = {name for row in _DRAWN_PARAMETERS.values() for name in row}
    claimed = {name for row in _CLAIMED_PARAMETERS.values() for name in row}
    assert drawn & claimed, "the tables share nothing, so they need not be unified"
    assert (drawn | claimed) <= set(_ARITY)
    for name in drawn & claimed:
        assert _ARITY[name] == 1, f"{name} is shared, so it must be a plain size"
