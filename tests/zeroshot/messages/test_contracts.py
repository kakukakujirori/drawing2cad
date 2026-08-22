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
from zeroshot.pipeline.messages.contracts import (
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
    render_hypothesis,
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
            FeatureGeometry(kind=kind, source="exact", axis="z", parameters=thinned)


@pytest.mark.parametrize("kind", list(GeometryKind))
def test_no_kind_accepts_a_size_it_is_not_measured_by(kind: GeometryKind) -> None:
    extra = Parameter(name="tube_radius", values=[1.0])
    if extra.name.value in _CLAIMED_PARAMETERS[kind]:
        pytest.skip(f"{kind} is measured by {extra.name.value}")
    with pytest.raises(ValidationError, match="unknown"):
        FeatureGeometry(
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
                view="front", entity=entity, edge_style="visible", parameters=thinned
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
    assert "start" in _DRAWN_PARAMETERS[DrawnEntity.ARC]
    assert "start" not in _DRAWN_PARAMETERS[DrawnEntity.CIRCLE]


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


def test_a_feature_id_starts_at_one() -> None:
    with pytest.raises(ValidationError, match="1 or greater"):
        feature(0, "a boss")


def test_feature_ids_are_unique() -> None:
    """Two proposers each number from 1, so the reducer has to renumber. The
    message is what tells it to."""
    with pytest.raises(ValidationError, match="renumber"):
        hypothesis(proposal=[feature(1, "a boss"), feature(1, "a hole")])


def test_a_feature_is_named_by_its_id_and_not_by_its_place_in_the_list() -> None:
    """Two numberings ride on `proposal`: the ids the model chose and the
    positions the list gives it for free. They are allowed to disagree, because
    an id has to survive a revision that drops or reorders a feature -- so
    nothing may resolve a feature by index."""
    revised = hypothesis(proposal=[feature(4, "a hole"), feature(2, "a boss")])

    assert [held.id for held in revised.proposal] == [4, 2]


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


def test_a_feature_must_cite_evidence() -> None:
    """The split puts the exact numbers in the evidence, so a feature with none
    hands the stages after it a type and a size and nothing to place them by."""
    with pytest.raises(ValidationError, match="cites no evidence"):
        feature(1, "a boss", evidence=[])


def test_the_two_tables_agree_on_every_name_they_share() -> None:
    """One arity table serves the reading and the claim, which is only sound
    while a name they both use means the same shape on either side."""
    drawn = {name for row in _DRAWN_PARAMETERS.values() for name in row}
    claimed = {name for row in _CLAIMED_PARAMETERS.values() for name in row}
    assert drawn & claimed, "the tables share nothing, so they need not be unified"
    assert (drawn | claimed) <= set(_ARITY)
    for name in drawn & claimed:
        assert _ARITY[name] == 1, f"{name} is shared, so it must be a plain size"


def test_rendering_keeps_every_digit_the_stage_transcribed() -> None:
    """The coder builds from these numbers, so the rendering may shorten the
    layout but never the value. `f"{x:g}"` would take 33.0015507591 down to
    six figures and move the part by a hundredth of a millimetre."""
    reading = evidence(
        "line", start=[33.0015507591, 40.031937346], end=[133.0, 40.031937346]
    )
    rendered = render_hypothesis(
        hypothesis(proposal=[feature(1, "plate", evidence=[reading])])
    )

    for value in (33.0015507591, 40.031937346, 133.0):
        assert repr(value) in rendered, value


def test_rendering_leads_with_the_feature_and_hangs_its_readings_underneath() -> None:
    """Shape, not only size. A planner orienting itself reads the name, the
    description and the claim; the transcription it needs afterwards must not
    sit between them."""
    rendered = render_hypothesis(
        hypothesis(
            proposal=[
                feature(1, "plate", geometry=[geometry("cylinder")]),
                feature(2, "bore"),
            ]
        )
    )
    lines = rendered.splitlines()

    assert lines[0] == "sem1 plate"
    assert lines[1].strip() == "plate"
    assert lines[2].startswith("  geometry: cylinder(")
    assert lines[3].startswith("    front line visible")
    assert any(line == "sem2 bore" for line in lines)
    assert lines[-1].startswith("rationale:")


def test_a_feature_claiming_nothing_says_so_rather_than_going_quiet() -> None:
    """An empty `geometry` is a real answer -- nothing about the feature was
    determined -- and a rendering that dropped the line would read as though
    the field did not exist."""
    rendered = render_hypothesis(hypothesis(proposal=[feature(1, "plate")]))

    assert "geometry: (none stated)" in rendered


def test_rendering_costs_a_fraction_of_the_json_it_replaces() -> None:
    """Why this exists. Measured on a real run: 50,482 characters of
    `model_dump_json(indent=2)` for seven features, of which 14k was braces,
    key names and indentation. The bound is loose so that a later field cannot
    silently undo it."""
    many = hypothesis(
        proposal=[
            feature(
                index,
                "rounded base plate",
                geometry=[geometry("cylinder")],
                evidence=[evidence("spline") for _ in range(4)],
            )
            for index in range(1, 8)
        ]
    )

    assert len(render_hypothesis(many)) < len(many.model_dump_json(indent=2)) / 2


def test_rendering_survives_a_kind_that_has_no_axis() -> None:
    """`axis` is null for a kind that has none, which the contract says in as
    many words. The rendering dereferenced it anyway and took a run down at the
    operations node, an hour in -- the fixtures had always passed an axis, so
    nothing here exercised the branch the contract documents."""
    rendered = render_hypothesis(
        hypothesis(
            proposal=[feature(1, "ball end", geometry=[geometry("sphere", axis=None)])]
        )
    )

    assert "sphere(radius=5.0) exact" in rendered
    assert "axis" not in rendered.split("geometry:")[1].splitlines()[0]


def test_every_optional_field_of_the_contract_renders() -> None:
    """The same class of hole, closed once rather than field by field: build a
    feature with every nullable field null and every one filled, and render
    both."""
    bare = feature(
        1, "bare", geometry=[geometry("sphere", axis=None)], open_question=None
    )
    full = feature(
        2,
        "full",
        geometry=[geometry("cylinder", axis="z")],
        open_question="is it blind?",
    )

    rendered = render_hypothesis(hypothesis(proposal=[bare, full]))

    assert "sem1 bare" in rendered
    assert "sem2 full" in rendered
    assert "open question: is it blind?" in rendered
    assert "None" not in rendered
