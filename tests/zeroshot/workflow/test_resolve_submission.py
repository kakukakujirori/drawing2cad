"""Resolving the addresses a stage's prose points at.

An address is the one place a number is written down, so what these check is
that the value arriving beside it is the value the round states -- and that it
arrives once however often a text passes through here.
"""

from collections.abc import Sequence

from tests.zeroshot.contracts import evidence, feature, geometry, hypothesis, sheet
from zeroshot.pipeline.messages.contracts import (
    Dimension,
    DimensionKind,
    DrawingEvidence,
    DrawingSource,
    Operation,
    OperationPlan,
    OperationVerb,
    SemanticHypothesis,
)
from zeroshot.pipeline.workflow.resolve_submission import (
    _references_resolved_in_prose,
    resolve_references,
    unresolved_references,
    without_resolved_values,
)


def read(
    *entries: DrawingEvidence,
    figures: Sequence[Dimension] = (),
) -> DrawingSource:
    """A one-sheet drawing holding exactly these entries and printed figures."""
    return DrawingSource(
        sheets=[sheet("front", evidence=list(entries), dimensions=list(figures))]
    )


def resolved(text: str, held: SemanticHypothesis, drawn: DrawingSource = None) -> str:  # type: ignore[assignment]
    return _references_resolved_in_prose(text, held, drawn or read())


def op(name: str, *, detail: str) -> Operation:
    return Operation(
        name=name,
        verb=OperationVerb.EXTRUDE,
        detail=detail,
        depends_on=[],
        semantics=[],
    )


def plan(*operations: Operation) -> OperationPlan:
    return OperationPlan(proposal=list(operations), rationale="because")


def _torus_feature() -> SemanticHypothesis:
    """One feature whose size is stated, so a reference to it has an answer.

    Full precision, as a reading off a DXF is. That is also what makes a copy
    of it recognisable: nobody arrives at fourteen decimal places by thinking.
    """
    return hypothesis(
        proposal=[
            feature(
                "sem_shoulder_blend",
                "shoulder blend",
                geometry=[
                    geometry(
                        "torus",
                        name="geo_blend_torus",
                        major_radius=11.31245992416,
                        tube_radius=3.39440063713,
                    ),
                    geometry("plane", axis="x", name="geo_side_plane"),
                ],
            )
        ]
    )


def _ball(radius: float = 4.25) -> SemanticHypothesis:
    return hypothesis(
        proposal=[
            feature(
                "sem_main_bore",
                "main bore",
                geometry=[geometry("sphere", name="geo_ball", radius=radius)],
            )
        ]
    )


def test_a_reference_is_filled_in_before_the_coder_reads_it() -> None:
    """The whole of the arrangement. A planner made to retype a spline's
    control points will eventually mistype one, and has: a reference resolved
    on the way out never passes through a model's output, so that cannot
    happen rather than being caught after it has."""
    written = resolved(
        "Sweep a blend of sem_shoulder_blend.geo_blend_torus.major_radius",
        _torus_feature(),
    )

    assert written.endswith(
        "sem_shoulder_blend.geo_blend_torus.major_radius (= 11.31245992416)"
    )


def test_a_reference_says_which_geometry_when_a_feature_claims_several() -> None:
    held = _torus_feature()

    assert "(= 3.39440063713)" in resolved(
        "sem_shoulder_blend.geo_blend_torus.tube_radius", held
    )
    assert (
        resolved("sem_shoulder_blend.geo_side_plane.radius", held)
        == "sem_shoulder_blend.geo_side_plane.radius"
    )


def test_a_canonical_reference_names_one_geometry_even_when_kinds_repeat() -> None:
    held = hypothesis(
        proposal=[
            feature(
                "sem_two_spherical_ends",
                "two spherical ends",
                geometry=[
                    geometry("sphere", name="geo_left_end", radius=4.25),
                    geometry("sphere", name="geo_right_end", radius=9.75),
                ],
            )
        ]
    )

    assert resolved("sem_two_spherical_ends.geo_right_end.radius", held).endswith(
        "geo_right_end.radius (= 9.75)"
    )


def test_a_drawing_entry_is_addressed_without_the_feature_that_cites_it() -> None:
    """An `ev_` name is unique across the drawing, so it needs no scope -- and
    two features resting on the same entry cite it by the very same address."""
    drawn = read(
        evidence("circle", name="ev_front_circle", center=[1.0, 2.0], radius=3.0),
        evidence("circle", name="ev_right_circle", center=[12.0, 13.0], radius=5.0),
    )
    held = hypothesis(
        proposal=[feature("sem_main_bore", "main bore", evidence=["ev_front_circle"])]
    )

    assert resolved("at ev_right_circle.center", held, drawn) == (
        "at ev_right_circle.center (= [12.0 13.0])"
    )
    assert resolved("radius ev_front_circle.radius", held, drawn).endswith(
        "ev_front_circle.radius (= 3.0)"
    )


def test_a_printed_figure_is_addressed_the_same_way() -> None:
    """The printed value is the authoritative one, so it is the one number a
    plan should never have to retype."""
    drawn = read(
        evidence("circle", name="ev_front_circle", radius=3.0),
        figures=[
            Dimension(
                name="dim_bore_diameter",
                kind=DimensionKind.DIAMETER,
                text="⌀6 THRU",
                nominal=6.0,
                quantity=2,
                note=None,
            )
        ],
    )
    held = _ball()

    assert resolved("bore dim_bore_diameter.nominal", held, drawn) == (
        "bore dim_bore_diameter.nominal (= 6.0)"
    )
    assert resolved("dim_bore_diameter.quantity", held, drawn) == (
        "dim_bore_diameter.quantity (= 2.0)"
    )


def test_a_bare_drawing_name_is_a_mention_rather_than_an_address() -> None:
    """The drawing stores those names in its own answer, so expanding one
    wherever it appears would rewrite the drawing into itself."""
    drawn = read(evidence("circle", name="ev_front_circle", radius=3.0))

    assert resolved("the circle ev_front_circle", _ball(), drawn) == (
        "the circle ev_front_circle"
    )


def test_a_reference_can_name_one_coordinate_of_a_point() -> None:
    """An extent runs between coordinates: a slot's height is one reading's y
    less another's, and neither whole point states it."""
    drawn = read(
        evidence("circle", name="ev_front_circle", center=[1.5, 2.5], radius=3.0)
    )

    assert resolved("at ev_front_circle.center.x", _ball(), drawn) == (
        "at ev_front_circle.center.x (= 1.5)"
    )
    assert resolved("at ev_front_circle.center.y", _ball(), drawn) == (
        "at ev_front_circle.center.y (= 2.5)"
    )


def test_a_coordinate_asked_of_a_single_number_is_left_alone() -> None:
    """A radius is one number, so `.x` names nothing in it."""
    drawn = read(
        evidence("circle", name="ev_front_circle", center=[1.5, 2.5], radius=3.0)
    )

    assert resolved("ev_front_circle.radius.x", _ball(), drawn) == (
        "ev_front_circle.radius.x"
    )


def test_a_claim_stopping_at_its_name_resolves_to_everything_it_states() -> None:
    held = hypothesis(
        proposal=[
            feature(
                "sem_main_bore",
                "main bore",
                geometry=[
                    geometry("cylinder", name="geo_cylinder", radius=3.0, height=16.5)
                ],
            )
        ]
    )

    assert resolved("bounded by sem_main_bore.geo_cylinder", held) == (
        "bounded by sem_main_bore.geo_cylinder (= radius=3.0 height=16.5)"
    )


def test_a_member_that_states_no_size_is_left_as_the_name_it_is() -> None:
    """A plane has no size, so there is nothing to put beside it."""
    held = hypothesis(
        proposal=[
            feature("sem_base", "base", geometry=[geometry("plane", name="geo_top")])
        ]
    )

    assert resolved("flat on sem_base.geo_top", held) == "flat on sem_base.geo_top"


def test_resolving_twice_annotates_once() -> None:
    """The snapshot a model reads back is already resolved, so the text it
    copies forward arrives here annotated."""
    held = _ball()
    once = resolved("a ball of sem_main_bore.geo_ball.radius", held)

    assert resolved(once, held) == once
    assert once.count("4.25") == 1


def test_an_annotation_that_no_longer_holds_is_replaced() -> None:
    """A revised hypothesis moves the number the last round wrote down."""
    assert resolved("sem_main_bore.geo_ball.radius (= 9.75)", _ball()) == (
        "sem_main_bore.geo_ball.radius (= 4.25)"
    )


def test_a_drawing_revision_refreshes_an_old_annotation() -> None:
    old = read(evidence("circle", name="ev_front_circle", radius=3.0))
    revised = read(evidence("circle", name="ev_front_circle", radius=4.5))
    annotated = resolved("ev_front_circle.radius", _ball(), old)

    assert resolved(annotated, _ball(), revised) == ("ev_front_circle.radius (= 4.5)")


def test_deleting_drawing_evidence_removes_its_old_annotation() -> None:
    old = read(evidence("circle", name="ev_front_circle", radius=3.0))
    annotated = resolved("ev_front_circle.radius", _ball(), old)

    assert resolved(annotated, _ball(), read()) == "ev_front_circle.radius"
    assert unresolved_references(annotated, _ball(), read()) == [
        "ev_front_circle.radius"
    ]


def test_a_parenthesis_of_the_models_own_is_left_alone() -> None:
    """`= ` is what tells this file's annotation from the planner's aside."""
    assert resolved(
        "sem_main_bore.geo_ball.radius (the seat, not the bore)", _ball()
    ) == ("sem_main_bore.geo_ball.radius (= 4.25) (the seat, not the bore)")


def test_a_resolved_value_can_be_taken_back_out() -> None:
    assert (
        without_resolved_values(
            "a ball of sem_main_bore.geo_ball.radius (= 4.25) at (0, 0)"
        )
        == "a ball of sem_main_bore.geo_ball.radius at (0, 0)"
    )


def test_every_string_in_an_answer_is_resolved() -> None:
    """A reference is worth as much in a ticket summary as in an operation."""
    written = plan(op("op_bore", detail="cut sem_main_bore.geo_ball.radius deep"))

    answered = resolve_references(written, _ball(), read())

    assert answered.proposal[0].detail == (
        "cut sem_main_bore.geo_ball.radius (= 4.25) deep"
    )
    assert answered.proposal[0].verb is OperationVerb.EXTRUDE


def test_drawing_addresses_resolve_before_a_round_has_a_hypothesis() -> None:
    """The drawing stage answers its tickets before semantics has run."""
    drawn = read(evidence("circle", name="ev_front_circle", radius=3.0))

    assert _references_resolved_in_prose("ev_front_circle.radius", None, drawn) == (
        "ev_front_circle.radius (= 3.0)"
    )


def test_a_reference_to_something_the_round_lacks_is_left_alone() -> None:
    """Rendering is not the place to fail; contextual validation owns that."""
    assert resolved("sem_shoulder_blend.geo_blend_torus.height", _torus_feature()) == (
        "sem_shoulder_blend.geo_blend_torus.height"
    )
    assert resolved("sem_missing.geo_sphere.radius", _torus_feature()) == (
        "sem_missing.geo_sphere.radius"
    )


def test_a_list_parameter_is_resolved_as_one_exactly_addressed_value() -> None:
    drawn = read(evidence("spline", name="ev_front_spline"))

    assert "(= [0.0 0.0 1.0 1.0 2.0 0.0])" in resolved(
        "follow ev_front_spline.control_points", _ball(), drawn
    )


def test_claim_and_entry_parameters_are_separate_named_addresses() -> None:
    held = hypothesis(
        proposal=[
            feature(
                "sem_main_bore",
                "main bore",
                geometry=[
                    geometry(
                        "cylinder",
                        name="geo_cylinder",
                        radius=3.40755883124,
                        height=16.5366825634,
                    )
                ],
                evidence=["ev_front_circle"],
            )
        ]
    )
    drawn = read(
        evidence("circle", name="ev_front_circle", radius=99.9),
        evidence("circle", name="ev_right_circle", radius=88.8),
    )

    assert "(= 3.40755883124)" in resolved(
        "sem_main_bore.geo_cylinder.radius", held, drawn
    )
    assert "(= 99.9)" in resolved("ev_front_circle.radius", held, drawn)


def test_what_resolves_and_what_the_validator_refuses_are_one_answer() -> None:
    """Two definitions of a good address drifted apart once already: the
    resolver skipped one ending a sentence that validation had accepted."""
    held = hypothesis(
        proposal=[
            feature(
                "sem_main_bore",
                "main bore",
                geometry=[geometry("sphere", name="geo_ball", radius=4.25)],
                evidence=["ev_edge"],
            )
        ]
    )
    drawn = read(
        evidence("line", name="ev_edge", start=[1.0, 2.0], end=[3.0, 4.0]),
        figures=[
            Dimension(
                name="dim_width",
                kind=DimensionKind.LINEAR,
                text="100",
                nominal=100.0,
                quantity=1,
                note=None,
            )
        ],
    )
    sound = [
        "sem_main_bore.geo_ball.radius",
        "ev_edge.start",
        "ev_edge.start.x",
        "dim_width.nominal",
    ]

    for address in sound:
        assert unresolved_references(f"cut at {address}.", held, drawn) == []
        assert resolved(f"cut at {address}.", held, drawn) != f"cut at {address}."

    assert unresolved_references(
        "cut at ev_edge.middle and sem_absent.geo_ball.radius.", held, drawn
    ) == ["ev_edge.middle", "sem_absent.geo_ball.radius"]


def test_the_old_feature_scoped_entry_address_is_refused() -> None:
    """An entry belongs to the drawing now, so a feature cannot scope one."""
    held = hypothesis(
        proposal=[feature("sem_main_bore", "main bore", evidence=["ev_edge"])]
    )
    drawn = read(evidence("line", name="ev_edge", start=[1.0, 2.0], end=[3.0, 4.0]))

    assert unresolved_references(
        "cut at sem_main_bore.ev_edge.start.", held, drawn
    ) == ["sem_main_bore.ev_edge.start"]


def test_a_member_stating_no_size_is_sound_but_has_nothing_to_write() -> None:
    """Naming a plane is a fair way to say which face, so it is not refused."""
    held = hypothesis(
        proposal=[
            feature("sem_base", "base", geometry=[geometry("plane", name="geo_top")])
        ]
    )

    assert unresolved_references("flat on sem_base.geo_top.", held, read()) == []
    assert resolved("flat on sem_base.geo_top.", held) == "flat on sem_base.geo_top."
