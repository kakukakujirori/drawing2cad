"""Resolving the addresses a stage's prose points at.

An address is the one place a number is written down, so what these check is
that the value arriving beside it is the value the hypothesis states -- and
that it arrives once however often a text passes through here.
"""

from tests.zeroshot.contracts import evidence, feature, geometry, hypothesis
from zeroshot.pipeline.messages.contracts import (
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
            geometry_feature(),
        ]
    )


def geometry_feature():
    return feature(
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


def test_a_reference_is_filled_in_before_the_coder_reads_it() -> None:
    """The whole of the arrangement. A planner made to retype a spline's
    control points will eventually mistype one, and has: a reference resolved
    on the way out never passes through a model's output, so that cannot
    happen rather than being caught after it has."""
    resolved = _references_resolved_in_prose(
        "Sweep a blend of sem_shoulder_blend.geo_blend_torus.major_radius",
        _torus_feature(),
    )

    assert resolved.endswith(
        "sem_shoulder_blend.geo_blend_torus.major_radius (= 11.31245992416)"
    )


def test_a_reference_says_which_geometry_when_a_feature_claims_several() -> None:
    held = _torus_feature()

    assert "(= 3.39440063713)" in _references_resolved_in_prose(
        "sem_shoulder_blend.geo_blend_torus.tube_radius", held
    )
    assert (
        _references_resolved_in_prose("sem_shoulder_blend.geo_side_plane.radius", held)
        == "sem_shoulder_blend.geo_side_plane.radius"
    )


def test_a_canonical_reference_names_one_geometry_even_when_kinds_repeat() -> None:
    repeated = feature(
        "sem_two_spherical_ends",
        "two spherical ends",
        geometry=[
            geometry("sphere", name="geo_left_end", radius=4.25),
            geometry("sphere", name="geo_right_end", radius=9.75),
        ],
    )
    held = hypothesis(proposal=[repeated])

    assert _references_resolved_in_prose(
        "sem_two_spherical_ends.geo_right_end.radius", held
    ).endswith("geo_right_end.radius (= 9.75)")


def test_a_canonical_reference_names_one_evidence_reading_and_its_vector() -> None:
    bore = feature(
        "sem_main_bore",
        "main bore",
        evidence=[
            evidence(
                "circle",
                name="ev_front_circle",
                center=[1.0, 2.0],
                radius=3.0,
            ),
            evidence(
                "circle",
                name="ev_right_circle",
                center=[12.0, 13.0],
                radius=5.0,
            ),
        ],
    )
    held = hypothesis(proposal=[bore])

    assert _references_resolved_in_prose(
        "at sem_main_bore.ev_right_circle.center", held
    ) == ("at sem_main_bore.ev_right_circle.center (= [12.0 13.0])")
    assert _references_resolved_in_prose(
        "radius sem_main_bore.ev_front_circle.radius", held
    ).endswith("sem_main_bore.ev_front_circle.radius (= 3.0)")


def test_a_reference_can_name_one_coordinate_of_a_point() -> None:
    """An extent runs between coordinates: a slot's height is one reading's y
    less another's, and neither whole point states it."""
    bore = feature(
        "sem_main_bore",
        "main bore",
        evidence=[
            evidence("circle", name="ev_front_circle", center=[1.5, 2.5], radius=3.0)
        ],
    )
    held = hypothesis(proposal=[bore])

    assert _references_resolved_in_prose(
        "at sem_main_bore.ev_front_circle.center.x", held
    ) == ("at sem_main_bore.ev_front_circle.center.x (= 1.5)")
    assert _references_resolved_in_prose(
        "at sem_main_bore.ev_front_circle.center.y", held
    ) == ("at sem_main_bore.ev_front_circle.center.y (= 2.5)")


def test_a_coordinate_asked_of_a_single_number_is_left_alone() -> None:
    """A radius is one number, so `.x` names nothing in it."""
    bore = feature(
        "sem_main_bore",
        "main bore",
        evidence=[
            evidence("circle", name="ev_front_circle", center=[1.5, 2.5], radius=3.0)
        ],
    )
    held = hypothesis(proposal=[bore])

    assert (
        _references_resolved_in_prose("sem_main_bore.ev_front_circle.radius.x", held)
        == "sem_main_bore.ev_front_circle.radius.x"
    )


def test_an_address_stopping_at_the_member_resolves_to_the_whole_reading() -> None:
    """A profile is named by the readings that bound it, and an extent is a
    distance between a reading's numbers rather than any one of them."""
    bore = feature(
        "sem_main_bore",
        "main bore",
        evidence=[
            evidence(
                "line", name="ev_top_edge", start=[33.0, 146.4], end=[133.0, 146.4]
            )
        ],
    )
    held = hypothesis(proposal=[bore])

    assert _references_resolved_in_prose(
        "bounded by sem_main_bore.ev_top_edge", held
    ) == (
        "bounded by sem_main_bore.ev_top_edge (= start=[33.0 146.4] end=[133.0 146.4])"
    )


def test_a_member_that_states_no_size_is_left_as_the_name_it_is() -> None:
    """A plane has no size, so there is nothing to put beside it."""
    held = hypothesis(
        proposal=[
            feature(
                "sem_base",
                "base",
                geometry=[geometry("plane", name="geo_top_face")],
            )
        ]
    )

    assert (
        _references_resolved_in_prose("flat on sem_base.geo_top_face", held)
        == "flat on sem_base.geo_top_face"
    )


def test_resolving_twice_annotates_once() -> None:
    """The snapshot a model reads back is already resolved, so the text it
    copies forward arrives here annotated."""
    held = hypothesis(
        proposal=[
            feature(
                "sem_main_bore",
                "main bore",
                geometry=[geometry("sphere", name="geo_ball", radius=4.25)],
            )
        ]
    )
    once = _references_resolved_in_prose(
        "a ball of sem_main_bore.geo_ball.radius", held
    )

    assert _references_resolved_in_prose(once, held) == once
    assert once.count("4.25") == 1


def test_an_annotation_that_no_longer_holds_is_replaced() -> None:
    """A revised hypothesis moves the number the last round wrote down."""
    held = hypothesis(
        proposal=[
            feature(
                "sem_main_bore",
                "main bore",
                geometry=[geometry("sphere", name="geo_ball", radius=4.25)],
            )
        ]
    )

    assert _references_resolved_in_prose(
        "sem_main_bore.geo_ball.radius (= 9.75)", held
    ) == ("sem_main_bore.geo_ball.radius (= 4.25)")


def test_a_parenthesis_of_the_models_own_is_left_alone() -> None:
    """`= ` is what tells this file's annotation from the planner's aside."""
    held = hypothesis(
        proposal=[
            feature(
                "sem_main_bore",
                "main bore",
                geometry=[geometry("sphere", name="geo_ball", radius=4.25)],
            )
        ]
    )

    assert (
        _references_resolved_in_prose(
            "sem_main_bore.geo_ball.radius (the seat, not the bore)", held
        )
        == "sem_main_bore.geo_ball.radius (= 4.25) (the seat, not the bore)"
    )


def test_a_resolved_value_can_be_taken_back_out() -> None:
    assert (
        without_resolved_values(
            "a ball of sem_main_bore.geo_ball.radius (= 4.25) at (0, 0)"
        )
        == "a ball of sem_main_bore.geo_ball.radius at (0, 0)"
    )


def test_every_string_in_an_answer_is_resolved() -> None:
    """A reference is worth as much in a ticket summary as in an operation."""
    held = hypothesis(
        proposal=[
            feature(
                "sem_main_bore",
                "main bore",
                geometry=[geometry("sphere", name="geo_ball", radius=4.25)],
            )
        ]
    )
    written = plan(op("op_bore", detail="cut sem_main_bore.geo_ball.radius deep"))

    resolved = resolve_references(written, held)

    assert resolved.proposal[0].detail == (
        "cut sem_main_bore.geo_ball.radius (= 4.25) deep"
    )
    assert resolved.proposal[0].verb is OperationVerb.EXTRUDE


def test_a_reference_to_something_the_hypothesis_lacks_is_left_alone() -> None:
    """Rendering is not the place to fail; contextual validation owns that."""
    assert (
        _references_resolved_in_prose(
            "sem_shoulder_blend.geo_blend_torus.height", _torus_feature()
        )
        == "sem_shoulder_blend.geo_blend_torus.height"
    )
    assert (
        _references_resolved_in_prose("sem_missing.geo_sphere.radius", _torus_feature())
        == "sem_missing.geo_sphere.radius"
    )


def test_a_list_parameter_is_resolved_as_one_exactly_addressed_value() -> None:
    held = hypothesis(
        proposal=[
            feature(
                "sem_blend_profile",
                "blend",
                evidence=[evidence("spline", name="ev_front_spline")],
            )
        ]
    )

    assert "(= [0.0 0.0 1.0 1.0 2.0 0.0])" in _references_resolved_in_prose(
        "follow sem_blend_profile.ev_front_spline.control_points", held
    )


def test_claim_and_evidence_parameters_are_separate_named_addresses() -> None:
    bore = feature(
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
        evidence=[
            evidence("arc", name="ev_front_arc", radius=99.9),
            evidence("arc", name="ev_right_arc", radius=88.8),
        ],
    )
    held = hypothesis(proposal=[bore])

    assert "(= 3.40755883124)" in _references_resolved_in_prose(
        "sem_main_bore.geo_cylinder.radius", held
    )
    assert "(= 99.9)" in _references_resolved_in_prose(
        "sem_main_bore.ev_front_arc.radius", held
    )


def test_what_resolves_and_what_the_validator_refuses_are_one_answer() -> None:
    """Two definitions of a good address drifted apart once already: the
    resolver skipped one ending a sentence that validation had accepted."""
    held = hypothesis(
        proposal=[
            feature(
                "sem_main_bore",
                "main bore",
                geometry=[geometry("sphere", name="geo_ball", radius=4.25)],
                evidence=[
                    evidence("line", name="ev_edge", start=[1.0, 2.0], end=[3.0, 4.0])
                ],
            )
        ]
    )
    sound = [
        "sem_main_bore.geo_ball.radius",
        "sem_main_bore.ev_edge",
        "sem_main_bore.ev_edge.start",
        "sem_main_bore.ev_edge.start.x",
    ]

    for address in sound:
        assert unresolved_references(f"cut at {address}.", held) == []
        assert (
            _references_resolved_in_prose(f"cut at {address}.", held)
            != f"cut at {address}."
        )

    assert unresolved_references(
        "cut at sem_main_bore.ev_edge.middle and sem_absent.geo_ball.radius.", held
    ) == ["sem_main_bore.ev_edge.middle", "sem_absent.geo_ball.radius"]


def test_a_member_stating_no_size_is_sound_but_has_nothing_to_write() -> None:
    """Naming a plane is a fair way to say which face, so it is not refused."""
    held = hypothesis(
        proposal=[
            feature(
                "sem_base", "base", geometry=[geometry("plane", name="geo_top_face")]
            )
        ]
    )

    assert unresolved_references("flat on sem_base.geo_top_face.", held) == []
    assert _references_resolved_in_prose("flat on sem_base.geo_top_face.", held) == (
        "flat on sem_base.geo_top_face."
    )
