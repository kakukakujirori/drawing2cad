"""The plan as a graph: what it refuses, and what the order is derived from.

The point of taking a graph instead of a list is that a dependency can be
checked and an implied sequence cannot. These tests are that check -- and the
one that matters most is the last: the same plan must always come out in the
same order, because the coder builds what comes out, not what was written.
"""

from collections.abc import Sequence

import pytest
from pydantic import BaseModel, ValidationError

from tests.zeroshot.contracts import evidence, feature, geometry, hypothesis
from zeroshot.pipeline.messages.contracts import (
    Operation,
    OperationPlan,
    OperationVerb,
    SemanticHypothesis,
    fingerprint,
    linearise,
    render_plan,
    render_plan_review,
    resolve_reference,
    review_plan,
)


def op(
    name: str,
    *,
    needs: Sequence[str] = (),
    builds: Sequence[int] = (),
    detail: str = "",
    verb: OperationVerb = OperationVerb.EXTRUDE,
) -> Operation:
    return Operation(
        name=name,
        verb=verb,
        detail=detail or f"operation {name}",
        depends_on=list(needs),
        semantics=list(builds),
    )


def plan(*operations: Operation) -> OperationPlan:
    return OperationPlan(
        proposal=list(operations),
        rationale="because",
    )


def _features(*ids: int) -> SemanticHypothesis:
    """A hypothesis that establishes exactly `ids`, and nothing else about it."""
    return hypothesis(
        proposal=[feature(identifier, f"feature {identifier}") for identifier in ids]
    )


@pytest.mark.parametrize("contract", [OperationPlan, Operation])
def test_the_schema_is_one_strict_output_mode_accepts(
    contract: type[BaseModel],
) -> None:
    """Same constraint the semantics contract is held to: every property in
    `required`, no `additionalProperties`, and no prose outside the field
    descriptions, because the schema is sent to the model."""
    schema = contract.model_json_schema()

    assert "description" not in schema
    for held in [schema, *schema.get("$defs", {}).values()]:
        if "enum" in held:
            # A closed set of strings, which strict mode takes as it is. It has
            # no properties to close and no `required` to complete.
            assert "description" not in held
            continue
        assert set(held.get("required", [])) == set(held.get("properties", {}))
        assert held.get("additionalProperties") is False


def test_a_plan_holds_at_least_one_operation() -> None:
    with pytest.raises(ValidationError, match="at least one"):
        plan()


def test_operation_names_are_unique() -> None:
    """A name is the whole of an operation's identity, so two steps sharing one
    leaves every later stage unable to say which it means."""
    with pytest.raises(ValidationError, match="both called op_base"):
        plan(op("op_base"), op("op_base"))


@pytest.mark.parametrize(
    "name",
    ["base_plate", "op_BasePlate", "op-bore", "op_bore through", "op_", "", "opbore"],
)
def test_a_name_that_does_not_mark_itself_as_a_step_is_refused(name: str) -> None:
    """`op_` is what keeps a step and the feature it builds from arriving as
    the same word, and it is a prefix rather than advice because a name is read
    back out of a comment marker by a regular expression."""
    with pytest.raises(ValidationError):
        plan(
            Operation(
                name=name,
                verb=OperationVerb.EXTRUDE,
                detail="do it",
                depends_on=[],
                semantics=[],
            )
        )


def test_a_dependency_on_an_operation_that_does_not_exist_is_refused() -> None:
    """The plan's own naming is the one thing it can be held to on its own,
    and a reference into nothing would linearise by crashing."""
    with pytest.raises(ValidationError, match=r"op_b depends on op_g"):
        plan(op("op_a"), op("op_b", needs=["op_g"]))


def test_an_operation_cannot_wait_on_itself() -> None:
    with pytest.raises(ValidationError, match="op_a depends on itself"):
        plan(op("op_a", needs=["op_a"]))


def test_a_cycle_is_refused_and_named() -> None:
    """`middleware/model_retry.py` hands this message back for a free retry, so
    it has to say which dependency to drop rather than that one exists."""
    # `->` reads as "waits on", so the chain runs the way the dependencies do.
    with pytest.raises(ValidationError, match=r"op_a -> op_b -> op_c -> op_a"):
        plan(
            op("op_a", needs=["op_b"]),
            op("op_b", needs=["op_c"]),
            op("op_c", needs=["op_a"]),
        )


def test_the_build_order_follows_the_dependencies_not_the_writing_order() -> None:
    """The whole reason for the graph. Written 4, 1, 3, 2 and built 1, 2, 3, 4."""
    written = plan(
        op("op_d", needs=["op_b", "op_c"]),
        op("op_a"),
        op("op_c", needs=["op_a"]),
        op("op_b", needs=["op_a"]),
    )

    assert [held.name for held in linearise(written)] == [
        "op_a",
        "op_b",
        "op_c",
        "op_d",
    ]


def test_an_operation_lands_beside_the_ones_it_consumes() -> None:
    """Depth-first, not breadth-first. Two independent chains come out whole
    rather than interleaved band by band, so a fillet reads next to the edge it
    rounds instead of in a later heap of details."""
    two_chains = plan(
        op("op_a"),
        op("op_b", needs=["op_a"]),
        op("op_c", needs=["op_b"]),
        op("op_d"),
        op("op_e", needs=["op_d"]),
    )

    assert [held.name for held in linearise(two_chains)] == [
        "op_a",
        "op_b",
        "op_c",
        "op_d",
        "op_e",
    ]


def test_the_same_plan_always_linearises_the_same_way() -> None:
    """The coder builds the derived order, so an order that moved between two
    runs of the same plan would be a difference nobody wrote."""
    written = plan(
        op("op_c", needs=["op_a"]),
        op("op_b", needs=["op_a"]),
        op("op_a"),
        op("op_d", needs=["op_a"]),
    )

    assert (
        len({tuple(held.name for held in linearise(written)) for _ in range(20)}) == 1
    )


def test_every_operation_is_placed_exactly_once() -> None:
    """A diamond reaches its root twice; the coder must build it once."""
    diamond = plan(
        op("op_a"),
        op("op_b", needs=["op_a"]),
        op("op_c", needs=["op_a"]),
        op("op_d", needs=["op_b", "op_c"]),
    )

    order = [held.name for held in linearise(diamond)]

    assert sorted(order) == ["op_a", "op_b", "op_c", "op_d"]
    assert len(order) == len(set(order))


def test_a_feature_no_operation_builds_is_reported() -> None:
    """The check the plan's validator cannot make: a feature id points into an
    answer the planner produced separately, so only something holding both can
    tell that sem2 was established and then dropped."""
    review = review_plan(
        plan(op("op_a", builds=[1]), op("op_b", builds=[3])), _features(1, 2, 3)
    )

    assert review.uncovered == [2]
    assert review.unknown == []
    assert not review.sound


def test_a_feature_the_plan_invents_is_reported() -> None:
    review = review_plan(plan(op("op_a", builds=[9])), _features(1))

    assert review.unknown == [9]
    assert review.uncovered == [1]


def test_a_plan_that_accounts_for_every_feature_is_complete() -> None:
    review = review_plan(
        plan(op("op_a", builds=[1, 2]), op("op_b", builds=[2])), _features(1, 2)
    )

    assert review.sound


def test_the_rendering_shows_the_derived_order_and_what_each_step_needs() -> None:
    """What the coder is handed. It is the order, not the graph: following it
    is the job, and a plan whose dependencies were wrong reads as wrong here."""
    rendered = render_plan(
        plan(op("op_b", needs=["op_a"], builds=[4]), op("op_a", builds=[1])),
        _features(1, 4),
    )
    lines = rendered.splitlines()

    assert "step 1  op_a extrude (needs nothing; builds sem1)" in lines[1]
    assert "step 2  op_b extrude (after op_a; builds sem4)" in lines[2]


def test_the_step_number_is_derived_and_not_part_of_the_plan() -> None:
    """The position is worked out from the dependencies, so it is put on at
    the point of reading. Holding it in the plan would be a second thing for
    the planner to keep in step with the first."""
    written = plan(op("op_b", needs=["op_a"], builds=[1]), op("op_a", builds=[1]))

    assert "step" not in written.model_dump_json()
    assert [
        line.split()[1]
        for line in render_plan(written, _features(1)).splitlines()[1:-1]
    ] == [
        "1",
        "2",
    ]


def test_an_operation_that_names_no_feature_says_so() -> None:
    """Rendering it as an empty list would read as a formatting slip; saying it
    plainly keeps the gap visible to whoever reads the plan."""
    assert "builds nothing named" in render_plan(plan(op("op_a")), _features(1))


def test_the_gap_is_reported_by_naming_the_features_it_left_out() -> None:
    """What the planner is sent back with. Named rather than counted: "sem2,
    sem5" is something to go and plan, where "two are missing" sends the stage
    back to compare two lists it has already been given."""
    review = review_plan(plan(op("op_a", builds=[1])), _features(1, 2, 5))

    told = render_plan_review(review)

    assert "sem2, sem5" in told
    assert "sem1" not in told


def test_a_dangling_reference_is_reported_separately_from_a_gap() -> None:
    """They are different mistakes and take different corrections: one is a
    feature to go and build, the other a number to stop citing."""
    review = review_plan(plan(op("op_a", builds=[9])), _features(1))

    told = render_plan_review(review)

    assert "sem1" in told
    assert "sem9" in told
    assert told.index("sem1") < told.index("sem9")


def test_a_complete_plan_is_reported_as_nothing_at_all() -> None:
    """An empty string, not a sentence saying everything is fine: this only
    ever renders on the way back to a stage that has something to fix."""
    assert (
        render_plan_review(review_plan(plan(op("op_a", builds=[1])), _features(1)))
        == ""
    )


def test_a_fingerprint_follows_the_content_and_nothing_else() -> None:
    """Two plans that say the same thing are the same plan as far as anything
    downstream is concerned, and one that differs anywhere is not."""
    one = plan(op("op_a", builds=[1]))
    same = plan(op("op_a", builds=[1]))
    other = plan(op("op_a", builds=[2]))

    assert fingerprint(one) == fingerprint(same)
    assert fingerprint(one) != fingerprint(other)


def test_a_reading_knows_which_pair_it_was_taken_from() -> None:
    """The whole point of carrying the fingerprints. A reading kept in state
    outlives the work it measured, and this is what lets the two be replaced
    without anyone having to remember to throw the reading away."""
    established = _features(1, 2)
    measured = plan(op("op_a", builds=[1]), op("op_b", builds=[2]))
    review = review_plan(measured, established)

    assert review.describes(established, measured)


@pytest.mark.parametrize("changed", ["hypothesis", "plan"])
def test_a_reading_disowns_a_pair_that_has_moved_on(changed: str) -> None:
    established = _features(1, 2)
    measured = plan(op("op_a", builds=[1]), op("op_b", builds=[2]))
    review = review_plan(measured, established)

    if changed == "hypothesis":
        assert not review.describes(_features(1, 2, 3), measured)
    else:
        assert not review.describes(established, plan(op("op_a", builds=[1])))


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
        7,
        "shoulder blend",
        geometry=[
            geometry(
                "torus",
                major_radius=11.31245992416,
                tube_radius=3.39440063713,
            ),
            geometry("plane", axis="x"),
        ],
    )


def test_a_reference_is_filled_in_before_the_coder_reads_it() -> None:
    """The whole of the arrangement. A planner made to retype a spline's
    control points will eventually mistype one, and has: a reference resolved
    on the way out never passes through a model's output, so that cannot
    happen rather than being caught after it has."""
    resolved = resolve_reference("Sweep a blend of sem7.major_radius", _torus_feature())

    assert resolved == "Sweep a blend of sem7.major_radius (11.31245992416)"


def test_a_reference_says_which_geometry_when_a_feature_claims_several() -> None:
    """Qualified by kind rather than by position: an index into `geometry`
    would be a second numbering to keep in step with the hypothesis."""
    held = _torus_feature()

    assert "(3.39440063713)" in resolve_reference("sem7.torus.tube_radius", held)
    assert "(11.31245992416)" in resolve_reference("sem7.torus.major_radius", held)


def test_a_reference_to_something_the_hypothesis_lacks_is_left_alone() -> None:
    """Rendering is not the place to fail. The plan review has already refused
    the plan for this; what the reader gets meanwhile is the text as written."""
    assert resolve_reference("sem7.height", _torus_feature()) == "sem7.height"
    assert resolve_reference("sem9.radius", _torus_feature()) == "sem9.radius"


def test_a_list_of_numbers_stays_a_reference() -> None:
    """A control-point list expanded inline is unreadable, and the coder holds
    the hypothesis anyway, so the address is more use than the contents."""
    held = hypothesis(proposal=[feature(1, "blend")])

    assert resolve_reference("follow sem1.control_points", held) == (
        "follow sem1.control_points"
    )


def test_a_number_the_hypothesis_already_holds_is_refused() -> None:
    """What forces the reference. The planner is not asked politely to cite;
    a plan that copies is sent back."""
    review = review_plan(
        plan(op("op_a", builds=[7], detail="Sweep a blend of radius 11.31245992416")),
        _torus_feature(),
    )

    assert review.transcribed == {"op_a": ["11.31245992416"]}
    assert not review.sound
    assert "sem<id>.<parameter>" in render_plan_review(review)


def test_a_number_the_planner_worked_out_itself_is_left_alone() -> None:
    """Half a width, a clearance, a chosen depth. These are the planner's own
    and have nowhere else to live, which is what makes refusing on the others
    safe."""
    review = review_plan(
        plan(
            op(
                "op_a",
                builds=[7],
                detail="Cut 5.65622996208 deep, half of sem7.major_radius",
            )
        ),
        _torus_feature(),
    )

    assert review.transcribed == {}


def test_a_short_number_is_the_planners_own_words() -> None:
    """`25 mm` is a sentence, not a transcription, even where the hypothesis
    happens to hold 25."""
    held = hypothesis(
        proposal=[feature(7, "boss", geometry=[geometry("sphere", radius=25.0)])]
    )

    assert (
        review_plan(
            plan(op("op_a", builds=[7], detail="Extrude 25 mm")), held
        ).transcribed
        == {}
    )


def test_a_reference_that_stands_for_nothing_is_refused() -> None:
    review = review_plan(
        plan(op("op_a", builds=[7], detail="Sweep sem7.height along +z")),
        _torus_feature(),
    )

    assert review.unresolved == {"op_a": ["sem7.height"]}
    assert "sem7.height" in render_plan_review(review)


def test_a_plan_full_of_copied_numbers_is_still_told_in_a_sentence() -> None:
    """A first plan can hold well over a hundred of them. Naming every one
    would bury the instruction under the evidence for it."""
    radii = [11.31245992416 + n for n in range(40)]
    copied = " ".join(f"{radius:.11f}" for radius in radii)
    held = hypothesis(
        proposal=[
            feature(
                7,
                "blend",
                geometry=[geometry("sphere", radius=radius) for radius in radii],
            )
        ]
    )

    told = render_plan_review(
        review_plan(plan(op("op_a", builds=[7], detail=copied)), held)
    )

    assert "and 37 more" in told
    assert len(told) < 400


def test_a_claim_answers_before_the_readings_it_was_claimed_from() -> None:
    """A bore states one radius in `geometry` and may carry half a dozen arcs
    in `evidence`. `sem4.radius` means the one the feature claims."""
    bore = feature(
        4,
        "main bore",
        geometry=[geometry("cylinder", radius=3.40755883124, height=16.5366825634)],
        evidence=[evidence("arc", radius=99.9), evidence("arc", radius=88.8)],
    )

    assert "(3.40755883124)" in resolve_reference(
        "sem4.radius", hypothesis(proposal=[bore])
    )
