"""The plan as a graph: what it refuses, and what the order is derived from.

The point of taking a graph instead of a list is that a dependency can be
checked and an implied sequence cannot. These tests are that check -- and the
one that matters most is the last: the same plan must always come out in the
same order, because the coder builds what comes out, not what was written.
"""

from collections.abc import Sequence

import pytest
from pydantic import BaseModel, ValidationError

from tests.zeroshot.contracts import feature, hypothesis
from zeroshot.pipeline.messages.contracts import (
    Operation,
    OperationPlan,
    SemanticHypothesis,
    fingerprint,
    linearise,
    plan_coverage,
    render_plan,
    render_plan_coverage,
)


def op(
    identifier: int,
    *,
    needs: Sequence[int] = (),
    builds: Sequence[int] = (),
) -> Operation:
    return Operation(
        id=identifier,
        operation=f"operation {identifier}",
        depends_on=list(needs),
        semantics=list(builds),
    )


def plan(*operations: Operation) -> OperationPlan:
    return OperationPlan(proposal=list(operations), rationale="because")


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
        assert set(held.get("required", [])) == set(held.get("properties", {}))
        assert held.get("additionalProperties") is False


def test_a_plan_holds_at_least_one_operation() -> None:
    with pytest.raises(ValidationError, match="at least one"):
        plan()


def test_operation_ids_are_unique() -> None:
    with pytest.raises(ValidationError, match="unique"):
        plan(op(1), op(1))


def test_a_dependency_on_an_operation_that_does_not_exist_is_refused() -> None:
    """The plan's own numbering is the one thing it can be held to on its own,
    and a reference into nothing would linearise by crashing."""
    with pytest.raises(ValidationError, match=r"op2 depends on \[7\]"):
        plan(op(1), op(2, needs=[7]))


def test_an_operation_cannot_wait_on_itself() -> None:
    with pytest.raises(ValidationError, match="op1 depends on itself"):
        plan(op(1, needs=[1]))


def test_a_cycle_is_refused_and_named() -> None:
    """`middleware/model_retry.py` hands this message back for a free retry, so
    it has to say which dependency to drop rather than that one exists."""
    # `->` reads as "waits on", so the chain runs the way the dependencies do.
    with pytest.raises(ValidationError, match=r"op1 -> op2 -> op3 -> op1"):
        plan(op(1, needs=[2]), op(2, needs=[3]), op(3, needs=[1]))


def test_the_build_order_follows_the_dependencies_not_the_writing_order() -> None:
    """The whole reason for the graph. Written 4, 1, 3, 2 and built 1, 2, 3, 4."""
    written = plan(op(4, needs=[2, 3]), op(1), op(3, needs=[1]), op(2, needs=[1]))

    assert [held.id for held in linearise(written)] == [1, 2, 3, 4]


def test_an_operation_lands_beside_the_ones_it_consumes() -> None:
    """Depth-first, not breadth-first. Two independent chains come out whole
    rather than interleaved band by band, so a fillet reads next to the edge it
    rounds instead of in a later heap of details."""
    two_chains = plan(
        op(1),
        op(2, needs=[1]),
        op(3, needs=[2]),
        op(4),
        op(5, needs=[4]),
    )

    assert [held.id for held in linearise(two_chains)] == [1, 2, 3, 4, 5]


def test_the_same_plan_always_linearises_the_same_way() -> None:
    """The coder builds the derived order, so an order that moved between two
    runs of the same plan would be a difference nobody wrote."""
    written = plan(op(3, needs=[1]), op(2, needs=[1]), op(1), op(4, needs=[1]))

    assert len({tuple(held.id for held in linearise(written)) for _ in range(20)}) == 1


def test_every_operation_is_placed_exactly_once() -> None:
    """A diamond reaches its root twice; the coder must build it once."""
    diamond = plan(op(1), op(2, needs=[1]), op(3, needs=[1]), op(4, needs=[2, 3]))

    order = [held.id for held in linearise(diamond)]

    assert sorted(order) == [1, 2, 3, 4]
    assert len(order) == len(set(order))


def test_a_feature_no_operation_builds_is_reported() -> None:
    """The check the plan's validator cannot make: a feature id points into an
    answer the planner produced separately, so only something holding both can
    tell that sem2 was established and then dropped."""
    coverage = plan_coverage(
        plan(op(1, builds=[1]), op(2, builds=[3])), _features(1, 2, 3)
    )

    assert coverage.uncovered == [2]
    assert coverage.unknown == []
    assert not coverage.complete


def test_a_feature_the_plan_invents_is_reported() -> None:
    coverage = plan_coverage(plan(op(1, builds=[9])), _features(1))

    assert coverage.unknown == [9]
    assert coverage.uncovered == [1]


def test_a_plan_that_accounts_for_every_feature_is_complete() -> None:
    coverage = plan_coverage(
        plan(op(1, builds=[1, 2]), op(2, builds=[2])), _features(1, 2)
    )

    assert coverage.complete


def test_the_rendering_shows_the_derived_order_and_what_each_step_needs() -> None:
    """What the coder is handed. It is the order, not the graph: following it
    is the job, and a plan whose dependencies were wrong reads as wrong here."""
    rendered = render_plan(plan(op(2, needs=[1], builds=[4]), op(1, builds=[1])))
    lines = rendered.splitlines()

    assert "op1 (from nothing; builds sem1)" in lines[1]
    assert "op2 (after op1; builds sem4)" in lines[2]
    assert lines.index(next(x for x in lines if "op1 (" in x)) < lines.index(
        next(x for x in lines if "op2 (" in x)
    )


def test_an_operation_that_names_no_feature_says_so() -> None:
    """Rendering it as an empty list would read as a formatting slip; saying it
    plainly keeps the gap visible to whoever reads the plan."""
    assert "builds nothing named" in render_plan(plan(op(1)))


def test_the_gap_is_reported_by_naming_the_features_it_left_out() -> None:
    """What the planner is sent back with. Named rather than counted: "sem2,
    sem5" is something to go and plan, where "two are missing" sends the stage
    back to compare two lists it has already been given."""
    coverage = plan_coverage(plan(op(1, builds=[1])), _features(1, 2, 5))

    told = render_plan_coverage(coverage)

    assert "sem2, sem5" in told
    assert "sem1" not in told


def test_a_dangling_reference_is_reported_separately_from_a_gap() -> None:
    """They are different mistakes and take different corrections: one is a
    feature to go and build, the other a number to stop citing."""
    coverage = plan_coverage(plan(op(1, builds=[9])), _features(1))

    told = render_plan_coverage(coverage)

    assert "sem1" in told
    assert "sem9" in told
    assert told.index("sem1") < told.index("sem9")


def test_a_complete_plan_is_reported_as_nothing_at_all() -> None:
    """An empty string, not a sentence saying everything is fine: this only
    ever renders on the way back to a stage that has something to fix."""
    assert (
        render_plan_coverage(plan_coverage(plan(op(1, builds=[1])), _features(1))) == ""
    )


def test_a_fingerprint_follows_the_content_and_nothing_else() -> None:
    """Two plans that say the same thing are the same plan as far as anything
    downstream is concerned, and one that differs anywhere is not."""
    one = plan(op(1, builds=[1]))
    same = plan(op(1, builds=[1]))
    other = plan(op(1, builds=[2]))

    assert fingerprint(one) == fingerprint(same)
    assert fingerprint(one) != fingerprint(other)


def test_a_reading_knows_which_pair_it_was_taken_from() -> None:
    """The whole point of carrying the fingerprints. A reading kept in state
    outlives the work it measured, and this is what lets the two be replaced
    without anyone having to remember to throw the reading away."""
    established = _features(1, 2)
    measured = plan(op(1, builds=[1]), op(2, builds=[2]))
    coverage = plan_coverage(measured, established)

    assert coverage.describes(established, measured)


@pytest.mark.parametrize("changed", ["hypothesis", "plan"])
def test_a_reading_disowns_a_pair_that_has_moved_on(changed: str) -> None:
    established = _features(1, 2)
    measured = plan(op(1, builds=[1]), op(2, builds=[2]))
    coverage = plan_coverage(measured, established)

    if changed == "hypothesis":
        assert not coverage.describes(_features(1, 2, 3), measured)
    else:
        assert not coverage.describes(established, plan(op(1, builds=[1])))
