"""The planning stage's answer: what to build, and what each step needs first.

A plan is a graph rather than a list because the order a model writes its steps
in is a guess about sequencing that nothing can check, while a dependency it
states is a claim that can be. The sequence the coder is given is derived from
the graph here, by code, so a plan that is right cannot be spoiled by being
written down in the wrong order.
"""

from collections.abc import Iterable
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from zeroshot.pipeline.messages.contracts.fingerprint import fingerprint
from zeroshot.pipeline.messages.contracts.semantics import SemanticHypothesis


class Operation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int = Field(
        ...,
        description=(
            "Identifier for this operation, unique within the plan and 1 or "
            "greater. Later stages cite it as op<id>. Keep an operation's id "
            "when you revise it, so that a reference to it stays true."
        ),
    )
    operation: str = Field(
        ...,
        description=(
            "One modelling operation, carrying the dimensions, references and "
            "directions needed to carry it out without guessing."
        ),
    )
    depends_on: list[int] = Field(
        ...,
        description=(
            "The operations whose result this one consumes, as their plain id "
            'numbers -- 3, not "op3". Empty for an operation that starts '
            "from nothing. This is what fixes the build order; the order you "
            "happen to list operations in does not."
        ),
    )
    semantics: list[int] = Field(
        ...,
        description=(
            "The hypothesis features this operation helps build, as their "
            "plain id numbers -- 7 for the feature written sem7, not "
            '"sem7". A feature may take several operations, and an '
            "operation may serve several features."
        ),
    )


class OperationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal: list[Operation] = Field(
        ...,
        description=(
            "Every operation the part takes. List them in whatever order you "
            "reason in; `depends_on` is what decides the order they are built "
            "in."
        ),
    )
    rationale: str = Field(
        ...,
        description="Why this decomposition, and why the dependencies it states.",
    )

    @model_validator(mode="after")
    def require_a_resolvable_acyclic_plan(self) -> Self:
        """Everything about the plan that can be checked without the drawing.

        Whether the plan covers the hypothesis cannot be settled here -- the
        features live in another answer this model never sees -- so that check
        belongs to the graph. What is checkable here is that the graph is a
        graph: ids that exist, and no step that waits on itself.
        """
        if not self.proposal:
            raise ValueError("proposal must hold at least one operation")

        ids = [operation.id for operation in self.proposal]
        if any(identifier < 1 for identifier in ids):
            raise ValueError("operation ids must be 1 or greater")
        if len(set(ids)) != len(ids):
            raise ValueError("operation ids must be unique; renumber the plan from 1")

        known = set(ids)
        for operation in self.proposal:
            unknown = sorted(set(operation.depends_on) - known)
            if unknown:
                raise ValueError(
                    f"op{operation.id} depends on {unknown}, which no operation "
                    "in this plan produces"
                )
            if operation.id in operation.depends_on:
                raise ValueError(f"op{operation.id} depends on itself")

        cycle = _first_cycle(self.proposal)
        if cycle:
            raise ValueError(
                "operations wait on each other and nothing can start: "
                + " -> ".join(f"op{identifier}" for identifier in cycle)
            )
        return self


def _first_cycle(operations: Iterable[Operation]) -> list[int]:
    """One cycle, named, or an empty list.

    Named rather than merely detected because the message is what the model
    reads back through `middleware/model_retry.py`: "op3 -> op7 -> op3" says
    which dependency to drop, where "the plan is cyclic" does not.
    """
    needs = {operation.id: list(operation.depends_on) for operation in operations}
    done: set[int] = set()
    path: list[int] = []

    def walk(identifier: int) -> list[int]:
        if identifier in path:
            return [*path[path.index(identifier) :], identifier]
        if identifier in done:
            return []
        path.append(identifier)
        for needed in needs.get(identifier, ()):
            found = walk(needed)
            if found:
                return found
        path.pop()
        done.add(identifier)
        return []

    for identifier in needs:
        found = walk(identifier)
        if found:
            return found
    return []


def linearise(plan: OperationPlan) -> list[Operation]:
    """The plan as the single sequence the coder builds, dependencies first.

    Depth-first rather than breadth-first so that an operation arrives directly
    after the ones it consumes: a fillet lands beside the edge it rounds
    instead of in a later band with every other detail. Where the graph leaves
    two operations free to go in either order, the order they were written in
    decides, so the same plan always linearises the same way.
    """
    by_id = {operation.id: operation for operation in plan.proposal}
    order: list[Operation] = []
    placed: set[int] = set()

    def place(operation: Operation) -> None:
        if operation.id in placed:
            return
        for needed in operation.depends_on:
            place(by_id[needed])
        placed.add(operation.id)
        order.append(operation)

    for operation in plan.proposal:
        place(operation)
    return order


class PlanCoverage(BaseModel):
    """Which features the plan accounts for, and which it invents.

    `uncovered` is the one worth having: a feature the hypothesis established
    and no operation builds is a feature lost between two stages, which nothing
    downstream can notice -- the model comes out whole, just missing something.

    A model rather than a tuple because it is checkpointed: a `NamedTuple`
    round-trips with its fields turned into lists, so a restored run would hold
    something that compares unequal to what it saved.
    """

    model_config = ConfigDict(extra="forbid")

    uncovered: list[int]
    unknown: list[int]
    of_hypothesis: str
    of_plan: str

    @property
    def complete(self) -> bool:
        return not self.uncovered and not self.unknown

    def describes(self, hypothesis: SemanticHypothesis, plan: OperationPlan) -> bool:
        """Whether this still says anything about the pair given.

        A reading kept in state outlives the work it measured. Asking it this
        first is what lets the two be replaced without anyone having to
        remember to throw the reading away.
        """
        return self.of_hypothesis == fingerprint(
            hypothesis
        ) and self.of_plan == fingerprint(plan)


def plan_coverage(plan: OperationPlan, hypothesis: SemanticHypothesis) -> PlanCoverage:
    """Compare what the plan builds against what the hypothesis established.

    Both whole, rather than the hypothesis broken into its ids and its
    fingerprint: split apart they are two things that have to be about the same
    answer with nothing to check that they are, which is the shape of mistake
    this reading exists to catch.
    """
    established = {feature.id for feature in hypothesis.proposal}
    built = {sem for operation in plan.proposal for sem in operation.semantics}
    return PlanCoverage(
        uncovered=sorted(established - built),
        unknown=sorted(built - established),
        of_hypothesis=fingerprint(hypothesis),
        of_plan=fingerprint(plan),
    )


def render_plan(plan: OperationPlan) -> str:
    """The plan as the coder reads it: one line per step, in build order.

    The derived order is shown rather than the graph, because the coder's job
    is to follow it. It is stated as an order so that a plan whose
    dependencies were wrong reads as wrong here, at the one point where a
    person or a later stage still looks at it.
    """
    heading = (
        "Build in this order, which follows from the plan's dependencies "
        "rather than the order it was written in:"
    )
    lines = [heading]
    for operation in linearise(plan):
        needs = (
            "after "
            + ", ".join(f"op{identifier}" for identifier in operation.depends_on)
            if operation.depends_on
            else "from nothing"
        )
        builds = ", ".join(f"sem{identifier}" for identifier in operation.semantics)
        lines.append(
            f"  op{operation.id} ({needs}; builds {builds or 'nothing named'}) "
            f"{operation.operation}"
        )
    lines.append(f"rationale: {plan.rationale}")
    return "\n".join(lines)


def render_plan_coverage(coverage: PlanCoverage) -> str:
    """What to tell the planner about the features its plan did not account for.

    Named rather than counted: "sem2 and sem5" is something to go and plan,
    where "two features are missing" sends the stage back to compare two lists
    it has already been given.
    """
    faults = []
    if coverage.uncovered:
        named = ", ".join(f"sem{identifier}" for identifier in coverage.uncovered)
        faults.append(
            f"The hypothesis establishes {named}, and no operation in the plan "
            "builds them. Add the operations they take, or say in the rationale "
            "why the part is complete without them."
        )
    if coverage.unknown:
        named = ", ".join(f"sem{identifier}" for identifier in coverage.unknown)
        faults.append(
            f"The plan cites {named}, which the hypothesis does not contain. "
            "Cite the features it does have."
        )
    return " ".join(faults)
