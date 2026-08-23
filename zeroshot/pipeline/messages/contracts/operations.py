"""The planning stage's answer: what to build, and what each step needs first.

A plan is a graph rather than a list because the order a model writes its steps
in is a guess about sequencing that nothing can check, while a dependency it
states is a claim that can be. The sequence the coder is given is derived from
the graph here, by code, so a plan that is right cannot be spoiled by being
written down in the wrong order.
"""

import re
from collections.abc import Iterable, Mapping
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from zeroshot.pipeline.messages.contracts.fingerprint import fingerprint
from zeroshot.pipeline.messages.contracts.semantics import (
    SemanticFeature,
    SemanticHypothesis,
)


# The modelling operations a plan may be made of.
#
# A verb belongs here only if the step it names hands the next step a solid.
# That rule is what keeps a plan a plan: there is no verb for declaring a
# convention or drawing a profile, so neither can be entered as an operation,
# and neither can satisfy the coverage check while building nothing.
#
# The list is enumerated from the 25 solid-producing methods of `cq.Workplane`
# (of 107 public) and OCC's BRepPrimAPI, BRepAlgoAPI, BRepFilletAPI,
# BRepOffsetAPI and BRepFeat.
#
# Left out on purpose:
# - primitives, since a box is an extrusion and a sphere a revolution;
# - Splitter, which is cutting with a tool
# - MakeOffsetShape and DraftAngle, which nothing in CadQuery builds
# - Array helpers, which place points and return no solid.
#
# A verb missing from a closed enum costs a mislabel rather than a refusal,
# since `detail` still says what the step does.
#
# No docstring: a class docstring reaches the model as the schema's
# `description`, and this is for whoever edits the list.
class OperationVerb(StrEnum):
    EXTRUDE = "extrude"
    REVOLVE = "revolve"
    SWEEP = "sweep"
    LOFT = "loft"
    CUT = "cut"
    HOLE = "hole"
    FILLET = "fillet"
    CHAMFER = "chamfer"
    SHELL = "shell"
    UNION = "union"
    INTERSECT = "intersect"
    MIRROR = "mirror"


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
    verb: OperationVerb = Field(
        ...,
        description=(
            "Which modelling operation this step performs. Each step takes "
            "the part as it stands and returns it changed, so every entry "
            "names something that acts on the solid. A profile belongs to the "
            "entry that extrudes, revolves or sweeps it."
        ),
    )
    detail: str = Field(
        ...,
        description=(
            "What this step does, in a sentence or two: the profile or edges "
            "it acts on, the direction it goes in, and where on the part it "
            "lands. Cite a measurement the hypothesis states as "
            "sem<id>.<parameter> -- sem7.radius, or sem1.torus.major_radius "
            "where one feature claims more than one geometry. The citation is "
            "resolved to the value before the coder reads it, so cite rather "
            "than copy. Write out plainly any number the hypothesis does not "
            "state: a depth or an offset you worked out yourself."
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
        description=(
            "Why this decomposition, and why the dependencies it states. "
            "Reasoning about the shape of the plan; measurements and positions "
            "belong to the operations themselves."
        ),
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


# `sem7.radius`, or `sem1.torus.major_radius` when a feature claims more than
# one geometry. A feature is the deepest address: anything finer would be a
# second numbering to keep in step with the hypothesis.
_REFERENCE = re.compile(r"\bsem(\d+)\.([a-z_]+)(?:\.([a-z_]+))?\b")

# A number written to this many decimal places did not come from anybody's
# head. Below it, `25` and `1.5` are the planner's own words and are left alone.
_COPIED_DECIMALS = 4
_DECIMAL = re.compile(rf"\d+\.\d{{{_COPIED_DECIMALS},}}")

# How many of a fault's examples to name before counting the rest.
_NAMED_AT_MOST = 3


def _values_of(feature: SemanticFeature, kind: str | None, name: str) -> list[float]:
    """Every number `feature` holds under `name`, or nothing if it holds none.

    Either half of a feature can answer, because the planner should not have to
    know which half a name lives in: `sem7.radius` is a question about sem7,
    not about how the contract files it. `geometry` answers first, since that
    is the feature's own claim -- a bore states one radius there and may carry
    half a dozen arcs in its `evidence`, and the one is what a plan means.

    A kind-qualified reference asks about `geometry` alone. `sem1.torus` names
    a claim, and the readings are not filed under one.
    """
    claimed = [
        parameter.values
        for claim in feature.geometry
        if kind is None or claim.kind.value == kind
        for parameter in claim.parameters
        if parameter.name.value == name
    ]
    if claimed or kind is not None:
        return [number for values in claimed for number in values]

    return [
        number
        for reading in feature.evidence
        for parameter in reading.parameters
        if parameter.name.value == name
        for number in parameter.values
    ]


def resolve_reference(text: str, hypothesis: SemanticHypothesis) -> str:
    """Replace each `sem<id>.<name>` in `text` with the number it stands for.

    This is what makes referencing worth asking for. A planner made to carry
    geometry in prose has to retype it, and a retyped number can be mistyped: in
    one run a spline control point the hypothesis states to fourteen places came
    across as `71.648586?`, and the coder spent two turns finding the broken
    token. A reference resolved here never passes through a model's output, so
    that failure cannot happen rather than being caught after it has.

    Only single numbers are substituted. A control-point list expands to
    something nobody would read, and the coder holds the hypothesis anyway, so
    the reference is left standing as the address it is.
    """
    by_id = {feature.id: feature for feature in hypothesis.proposal}

    def substitute(match: re.Match[str]) -> str:
        feature = by_id.get(int(match[1]))
        if feature is None:
            return match[0]
        kind, name = (match[2], match[3]) if match[3] else (None, match[2])
        values = _values_of(feature, kind, name)
        return f"{match[0]} ({values[0]})" if len(values) == 1 else match[0]

    return _REFERENCE.sub(substitute, text)


def _unresolved(detail: str, by_id: Mapping[int, SemanticFeature]) -> list[str]:
    """The references in `detail` that stand for nothing."""
    broken = []
    for match in _REFERENCE.finditer(detail):
        feature = by_id.get(int(match[1]))
        kind, name = (match[2], match[3]) if match[3] else (None, match[2])
        if feature is None or not _values_of(feature, kind, name):
            broken.append(match[0])
    return broken


def _transcribed(detail: str, held: Iterable[float]) -> list[str]:
    """The numbers in `detail` that the hypothesis already holds.

    Matched at the precision the planner wrote, so a value rounded on the way
    across is caught alongside one copied whole. Numbers the planner worked out
    for itself -- half a width, a clearance -- match nothing and are left alone,
    which is what makes this safe to refuse on.
    """
    held = list(held)
    copied = []
    for literal in _DECIMAL.findall(detail):
        places = len(literal.split(".")[1])
        if any(round(number, places) == float(literal) for number in held):
            copied.append(literal)
    return copied


def _numbers_in(hypothesis: SemanticHypothesis) -> list[float]:
    return [
        number
        for feature in hypothesis.proposal
        for parameters in (
            [p for claim in feature.geometry for p in claim.parameters]
            + [p for reading in feature.evidence for p in reading.parameters]
        )
        for number in parameters.values
    ]


class PlanReview(BaseModel):
    """Everything wrong with a plan that only the hypothesis can reveal.

    Four faults, one reading, because they are all the same kind of thing: a
    claim the plan makes about an answer produced by another stage, which
    neither answer can check alone. `uncovered` is the one that would otherwise
    go unnoticed -- a feature established and then never built comes out as a
    model that is whole and quietly missing something.

    A model rather than a tuple because it is checkpointed: a `NamedTuple`
    round-trips with its fields turned into lists, so a restored run would hold
    something that compares unequal to what it saved.
    """

    model_config = ConfigDict(extra="forbid")

    uncovered: list[int]
    unknown: list[int]
    transcribed: dict[int, list[str]]
    unresolved: dict[int, list[str]]
    of_hypothesis: str
    of_plan: str

    @property
    def sound(self) -> bool:
        return not (
            self.uncovered or self.unknown or self.transcribed or self.unresolved
        )

    def describes(self, hypothesis: SemanticHypothesis, plan: OperationPlan) -> bool:
        """Whether this still says anything about the pair given.

        A reading kept in state outlives the work it measured. Asking it this
        first is what lets the two be replaced without anyone having to
        remember to throw the reading away.
        """
        return self.of_hypothesis == fingerprint(
            hypothesis
        ) and self.of_plan == fingerprint(plan)


def review_plan(plan: OperationPlan, hypothesis: SemanticHypothesis) -> PlanReview:
    """Read the plan against the hypothesis it was made from.

    Both whole, rather than the hypothesis broken into its ids and its
    fingerprint: split apart they are two things that have to be about the same
    answer with nothing to check that they are, which is the shape of mistake
    this reading exists to catch.
    """
    established = {feature.id for feature in hypothesis.proposal}
    built = {sem for operation in plan.proposal for sem in operation.semantics}
    by_id = {feature.id: feature for feature in hypothesis.proposal}
    held = _numbers_in(hypothesis)

    return PlanReview(
        uncovered=sorted(established - built),
        unknown=sorted(built - established),
        transcribed={
            operation.id: copied
            for operation in plan.proposal
            if (copied := _transcribed(operation.detail, held))
        },
        unresolved={
            operation.id: broken
            for operation in plan.proposal
            if (broken := _unresolved(operation.detail, by_id))
        },
        of_hypothesis=fingerprint(hypothesis),
        of_plan=fingerprint(plan),
    )


def render_plan(plan: OperationPlan, hypothesis: SemanticHypothesis) -> str:
    """The plan as the coder reads it: one line per step, in build order.

    The derived order is shown rather than the graph, because the coder's job
    is to follow it. It is stated as an order so that a plan whose
    dependencies were wrong reads as wrong here, at the one point where a
    person or a later stage still looks at it.

    The hypothesis is taken as well because the plan's references are resolved
    here. That is the whole of the arrangement: the planner names a number and
    this puts it in, so no number is ever retyped by a model.
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
            f"  op{operation.id} {operation.verb.value} "
            f"({needs}; builds {builds or 'nothing named'}) "
            f"{resolve_reference(operation.detail, hypothesis)}"
        )
    lines.append(f"rationale: {plan.rationale}")
    return "\n".join(lines)


def render_plan_review(review: PlanReview) -> str:
    """What to tell the planner about a plan the hypothesis contradicts.

    Named rather than counted throughout: "sem2 and sem5" is something to go
    and plan, where "two features are missing" sends the stage back to compare
    two lists it has already been given.
    """
    faults = []
    if review.uncovered:
        named = ", ".join(f"sem{identifier}" for identifier in review.uncovered)
        faults.append(
            f"The hypothesis establishes {named}, and no operation in the plan "
            "builds them. Add the operations they take, or say in the rationale "
            "why the part is complete without them."
        )
    if review.unknown:
        named = ", ".join(f"sem{identifier}" for identifier in review.unknown)
        faults.append(
            f"The plan cites {named}, which the hypothesis does not contain. "
            "Cite the features it does have."
        )
    for identifier, copied in sorted(review.transcribed.items()):
        # A first plan can hold well over a hundred of these, and naming every
        # one would bury the instruction under the evidence for it.
        named = ", ".join(copied[:_NAMED_AT_MOST])
        if len(copied) > _NAMED_AT_MOST:
            named += f" and {len(copied) - _NAMED_AT_MOST} more"
        faults.append(
            f"op{identifier} writes out {named}, which the hypothesis already "
            "holds. Cite it as sem<id>.<parameter> instead; the number is put "
            "in for you, and a number retyped is a number that can be mistyped."
        )
    for identifier, broken in sorted(review.unresolved.items()):
        named = ", ".join(broken)
        faults.append(
            f"op{identifier} refers to {named}, which the hypothesis does not "
            "hold. Refer to a feature it has, and to a measurement that feature "
            "states."
        )
    return " ".join(faults)
