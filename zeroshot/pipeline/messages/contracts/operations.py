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
    Parameter,
    SemanticFeature,
    SemanticHypothesis,
    render_parameter_values,
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

    name: str = Field(
        ...,
        description=(
            "A name for this step, unique within the plan, beginning op_ and "
            "carrying on in lower_snake_case: op_base_plate, op_bore_through, "
            "op_fillet_top_edges. The op_ marks it as a step, as sem_ marks a "
            "hypothesis feature, so that a step named after the feature it "
            "builds still reads as the step. Name it for what it does rather "
            "than for where it comes in the order, since the order is worked "
            "out from the dependencies and the name is what every later stage "
            "cites it by. Keep a step's name when you revise it, so that a "
            "reference to it stays true, and give a step you add a new name of "
            "its own rather than a number in a sequence."
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
            "sem_<feature>.geo_<claim>.<parameter> for a 3D claim or "
            "sem_<feature>.ev_<reading>.<parameter> for a drawing reading -- "
            "for example sem_main_bore.geo_cylinder.radius or "
            "sem_base_profile.ev_front_left_edge.start. The citation is "
            "resolved to the value before the coder reads it, so cite rather "
            "than copy. Write out plainly any number the hypothesis does not "
            "state: a depth or an offset you worked out yourself."
        ),
    )
    depends_on: list[str] = Field(
        ...,
        description=(
            "The operations whose result this one consumes, by their names -- "
            '["op_base_plate"]. Empty for an operation that starts from nothing. '
            "This is what fixes the build order; the order you happen to list "
            "operations in does not."
        ),
    )
    semantics: list[str] = Field(
        ...,
        description=(
            "The hypothesis features this operation helps build, by their "
            "stable sem_ names -- sem_main_bore, for example. Both operations "
            "and features are named identities: `depends_on` holds op_ names "
            "and `semantics` holds sem_ names. A "
            "feature may take several operations, and an operation may serve "
            "several features."
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
        graph: names that exist, and no step that waits on itself.
        """
        if not self.proposal:
            raise ValueError("proposal must hold at least one operation")

        names = [operation.name for operation in self.proposal]
        for name in names:
            _check_name(name)
        duplicated = sorted({name for name in names if names.count(name) > 1})
        if duplicated:
            raise ValueError(
                f"two operations are both called {', '.join(duplicated)}; give "
                "each one a name of its own, since a name is how every later "
                "stage tells them apart"
            )

        known = set(names)
        for operation in self.proposal:
            for semantic in operation.semantics:
                _check_semantic_name(semantic)
            duplicated_semantics = sorted(
                {
                    semantic
                    for semantic in operation.semantics
                    if operation.semantics.count(semantic) > 1
                }
            )
            if duplicated_semantics:
                raise ValueError(
                    f"{operation.name} lists {', '.join(duplicated_semantics)} "
                    "more than once in semantics"
                )
            unknown = sorted(set(operation.depends_on) - known)
            if unknown:
                raise ValueError(
                    f"{operation.name} depends on {', '.join(unknown)}, which "
                    "no operation in this plan produces"
                )
            if operation.name in operation.depends_on:
                raise ValueError(f"{operation.name} depends on itself")

        cycle = _first_cycle(self.proposal)
        if cycle:
            raise ValueError(
                "operations wait on each other and nothing can start: "
                + " -> ".join(cycle)
            )
        return self


# `op_` because a hypothesis feature is cited as `sem_main_bore`, and the two kinds of
# identifier travel together through prose the coder and the audit both read.
# A step named for the feature it builds is the likely case rather than the
# awkward one -- the operation that bores the main bore has little else to be
# called -- so the prefix is what keeps "the feature" and "the step that makes
# it" from arriving as the same word. It also puts every name out of reach of
# anything Python or the coding contract has already bound.
_NAME = re.compile(r"^op_[a-z][a-z0-9_]*$")
_SEMANTIC_NAME = re.compile(r"^sem_[a-z][a-z0-9_]*$")
_LONGEST_NAME = 40


def _check_name(name: str) -> None:
    if not _NAME.fullmatch(name):
        raise ValueError(
            f"{name!r} is not a usable operation name. Begin with op_ and "
            "carry on in lower_snake_case: op_base_plate, op_bore_through."
        )
    if len(name) > _LONGEST_NAME:
        raise ValueError(
            f"{name!r} is longer than {_LONGEST_NAME} characters. Name the "
            "step, do not describe it; the description belongs in `detail`."
        )


def _check_semantic_name(name: str) -> None:
    if not _SEMANTIC_NAME.fullmatch(name):
        raise ValueError(
            f"{name!r} is not a usable semantic feature name. Begin with "
            "sem_ and carry on in lower_snake_case: sem_base_body, "
            "sem_main_bore."
        )


def _first_cycle(operations: Iterable[Operation]) -> list[str]:
    """One cycle, named, or an empty list.

    Named rather than merely detected because the message is what the model
    reads back through `middleware/model_retry.py`: "base -> bore -> base" says
    which dependency to drop, where "the plan is cyclic" does not.
    """
    needs = {operation.name: list(operation.depends_on) for operation in operations}
    done: set[str] = set()
    path: list[str] = []

    def walk(name: str) -> list[str]:
        if name in path:
            return [*path[path.index(name) :], name]
        if name in done:
            return []
        path.append(name)
        for needed in needs.get(name, ()):
            found = walk(needed)
            if found:
                return found
        path.pop()
        done.add(name)
        return []

    for name in needs:
        found = walk(name)
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
    by_name = {operation.name: operation for operation in plan.proposal}
    order: list[Operation] = []
    placed: set[str] = set()

    def place(operation: Operation) -> None:
        if operation.name in placed:
            return
        for needed in operation.depends_on:
            place(by_name[needed])
        placed.add(operation.name)
        order.append(operation)

    for operation in plan.proposal:
        place(operation)
    return order


# Canonical addresses name the feature, the claim/reading within it, and the
# parameter by its semantic name: `sem_main_bore.geo_cylinder.radius`,
# `sem_base.ev_front_edge.start`. The parameter needs no identity of its own
# because `_checked` makes its vocabulary name unique inside that member.
_REFERENCE = re.compile(
    r"\b(sem_[a-z][a-z0-9_]*)\.((?:geo|ev)_[a-z][a-z0-9_]*)\.([a-z_]+)\b(?!\.)"
)
_REFERENCE_LIKE = re.compile(
    r"\bsem_[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+\b"
)

# A number written to this many decimal places did not come from anybody's
# head. Below it, `25` and `1.5` are the planner's own words and are left alone.
_COPIED_DECIMALS = 4
_DECIMAL = re.compile(rf"\d+\.\d{{{_COPIED_DECIMALS},}}")

# How many of a fault's examples to name before counting the rest.
_NAMED_AT_MOST = 3


def _parameters_of(
    feature: SemanticFeature, member_name: str, parameter_name: str
) -> list[Parameter]:
    """The parameter a canonical member address means, or nothing.

    The explicit geo_/ev_ namespace makes choosing by list position, geometry
    kind, or first match unnecessary. Member names are unique within their
    respective group, and parameter names are unique within each member.
    """
    if member_name.startswith("geo_"):
        return [
            parameter
            for claim in feature.geometry
            if claim.name == member_name
            for parameter in claim.parameters
            if parameter.name.value == parameter_name
        ]
    if member_name.startswith("ev_"):
        return [
            parameter
            for reading in feature.evidence
            if reading.name == member_name
            for parameter in reading.parameters
            if parameter.name.value == parameter_name
        ]
    return []


def resolve_reference(text: str, hypothesis: SemanticHypothesis) -> str:
    """Append the exact value named by every semantic reference in `text`.

    This is what makes referencing worth asking for. A planner made to carry
    geometry in prose has to retype it, and a retyped number can be mistyped: in
    one run a spline control point the hypothesis states to fourteen places came
    across as `71.648586?`, and the coder spent two turns finding the broken
    token. A reference resolved here never passes through a model's output, so
    that failure cannot happen rather than being caught after it has.

    A vector remains one parameter, so `sem_base.ev_front_edge.start` resolves
    to its whole `[x y]` pair. Long lists are expanded too: a spline's exact poles are more
    useful to the coder than an address it then has to search for elsewhere.
    An absent reference is left standing; plan review refuses it before coding.
    """
    by_name = {feature.name: feature for feature in hypothesis.proposal}

    def substitute(match: re.Match[str]) -> str:
        feature = by_name.get(match[1])
        if feature is None:
            return match[0]
        parameters = _parameters_of(feature, match[2], match[3])
        if len(parameters) != 1:
            return match[0]
        return f"{match[0]} ({render_parameter_values(parameters[0].values)})"

    return _REFERENCE.sub(substitute, text)


def _unresolved(detail: str, by_name: Mapping[str, SemanticFeature]) -> list[str]:
    """The references in `detail` that do not name exactly one parameter."""
    broken = []
    for token in _REFERENCE_LIKE.finditer(detail):
        match = _REFERENCE.fullmatch(token[0])
        if match is None:
            broken.append(token[0])
            continue
        feature = by_name.get(match[1])
        if feature is None or len(_parameters_of(feature, match[2], match[3])) != 1:
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

    uncovered: list[str]
    unknown: list[str]
    transcribed: dict[str, list[str]]
    unresolved: dict[str, list[str]]
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
    established = {feature.name for feature in hypothesis.proposal}
    built = {sem for operation in plan.proposal for sem in operation.semantics}
    by_name = {feature.name: feature for feature in hypothesis.proposal}
    held = _numbers_in(hypothesis)

    return PlanReview(
        uncovered=sorted(established - built),
        unknown=sorted(built - established),
        transcribed={
            operation.name: copied
            for operation in plan.proposal
            if (copied := _transcribed(operation.detail, held))
        },
        unresolved={
            operation.name: broken
            for operation in plan.proposal
            if (broken := _unresolved(operation.detail, by_name))
        },
        of_hypothesis=fingerprint(hypothesis),
        of_plan=fingerprint(plan),
    )


def operation_heading(operation: Operation, *, produces: str = "") -> str:
    """How an operation announces itself, wherever it is written down.

    One function because the plan the planner reads, the plan the coder reads
    and the marker the coder writes under all have to say the same thing about
    the same step. Two renderings that drift apart would leave the coder
    holding one account of what a step is for and the machine holding another.
    """
    needs = (
        "after " + ", ".join(operation.depends_on)
        if operation.depends_on
        else "needs nothing"
    )
    builds = ", ".join(operation.semantics)
    output = f" -> {produces}" if produces else ""
    return (
        f"{operation.name} {operation.verb.value}{output} "
        f"({needs}; builds {builds or 'nothing named'})"
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
    # The step number is put on here rather than held in the plan: it is a
    # position, and a position derived from the dependencies is not something
    # the planner should have to keep in step with them.
    for step, operation in enumerate(linearise(plan), start=1):
        lines.append(
            f"  step {step}  {operation_heading(operation)} "
            f"{resolve_reference(operation.detail, hypothesis)}"
        )
    lines.append(f"rationale: {plan.rationale}")
    return "\n".join(lines)


def render_plan_review(review: PlanReview) -> str:
    """What to tell the planner about a plan the hypothesis contradicts.

    Named rather than counted throughout: "sem_main_bore" is something to go
    and plan, where "two features are missing" sends the stage back to compare
    two lists it has already been given.
    """
    faults = []
    if review.uncovered:
        named = ", ".join(review.uncovered)
        faults.append(
            f"The hypothesis establishes {named}, and no operation in the plan "
            "builds them. Add the operations they take, or say in the rationale "
            "why the part is complete without them."
        )
    if review.unknown:
        named = ", ".join(review.unknown)
        faults.append(
            f"The plan cites {named}, which the hypothesis does not contain. "
            "Cite the features it does have."
        )
    for operation, copied in sorted(review.transcribed.items()):
        # A first plan can hold well over a hundred of these, and naming every
        # one would bury the instruction under the evidence for it.
        named = ", ".join(copied[:_NAMED_AT_MOST])
        if len(copied) > _NAMED_AT_MOST:
            named += f" and {len(copied) - _NAMED_AT_MOST} more"
        faults.append(
            f"{operation} writes out {named}, which the hypothesis already "
            "holds. Cite it as sem_<feature>.geo_<claim>.<parameter> or "
            "sem_<feature>.ev_<reading>.<parameter> instead; the number is put "
            "in for you, and a number retyped is a number that can be mistyped."
        )
    for operation, broken in sorted(review.unresolved.items()):
        named = ", ".join(broken)
        faults.append(
            f"{operation} refers to {named}, which does not identify exactly "
            "one parameter in the hypothesis. Use the member name shown there, "
            "such as sem_main_bore.geo_cylinder.radius or "
            "sem_main_bore.ev_front_circle.center."
        )
    return " ".join(faults)
