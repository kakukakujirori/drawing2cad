"""The planning stage's answer: what to build, and what each step needs first.

A plan is a graph rather than a list because the order a model writes its steps
in is a guess about sequencing that nothing can check, while a dependency it
states is a claim that can be. The sequence the coder is given is derived from
the graph here, by code, so a plan that is right cannot be spoiled by being
written down in the wrong order.
"""

import re
from collections.abc import Iterable
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
_NAME = re.compile(r"^op_[a-z0-9_]+$")
_SEMANTIC_NAME = re.compile(r"^sem_[a-z0-9_]+$")
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
    r"\b(sem_[a-z0-9_]+)\.((?:geo|ev)_[a-z0-9_]+)\.([a-z_]+)\b(?!\.)"
)


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
    An absent reference is left standing; contextual validation refuses it
    before coding.
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
