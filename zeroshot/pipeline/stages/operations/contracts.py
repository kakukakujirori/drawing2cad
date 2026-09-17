"""The planning stage's answer: the operations to build, listed in build order."""

import re
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from zeroshot.pipeline.stages.resolve_refs import without_annotations
from zeroshot.pipeline.stages.types import Member


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
            "op_fillet_top_edges. The op_ marks it as a step, as sem_ marks an "
            "interpreted feature, so that a step named after the feature it "
            "builds still reads as the step. Later stages cite a step by its "
            "name, and a revision may insert or move steps, so name it for "
            "what it does, not for its position. Keep a step's name when you "
            "revise it, and give a step you add a new name of its own."
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
            "lands. Preserve the interpretation's datum and feature placement. "
            "Cite a feature parameter as sem_main_bore.radius or "
            "sem_main_bore.center, or a printed figure as "
            "dim_bore_diameter.nominal_value. The pipeline annotates references "
            "with their scalar, array or null values. Null means unknown, not zero. "
            "Write out only construction choices that the interpretation does not state."
        ),
    )
    semantics: list[str] = Field(
        ...,
        description=(
            "The interpreted features this operation helps build, by their "
            "stable sem_ names -- sem_main_bore, for example. A feature may take several operations, and an operation may serve "
            "several features."
        ),
    )


class OperationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal: list[Operation] = Field(
        ...,
        description=(
            "Every operation the part takes, in build order. Each one changes "
            "the previous operation's result, unless its `detail` says which "
            "results it takes instead."
        ),
    )
    rationale: str = Field(
        ...,
        description=(
            "Why this decomposition and this order. "
            "Reasoning about the shape of the plan; measurements and positions "
            "belong to the operations themselves."
        ),
    )

    @model_validator(mode="after")
    def require_well_named_operations(self) -> Self:
        """Checks that need no drawing; feature coverage belongs to the graph."""
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
        return self

    def members(self) -> dict[str, Member]:
        """Each operation, and the features and dimensions it cites.

        The previous operation is compared too, since it is the implicit input.
        The pipeline annotates references with values, so compare the text as written.
        """
        names = [None, *(operation.name for operation in self.proposal)]
        return {
            operation.name: Member(
                (
                    operation.model_copy(
                        update={"detail": without_annotations(operation.detail)}
                    ),
                    previous,
                ),
                frozenset({*operation.semantics, *_CITED.findall(operation.detail)}),
            )
            for operation, previous in zip(self.proposal, names, strict=False)
        }


# `op_` because a interpreted feature is cited as `sem_main_bore`, and the two kinds of
# identifier travel together through prose the coder and the audit both read.
# A step named for the feature it builds is the likely case rather than the
# awkward one -- the operation that bores the main bore has little else to be
# called -- so the prefix is what keeps "the feature" and "the step that makes
# it" from arriving as the same word. It also puts every name out of reach of
# anything Python or the coding contract has already bound.
_NAME = re.compile(r"^op_[a-z0-9_]+$")
_SEMANTIC_NAME = re.compile(r"^sem_[a-z0-9_]+$")
_LONGEST_NAME = 40
_CITED = re.compile(r"\b(?:sem|dim)_[a-z0-9_]+")


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
