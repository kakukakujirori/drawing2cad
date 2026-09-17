"""What the operation plan contract refuses."""

from collections.abc import Sequence

import pytest
from pydantic import BaseModel, ValidationError

from zeroshot.pipeline.stages.operations.contracts import (
    Operation,
    OperationPlan,
    OperationVerb,
)


def op(
    name: str,
    *,
    builds: Sequence[int | str] = (),
    detail: str = "",
    verb: OperationVerb = OperationVerb.EXTRUDE,
) -> Operation:
    return Operation(
        name=name,
        verb=verb,
        detail=detail or f"operation {name}",
        semantics=[
            f"sem_feature_{held}" if isinstance(held, int) else held for held in builds
        ],
    )


def plan(*operations: Operation) -> OperationPlan:
    return OperationPlan(
        proposal=list(operations),
        rationale="because",
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


@pytest.mark.parametrize("name", ["main_bore", "sem-Main", "sem_"])
def test_semantics_entries_are_stable_semantic_names(name: str) -> None:
    with pytest.raises(ValidationError, match="usable semantic feature name"):
        plan(op("op_bore", builds=[name]))


def test_an_operation_does_not_repeat_one_semantic_name() -> None:
    with pytest.raises(ValidationError, match="more than once in semantics"):
        plan(op("op_bore", builds=["sem_main_bore", "sem_main_bore"]))


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
                semantics=[],
            )
        )
