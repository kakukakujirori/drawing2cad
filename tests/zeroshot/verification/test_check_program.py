import pytest

from zeroshot.pipeline.messages.contracts import (
    Operation,
    OperationPlan,
    OperationVerb,
)
from zeroshot.pipeline.verification import ProgramCheck, check_program


def _operation(name: str, *, depends_on: tuple[str, ...] = ()) -> Operation:
    return Operation(
        name=name,
        verb=OperationVerb.EXTRUDE,
        detail=f"build {name}",
        depends_on=list(depends_on),
        semantics=["sem_feature_1"],
    )


def _plan(*operations: Operation) -> OperationPlan:
    return OperationPlan(
        proposal=list(operations), rationale="build in dependency order"
    )


def test_accepts_one_module_level_result_for_every_planned_operation() -> None:
    plan = _plan(
        _operation("op_base"),
        _operation("op_hole", depends_on=("op_base",)),
    )
    source = """\
part = object()
ret_base = part
ret_hole: object = ret_base.cut(part)
result = ret_hole
"""

    assert check_program(source, plan) == ProgramCheck(
        result_assigned=True,
    )


def test_reports_missing_operations_in_a_stable_order_without_scheduling_them() -> None:
    plan = _plan(
        _operation("op_a_hole", depends_on=("op_z_base",)),
        _operation("op_z_base"),
    )

    check = check_program("result = object()", plan)

    # A check reports a set of violations; it does not impose a build order.
    assert check.missing_operations == ("op_a_hole", "op_z_base")
    assert not check.sound


def test_reports_a_reserved_return_name_for_an_unknown_operation() -> None:
    plan = _plan(_operation("op_base"))
    source = """\
ret_base = object()
ret_unplanned = ret_base
result = ret_base
"""

    check = check_program(source, plan)

    assert check.unknown_operations == ("op_unplanned",)
    assert not check.sound


def test_local_return_like_helpers_do_not_claim_an_operation() -> None:
    plan = _plan(_operation("op_base"))
    source = """\
def finish(ret_input):
    ret_temporary = ret_input
    return ret_temporary

ret_base = finish(object())
result = ret_base
"""

    assert check_program(source, plan).sound


def test_requires_a_module_level_result_assignment() -> None:
    plan = _plan(_operation("op_base"))
    source = """\
ret_base = object()

def build():
    result = ret_base
    return result
"""

    check = check_program(source, plan)

    assert not check.result_assigned
    assert not check.sound


def test_a_nested_return_assignment_does_not_implement_an_operation() -> None:
    plan = _plan(_operation("op_base"))
    source = """\
def build():
    ret_base = object()
    return ret_base

result = build()
"""

    check = check_program(source, plan)

    assert check.missing_operations == ("op_base",)
    assert not check.sound


def test_an_annotation_without_a_value_is_not_an_assignment() -> None:
    plan = _plan(_operation("op_base"))

    check = check_program("ret_base: object\nresult = object()", plan)

    assert check.missing_operations == ("op_base",)


@pytest.mark.parametrize(
    "source",
    (
        "ret_base.value = object()\nresult = object()",
        "ret_base[0] = object()\nresult = object()",
    ),
)
def test_mutating_a_return_object_does_not_assign_the_return_name(source: str) -> None:
    plan = _plan(_operation("op_base"))

    check = check_program(source, plan)

    assert check.missing_operations == ("op_base",)
    assert check.result_assigned


def test_mutating_a_result_object_does_not_assign_result() -> None:
    plan = _plan(_operation("op_base"))

    check = check_program(
        "ret_base = object()\nresult.value = ret_base",
        plan,
    )

    assert check.missing_operations == ()
    assert not check.result_assigned


def test_helpers_comments_and_strings_do_not_create_unknown_operations() -> None:
    plan = _plan(_operation("op_base"))
    source = """\
# ret_commented_out = object()
description = "ret_named_in_prose"
part = object()
ret_base = part
result = ret_base
"""

    assert check_program(source, plan).sound


def test_a_result_that_hands_back_its_input_does_not_implement_its_operation() -> None:
    plan = _plan(_operation("op_base"), _operation("op_hole"))
    source = """\
ret_base = object()
ret_hole = ret_base.clean()
result = ret_hole
"""

    check = check_program(source, plan)

    assert check.identity_operations == ("ret_hole",)
    assert not check.sound


def test_a_tidy_up_after_real_work_still_implements_its_operation() -> None:
    plan = _plan(_operation("op_base"), _operation("op_hole"))
    source = """\
ret_base = object()
ret_hole = ret_base.cut(object())
ret_hole = ret_hole.clean()
result = ret_hole
"""

    assert check_program(source, plan).sound


@pytest.mark.parametrize(
    "assignment",
    (
        "ret_hole = drill(ret_base)",
        "ret_hole = ret_base.cut(object())",
        "ret_hole = ret_base.union(object()).clean()",
        "ret_hole = ret_base.copy(deep=True)",
        "ret_hole = [ret_base]",
    ),
)
def test_an_expression_that_may_build_something_is_not_an_identity(
    assignment: str,
) -> None:
    plan = _plan(_operation("op_base"), _operation("op_hole"))
    source = f"ret_base = object()\n{assignment}\nresult = ret_hole\n"

    assert check_program(source, plan).identity_operations == ()


def test_a_result_nothing_reads_never_reaches_the_exported_solid() -> None:
    plan = _plan(_operation("op_base"), _operation("op_pad"), _operation("op_hole"))
    source = """\
ret_base = object()
ret_pad = ret_base.cut(object())
ret_hole = ret_base.cut(object())
result = ret_hole
"""

    check = check_program(source, plan)

    assert check.unconsumed_operations == ("ret_pad",)
    assert not check.sound


def test_the_last_result_is_consumed_by_the_exported_solid() -> None:
    plan = _plan(_operation("op_base"))
    source = """\
ret_base = object().cut(object())
result = ret_base
"""

    assert check_program(source, plan).unconsumed_operations == ()


def test_syntax_errors_keep_the_supplied_filename() -> None:
    plan = _plan(_operation("op_base"))

    with pytest.raises(SyntaxError) as raised:
        check_program("result = (", plan, filename="candidate.py")

    assert raised.value.filename == "candidate.py"
