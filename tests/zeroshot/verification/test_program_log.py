import json
from pathlib import Path

from zeroshot.pipeline.verification.program_log import (
    BooleanCall,
    ProgramLog,
    describe_ineffective_booleans,
    describe_inside_out,
    read_program_log,
)

_ENTRY = {
    "operation": "cut",
    "line": 12,
    "caller_line": 30,
    "volume_before": 1000.0,
    "volume_after": 1000.0,
    "body_box": [-5.0, -5.0, -5.0, 5.0, 5.0, 5.0],
    "tool_box": None,
}


def test_a_log_reads_back(tmp_path: Path) -> None:
    log = tmp_path / "_program_log.json"
    log.write_text(
        json.dumps({"booleans": [_ENTRY], "result_volumes": [992.0, None]}),
        encoding="utf-8",
    )

    assert read_program_log(log) == ProgramLog(
        booleans=(
            BooleanCall(
                operation="cut",
                line=12,
                caller_line=30,
                volume_before=1000.0,
                volume_after=1000.0,
                body_box=(-5.0, -5.0, -5.0, 5.0, 5.0, 5.0),
                tool_box=None,
            ),
        ),
        result_volumes=(992.0, None),
    )


def test_no_log_is_not_an_empty_log(tmp_path: Path) -> None:
    assert read_program_log(tmp_path / "_program_log.json") is None


def test_an_unreadable_log_is_no_log(tmp_path: Path) -> None:
    whole = json.dumps({"booleans": [_ENTRY], "result_volumes": None})
    truncated = tmp_path / "truncated.json"
    truncated.write_text(whole[:40], encoding="utf-8")
    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text(
        json.dumps({"booleans": [{"operation": "cut"}], "result_volumes": None}),
        encoding="utf-8",
    )

    assert read_program_log(truncated) is None
    assert read_program_log(incomplete) is None


def test_a_symlinked_log_is_not_followed(tmp_path: Path) -> None:
    target = tmp_path / "elsewhere.json"
    target.write_text(
        json.dumps({"booleans": [_ENTRY], "result_volumes": None}), encoding="utf-8"
    )
    log = tmp_path / "_program_log.json"
    log.symlink_to(target)

    assert read_program_log(log) is None


_BODY = (-50.0, -19.285, 0.0, 50.0, 19.285, 65.57)


def _call(
    operation: str,
    before: float | None,
    after: float | None,
    tool: tuple[float, ...] | None,
    *,
    line: int = 50,
    caller_line: int | None = None,
    body: tuple[float, ...] | None = _BODY,
) -> BooleanCall:
    return BooleanCall(
        operation=operation,
        line=line,
        caller_line=line if caller_line is None else caller_line,
        volume_before=before,
        volume_after=after,
        body_box=body,
        tool_box=tool,
    )


def test_a_cut_outside_the_body_shows_both_boxes() -> None:
    calls = [_call("cut", 1000.0, 1000.0, (78.815, -3.085, -1.0, 84.985, 3.085, 28.9))]

    (entry,) = describe_ineffective_booleans(calls, "model.py")

    assert entry == (
        "model.py:50 cut removed no material.\n"
        "  body   X -50.000..50.000, Y -19.285..19.285, Z 0.000..65.570 mm\n"
        "  cutter X 78.815..84.985, Y -3.085..3.085, Z -1.000..28.900 mm"
    )


def test_a_cut_in_an_empty_region_inside_the_body_says_so() -> None:
    calls = [_call("cut", 1000.0, 1000.0, (7.14, 7.34, 54.0, 9.86, 9.06, 55.5))]

    (entry,) = describe_ineffective_booleans(calls, "model.py")

    assert entry == (
        "model.py:50 cut removed no material: the cutter's region holds no "
        "material at this point.\n"
        "  body   X -50.000..50.000, Y -19.285..19.285, Z 0.000..65.570 mm\n"
        "  cutter X 7.140..9.860, Y 7.340..9.060, Z 54.000..55.500 mm"
    )


def test_a_union_inside_material_says_so() -> None:
    calls = [_call("fuse", 1000.0, 1000.0, (0.0, 0.0, 0.0, 2.0, 2.0, 2.0))]

    (entry,) = describe_ineffective_booleans(calls, "model.py")

    assert entry == (
        "model.py:50 union added no material: the tool's region is already "
        "material at this point.\n"
        "  body X -50.000..50.000, Y -19.285..19.285, Z 0.000..65.570 mm\n"
        "  tool X 0.000..2.000, Y 0.000..2.000, Z 0.000..2.000 mm"
    )


def test_an_intersect_with_a_disjoint_tool_gives_no_reason() -> None:
    calls = [_call("intersect", 1000.0, 0.0, (60.0, 0.0, 0.0, 62.0, 2.0, 2.0))]

    (entry,) = describe_ineffective_booleans(calls, "model.py")

    assert entry == (
        "model.py:50 intersect left no material.\n"
        "  body X -50.000..50.000, Y -19.285..19.285, Z 0.000..65.570 mm\n"
        "  tool X 60.000..62.000, Y 0.000..2.000, Z 0.000..2.000 mm"
    )


def test_an_intersect_with_overlapping_boxes_and_no_common_material() -> None:
    calls = [_call("intersect", 1000.0, 0.0, (0.0, 0.0, 0.0, 2.0, 2.0, 2.0))]

    (entry,) = describe_ineffective_booleans(calls, "model.py")

    assert entry.startswith(
        "model.py:50 intersect left no material: the body and the tool share no "
        "material.\n  body X"
    )


def test_an_intersect_that_keeps_the_whole_body() -> None:
    calls = [_call("intersect", 1000.0, 1000.0, (-60.0, -20.0, -1.0, 60.0, 20.0, 70.0))]

    (entry,) = describe_ineffective_booleans(calls, "model.py")

    assert entry == (
        "model.py:50 intersect removed no material: the body lies inside the tool.\n"
        "  body X -50.000..50.000, Y -19.285..19.285, Z 0.000..65.570 mm\n"
        "  tool X -60.000..60.000, Y -20.000..20.000, Z -1.000..70.000 mm"
    )


def test_a_cut_that_removes_the_whole_body() -> None:
    calls = [_call("cut", 1000.0, 0.0, (-60.0, -20.0, -1.0, 60.0, 20.0, 70.0))]

    (entry,) = describe_ineffective_booleans(calls, "model.py")

    assert entry == (
        "model.py:50 cut removed all material: the cutter covers the whole body.\n"
        "  body   X -50.000..50.000, Y -19.285..19.285, Z 0.000..65.570 mm\n"
        "  cutter X -60.000..60.000, Y -20.000..20.000, Z -1.000..70.000 mm"
    )


def test_an_empty_tool_or_body_is_named_and_the_other_box_shown() -> None:
    calls = [
        _call("cut", 1000.0, 1000.0, None, line=10),
        _call("fuse", 0.0, 0.0, (0.0, 0.0, 0.0, 1.0, 1.0, 1.0), line=20, body=None),
    ]

    assert describe_ineffective_booleans(calls, "model.py") == (
        (
            "model.py:10 cut removed no material: the cutter is empty.\n"
            "  body X -50.000..50.000, Y -19.285..19.285, Z 0.000..65.570 mm"
        ),
        (
            "model.py:20 union added no material: the body is empty.\n"
            "  tool X 0.000..1.000, Y 0.000..1.000, Z 0.000..1.000 mm"
        ),
    )


def test_an_unbounded_box_is_written_in_scientific_notation() -> None:
    calls = [_call("fuse", 1000.0, 1000.0, (-1e100, 29.0, -1e100, 1e100, 29.0, 1e100))]

    (entry,) = describe_ineffective_booleans(calls, "model.py")

    assert entry == (
        "model.py:50 union added no material.\n"
        "  body X -50.000..50.000, Y -19.285..19.285, Z 0.000..65.570 mm\n"
        "  tool X -1.000e+100..1.000e+100, Y 29.000..29.000, "
        "Z -1.000e+100..1.000e+100 mm"
    )


def test_calls_that_change_the_volume_are_not_named() -> None:
    tool = (0.0, 0.0, 0.0, 2.0, 2.0, 2.0)
    calls = [
        _call("cut", 1000.0, 992.0, tool),
        _call("fuse", 1000.0, 1008.0, tool),
        _call("intersect", 1000.0, 8.0, tool),
    ]

    assert describe_ineffective_booleans(calls, "model.py") == ()


def test_a_change_within_a_millionth_of_the_volume_counts_as_none() -> None:
    tool = (0.0, 0.0, 0.0, 2.0, 2.0, 2.0)

    assert describe_ineffective_booleans(
        [_call("cut", 1e6, 1e6 - 0.5, tool)], "model.py"
    )
    assert not describe_ineffective_booleans(
        [_call("cut", 1e6, 1e6 - 2.0, tool)], "model.py"
    )


def test_unmeasured_and_inside_out_volumes_are_not_judged() -> None:
    tool = (0.0, 0.0, 0.0, 2.0, 2.0, 2.0)
    calls = [
        _call("cut", None, 1000.0, tool),
        _call("cut", 140.0, -28575.0, tool),
        _call("cut", -5874.0, -5874.0, tool),
    ]

    assert describe_ineffective_booleans(calls, "model.py") == ()


def test_a_tool_touching_the_body_counts_as_outside() -> None:
    calls = [_call("cut", 1000.0, 1000.0, (50.0, -1.0, 0.0, 52.0, 1.0, 2.0))]

    (entry,) = describe_ineffective_booleans(calls, "model.py")

    assert entry.startswith("model.py:50 cut removed no material.\n  body")


def test_repeated_calls_of_one_line_are_counted_and_the_first_is_shown() -> None:
    first = (60.0, 0.0, 0.0, 62.0, 2.0, 2.0)
    calls = [
        _call("cut", 1000.0, 1000.0, first),
        _call("cut", 1000.0, 992.0, (0.0, 0.0, 0.0, 2.0, 2.0, 2.0)),
        _call("cut", 992.0, 992.0, (70.0, 0.0, 0.0, 72.0, 2.0, 2.0)),
    ]

    (entry,) = describe_ineffective_booleans(calls, "model.py")

    assert entry == (
        "model.py:50 cut removed no material in 2 of 3 calls. The first:\n"
        "  body   X -50.000..50.000, Y -19.285..19.285, Z 0.000..65.570 mm\n"
        "  cutter X 60.000..62.000, Y 0.000..2.000, Z 0.000..2.000 mm"
    )


def test_a_call_inside_a_helper_is_named_by_each_line_that_called_it() -> None:
    tool = (60.0, 0.0, 0.0, 62.0, 2.0, 2.0)
    calls = [
        _call("cut", 1000.0, 1000.0, tool, line=35, caller_line=41),
        _call("cut", 1000.0, 1000.0, tool, line=35, caller_line=48),
    ]

    entries = describe_ineffective_booleans(calls, "model.py")

    assert [entry.split(" cut ")[0] for entry in entries] == [
        "model.py:35 (called from model.py:41)",
        "model.py:35 (called from model.py:48)",
    ]


def test_entries_past_the_limit_are_named_by_line() -> None:
    tool = (60.0, 0.0, 0.0, 62.0, 2.0, 2.0)
    calls = [_call("cut", 1000.0, 1000.0, tool, line=line) for line in range(1, 7)]

    entries = describe_ineffective_booleans(calls, "model.py")

    assert len(entries) == 5
    assert entries[-1] == "Not shown: model.py:5, model.py:6."


def test_a_call_that_returns_an_inside_out_solid_is_named() -> None:
    calls = [
        _call("cut", 1000.0, 992.0, (0.0, 0.0, 0.0, 2.0, 2.0, 2.0), line=40),
        _call(
            "cut", 140.6, -28575.5, (37.664, -20.0, -387.961, 112.664, 0.0, -312.961)
        ),
        _call("cut", -28575.5, -28575.5, (0.0, 0.0, 0.0, 2.0, 2.0, 2.0), line=60),
    ]

    assert describe_inside_out(ProgramLog(tuple(calls), (-28575.5,)), "model.py") == (
        "model.py:50 cut returned an inside-out solid (one whose faces point "
        "inward): its volume is -28575.500 mm³, against 140.600 mm³ before the "
        "call.\n"
        "  body   X -50.000..50.000, Y -19.285..19.285, Z 0.000..65.570 mm\n"
        "  cutter X 37.664..112.664, Y -20.000..0.000, Z -387.961..-312.961 mm"
    )


def test_a_call_given_an_inside_out_body_is_named() -> None:
    calls = [_call("fuse", -50494.877, 5075.44, (0.0, 0.0, 0.0, 2.0, 2.0, 2.0))]

    message = describe_inside_out(ProgramLog(tuple(calls), None), "model.py")

    assert message is not None
    assert message.startswith(
        "model.py:50 union was given an inside-out body (one whose faces point "
        "inward): its volume is -50494.877 mm³.\n"
    )


def test_solids_with_positive_or_unmeasured_volume_are_not_inside_out() -> None:
    tool = (0.0, 0.0, 0.0, 2.0, 2.0, 2.0)
    calls = [
        _call("cut", 1000.0, 1000.0, tool),
        _call("intersect", 1000.0, -1e-9, tool),
        _call("cut", None, -5.0, tool),
    ]

    assert describe_inside_out(ProgramLog(tuple(calls), (1000.0,)), "model.py") is None


def test_an_inside_out_result_is_named_when_no_boolean_met_it() -> None:
    log = ProgramLog((), (-5874.195,))

    assert describe_inside_out(log, "model.py") == (
        "`result` is an inside-out solid (one whose faces point inward): its "
        "volume is -5874.195 mm³."
    )


def test_inside_out_solids_among_several_in_the_result_are_counted() -> None:
    log = ProgramLog((), (928.0, -1.25, 181.2, None))

    assert describe_inside_out(log, "model.py") == (
        "1 of the 3 solids in `result` is inside out (faces pointing inward): "
        "volume -1.250 mm³."
    )


def test_a_result_with_no_measured_solid_is_not_inside_out() -> None:
    assert describe_inside_out(ProgramLog((), None), "model.py") is None
    assert describe_inside_out(ProgramLog((), ()), "model.py") is None
    assert describe_inside_out(ProgramLog((), (None,)), "model.py") is None
