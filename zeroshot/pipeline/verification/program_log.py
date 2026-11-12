"""What the runner measured while a program ran.

`_run_program.py` logs each cut, fuse and intersect the program called and the
solids of its `result`. This reads the log back on the host, names the
booleans that changed nothing and finds the first inside-out solid.
"""

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# A volume change within this fraction of the body's volume, or an overlap of
# boxes within this fraction of the body's largest extent, counts as none; a
# volume below minus this fraction of the larger volume counts as negative.
_RELATIVE_TOLERANCE = 1e-6
_MAX_ENTRIES = 4
# A coordinate of this size or more, in mm, is written in scientific notation.
_FIXED_NOTATION_BELOW = 1e9
_NAME = {"cut": "cut", "fuse": "union", "intersect": "intersect"}
_TOOL = {"cut": "cutter", "fuse": "tool", "intersect": "tool"}
# Why a call whose tool overlaps the body's box had no effect.
_OVERLAPPING = {
    "cut removed no material": "the cutter's region holds no material at this point",
    "cut removed all material": "the cutter covers the whole body",
    "union added no material": "the tool's region is already material at this point",
    "intersect left no material": "the body and the tool share no material",
    "intersect removed no material": "the body lies inside the tool",
}


@dataclass(frozen=True)
class BooleanCall:
    """One boolean as the program called it.

    A value that could not be measured is None. A box is (xmin, ymin, zmin,
    xmax, ymax, zmax) in model mm.
    """

    operation: str  # "cut", "fuse" or "intersect", as CadQuery names the method
    line: int  # the innermost line of the program on the stack
    caller_line: int  # the outermost one; differs for a call inside a helper
    volume_before: float | None
    volume_after: float | None
    body_box: tuple[float, ...] | None
    tool_box: tuple[float, ...] | None


@dataclass(frozen=True)
class ProgramLog:
    booleans: tuple[BooleanCall, ...]
    # The volume of each solid in `result`; None when `result` was not measured.
    result_volumes: tuple[float | None, ...] | None


def read_program_log(path: Path) -> ProgramLog | None:
    """None when the runner left no readable log."""
    if path.is_symlink() or not path.is_file():
        return None
    try:
        log = json.loads(path.read_text(encoding="utf-8"))
        volumes = log["result_volumes"]
        return ProgramLog(
            booleans=tuple(
                BooleanCall(
                    operation=entry["operation"],
                    line=entry["line"],
                    caller_line=entry["caller_line"],
                    volume_before=entry["volume_before"],
                    volume_after=entry["volume_after"],
                    body_box=_box(entry["body_box"]),
                    tool_box=_box(entry["tool_box"]),
                )
                for entry in log["booleans"]
            ),
            result_volumes=None if volumes is None else tuple(volumes),
        )
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _box(values: list[float] | None) -> tuple[float, ...] | None:
    return None if values is None else tuple(values)


def describe_ineffective_booleans(
    calls: Sequence[BooleanCall], filename: str
) -> tuple[str, ...]:
    """Name each call that changed no volume, or left none.

    The calls of one line, reached from one caller line, with one finding make
    one entry: it counts them and shows the boxes of the first. Entries past
    `_MAX_ENTRIES` are named by their lines only.
    """
    totals = Counter((call.line, call.caller_line, call.operation) for call in calls)
    findings: dict[tuple[int, int, str, str | None], list[BooleanCall]] = {}
    for call in calls:
        effect = _effect(call)
        if effect is not None:
            key = (call.line, call.caller_line, effect, _reason(call, effect))
            findings.setdefault(key, []).append(call)
    entries = [
        _entry(found, effect, totals[(line, caller, found[0].operation)], filename)
        for (line, caller, effect, _), found in findings.items()
    ]
    if len(entries) <= _MAX_ENTRIES:
        return tuple(entries)
    hidden = dict.fromkeys(
        _where(line, caller, filename)
        for line, caller, *_ in list(findings)[_MAX_ENTRIES:]
    )
    return (*entries[:_MAX_ENTRIES], f"Not shown: {', '.join(hidden)}.")


def describe_inside_out(log: ProgramLog, filename: str) -> str | None:
    """Name the first boolean that was given or returned an inside-out solid.

    Such a solid's faces point inward, so its volume is negative. The calls
    after it carry it on, so they go unnamed. With no such boolean, `result`
    itself is checked: a solid can turn inside out outside any boolean.
    """
    for call in log.booleans:
        before, after = call.volume_before, call.volume_after
        if before is None or after is None:
            continue
        floor = _negative_below(before, after)
        where = (
            f"{_where(call.line, call.caller_line, filename)} {_NAME[call.operation]}"
        )
        if before < floor:
            head = (
                f"{where} was given an inside-out body (one whose faces point "
                f"inward): its volume is {before:.3f} mm³."
            )
        elif after < floor:
            head = (
                f"{where} returned an inside-out solid (one whose faces point "
                f"inward): its volume is {after:.3f} mm³, against {before:.3f} mm³ "
                "before the call."
            )
        else:
            continue
        return "\n".join([head, *_boxes(call)])
    measured = [volume for volume in log.result_volumes or () if volume is not None]
    negative = [volume for volume in measured if volume < _negative_below(*measured)]
    if not negative:
        return None
    if len(measured) == 1:
        return (
            "`result` is an inside-out solid (one whose faces point inward): its "
            f"volume is {negative[0]:.3f} mm³."
        )
    many = len(negative) > 1
    return (
        f"{len(negative)} of the {len(measured)} solids in `result` "
        f"{'are' if many else 'is'} inside out (faces pointing inward): "
        f"volume{'s' if many else ''} "
        f"{', '.join(f'{volume:.3f}' for volume in negative)} mm³."
    )


def _negative_below(*volumes: float) -> float:
    """A volume below this is negative beyond numerical noise."""
    return -_RELATIVE_TOLERANCE * max(map(abs, volumes))


def _effect(call: BooleanCall) -> str | None:
    before, after = call.volume_before, call.volume_after
    if before is None or after is None:
        return None
    # An inside-out solid's volume changes do not measure material.
    if min(before, after) < _negative_below(before, after):
        return None
    tolerance = _RELATIVE_TOLERANCE * abs(before)
    if call.operation == "cut":
        if before - after <= tolerance:
            return "cut removed no material"
        if after <= tolerance:
            return "cut removed all material"
    elif call.operation == "fuse":
        if after - before <= tolerance:
            return "union added no material"
    elif call.operation == "intersect":
        if after <= tolerance:
            return "intersect left no material"
        if before - after <= tolerance:
            return "intersect removed no material"
    return None


def _reason(call: BooleanCall, effect: str) -> str | None:
    """Why the call had no effect, unless its boxes lying apart show it."""
    if call.tool_box is None:
        return f"the {_TOOL[call.operation]} is empty"
    if call.body_box is None:
        return "the body is empty"
    if _apart(call.body_box, call.tool_box):
        return None
    return _OVERLAPPING[effect]


def _apart(body: tuple[float, ...], tool: tuple[float, ...]) -> bool:
    """Whether the boxes are disjoint or only touch."""
    size = max(high - low for low, high in zip(body[:3], body[3:], strict=True))
    return any(
        min(body[axis + 3], tool[axis + 3]) - max(body[axis], tool[axis])
        <= _RELATIVE_TOLERANCE * size
        for axis in range(3)
    )


def _entry(found: list[BooleanCall], effect: str, total: int, filename: str) -> str:
    first = found[0]
    reason = _reason(first, effect)
    boxes = _boxes(first)
    head = f"{_where(first.line, first.caller_line, filename)} {effect}"
    if total > 1:
        head += f" in {len(found)} of {total} calls"
    head += f": {reason}." if reason else "."
    if boxes and len(found) > 1:
        head += " The first:"
    return "\n".join([head, *boxes])


def _where(line: int, caller_line: int, filename: str) -> str:
    called = f" (called from {filename}:{caller_line})" if caller_line != line else ""
    return f"{filename}:{line}{called}"


def _boxes(call: BooleanCall) -> list[str]:
    """The measured boxes of the body and the tool, as lines under an entry."""
    labelled = [
        (label, box)
        for label, box in (
            ("body", call.body_box),
            (_TOOL[call.operation], call.tool_box),
        )
        if box is not None
    ]
    width = max((len(label) for label, _ in labelled), default=0)
    return [f"  {label:<{width}} {_describe(box)}" for label, box in labelled]


def _describe(box: tuple[float, ...]) -> str:
    return (
        ", ".join(
            f"{axis} {_coordinate(low)}..{_coordinate(high)}"
            for axis, low, high in zip("XYZ", box[:3], box[3:], strict=True)
        )
        + " mm"
    )


def _coordinate(value: float) -> str:
    # An unbounded shape's box reaches 1e100, which fixed notation writes out
    # digit by digit.
    if abs(value) >= _FIXED_NOTATION_BELOW:
        return f"{value:.3e}"
    # + 0.0 turns a rounded -0.0 into 0.0
    return f"{round(value, 3) + 0.0:.3f}"
