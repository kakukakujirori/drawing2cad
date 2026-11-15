"""Runs the coder's program inside the sandbox and exports what it built.

Staged into the workdir beside the program, because the sandbox binds nothing
of this repository: code that has to run in there arrives as a file. Called as
`python _run_program.py <program>`.

The host half is `run_cadquery.py`, which stages this and reads back what
lands in the workdir.
"""

import json
import math
import sys
import traceback
from collections import deque
from collections.abc import Callable, Sequence
from contextlib import suppress
from functools import wraps
from pathlib import Path
from typing import Any

RESULT_STEP = "output.step"
# Written once the program has run, also when it raised: the calls it made (see
# `_record_calls`), whether it ran to its end, and the volume of each solid in
# `result`.
PROGRAM_LOG = "_program_log.json"


def _report(error: BaseException) -> int:
    """Print the traceback without this runner's frames.

    They name a file the coder never wrote: the entry before the program, and
    the call recorder between the program and CadQuery.
    """
    report = traceback.TracebackException.from_exception(error, compact=True)
    _drop_own_frames(report)
    report.print(file=sys.stderr)
    return 1


def _drop_own_frames(report: traceback.TracebackException) -> None:
    report.stack = traceback.StackSummary.from_list(
        [frame for frame in report.stack if frame.filename != __file__]
    )
    for chained in (report.__cause__, report.__context__, *(report.exceptions or ())):
        if chained is not None:
            _drop_own_frames(chained)


def _export(value: object, path: Path | str) -> None:
    import cadquery as cq

    cq.exporters.export(value, str(path), exportType="STEP")


def _record_calls(program: Path, log: list[dict[str, Any]]) -> Callable[[], None]:
    """Patch CadQuery so that each call the program makes to the methods below
    appends an entry to `log`; return the function that removes the patches.

    An entry names the method ("operation") and the program's lines on the
    stack ("line" the innermost, "caller_line" the outermost), and holds the
    solids, as `_measure_solids` gives them, of
    - "body": the solid a boolean or a modifier works on, which for a Workplane
      method is its context solid; None for a builder, which makes a new solid;
    - "tools": each argument of a boolean that is a shape or a Workplane;
    - "output": what the call returned. None when it raised, and "error" is
      then "<exception type>: <message>".
    Only the outermost of nested calls is logged, as the one call the program
    made. Logging never changes what a call returns or raises.
    """
    import cadquery as cq
    from cadquery.occ_impl.shapes import Mixin3D

    # Methods that combine a solid with tools.
    booleans = {
        cq.Shape: ("cut", "fuse", "intersect"),
        cq.Compound: ("cut", "fuse", "intersect"),
        # Their own checks can raise before the shape's method is reached.
        cq.Workplane: ("union", "cut", "intersect"),
    }
    # Methods that change a solid by themselves.
    modifiers = {Mixin3D: ("fillet", "chamfer"), cq.Workplane: ("fillet", "chamfer")}
    # Methods that make a new solid.
    builders = {
        # They return the solid before it is combined with the context solid.
        cq.Workplane: ("_extrude", "_revolve", "_sweep"),
        # The function `Workplane.loft` builds with.
        sys.modules[cq.Workplane.__module__]: ("loft",),
        cq.Solid: ("extrudeLinear", "revolve", "sweep", "makeLoft"),
    }
    patched = [
        (owner, name, kind)
        for kind, methods in (
            ("boolean", booleans),
            ("modifier", modifiers),
            ("builder", builders),
        )
        for owner, names in methods.items()
        for name in names
    ]
    originals = {(owner, name): vars(owner)[name] for owner, name, _ in patched}
    filename = str(program)
    # The latest measurements: a chain gives each call's output to the next.
    recent: deque[tuple[Any, dict[str, Any]]] = deque(maxlen=16)
    depth = 0  # how many logged calls are running

    def logged(name: str, kind: str, original: Any) -> Any:
        """`original`, made to log each call the program makes to it."""
        is_classmethod = isinstance(original, classmethod)
        function = original.__func__ if is_classmethod else original

        @wraps(function)
        def call(*args: Any, **kwargs: Any) -> Any:
            nonlocal depth
            # Not logged when another logged call makes it, or not the program.
            if depth or not (lines := _program_lines(filename)):
                return function(*args, **kwargs)
            entry: dict[str, Any] = {
                "operation": name.lstrip("_"),
                "line": lines[0],
                "caller_line": lines[-1],
                "body": None,
                "tools": [],
                "output": None,
                "error": None,
            }
            # In the log before anything is measured: a failed measurement
            # loses only its own value.
            log.append(entry)
            depth += 1
            try:
                if kind == "boolean":
                    entry["tools"] = [
                        _measure_solids(arg, recent)
                        for arg in (*args[1:], *kwargs.values())
                        if isinstance(arg, cq.Shape | cq.Workplane)
                    ]
                if kind != "builder":
                    with suppress(Exception):
                        body = args[0]
                        if isinstance(body, cq.Workplane):
                            body = body._findType((cq.Solid,), True, True)
                        # A Workplane that finds no context solid has none.
                        entry["body"] = (
                            [] if body is None else _measure_solids(body, recent)
                        )
                output = function(*args, **kwargs)
            except BaseException as error:
                entry["error"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                depth -= 1
            entry["output"] = _measure_solids(output, recent)
            return output

        return classmethod(call) if is_classmethod else call

    for owner, name, kind in patched:
        setattr(owner, name, logged(name, kind, originals[owner, name]))

    def undo() -> None:
        for (owner, name), original in originals.items():
            setattr(owner, name, original)

    return undo


def _program_lines(filename: str) -> list[int]:
    """The program's lines on the stack, innermost first."""
    lines = []
    frame = sys._getframe()
    while frame is not None:
        if frame.f_code.co_filename == filename:
            lines.append(frame.f_lineno)
        frame = frame.f_back
    return lines


def _volume(shape: Any) -> float | None:
    with suppress(Exception):
        volume = float(shape.Volume())
        if math.isfinite(volume):
            return volume
    return None


def _box(shape: Any) -> list[float] | None:
    """(xmin, ymin, zmin, xmax, ymax, zmax); an empty shape has none."""
    with suppress(Exception):
        box = shape.BoundingBox()
        return [box.xmin, box.ymin, box.zmin, box.xmax, box.ymax, box.zmax]
    return None


def _measure_solids(
    value: Any, recent: deque[tuple[Any, dict[str, Any]]]
) -> list[dict[str, Any]] | None:
    """Measure each solid of `value`, a shape or a Workplane's stack of shapes.

    Returns one {"valid", "volume", "box"} per solid: OCCT's validity check,
    the signed volume (negative for an inside-out solid) and the box (xmin,
    ymin, zmin, xmax, ymax, zmax), each None where it cannot be measured. The
    list is empty when `value` holds no solid, and None when its solids cannot
    be listed. A solid found in `recent`, the latest (solid, measurement)
    pairs, in the same orientation is not measured again.
    """
    import cadquery as cq

    try:
        shapes = value.objects if isinstance(value, cq.Workplane) else [value]
        solids = [
            solid
            for shape in shapes
            if isinstance(shape, cq.Shape)
            for solid in shape.Solids()
        ]
        measurements = []
        for solid in solids:
            measurement = next(
                (known for seen, known in recent if seen.IsEqual(solid.wrapped)),
                None,
            )
            if measurement is None:
                valid = None
                with suppress(Exception):
                    valid = bool(solid.isValid())
                measurement = {
                    "valid": valid,
                    "volume": _volume(solid),
                    "box": _box(solid),
                }
                recent.append((solid.wrapped, measurement))
            measurements.append(measurement)
        return measurements
    except Exception:  # noqa: BLE001 - not measured
        return None


def _solid_volumes(value: object) -> list[float | None] | None:
    """The volume of each solid in `result`, made into one shape as the export does.

    A value that iterating would use up stays unmeasured: the export needs it.
    """
    import cadquery as cq
    from cadquery.occ_impl.shapes import compound

    with suppress(Exception):
        if isinstance(value, cq.Shape):
            shape = value
        elif isinstance(value, cq.Workplane | list | tuple):
            shape = compound(*value)
        else:
            return None
        return [_volume(solid) for solid in shape.Solids()]
    return None


def main(argv: Sequence[str]) -> int:
    program = Path(argv[1])
    namespace: dict[str, object] = {"__name__": "__main__", "__file__": str(program)}
    # Compiled before logging starts: a program that does not compile never
    # imports CadQuery, so its report carries none of CadQuery's import output.
    try:
        code = compile(program.read_text(encoding="utf-8"), str(program), "exec")
    except Exception as error:  # noqa: BLE001 - reported below as an exit code
        return _report(error)
    calls: list[dict[str, Any]] = []
    try:
        undo = _record_calls(program, calls)
    except Exception:  # noqa: BLE001 - the program then runs unlogged
        undo = None
    completed = False
    try:
        # The program leaves `result` in `namespace`.
        exec(code, namespace)  # noqa: S102 - the program is the input, and the sandbox is the guard
        completed = True
    except Exception as error:  # noqa: BLE001 - reported below as an exit code
        return _report(error)
    finally:
        if undo is not None:
            undo()
            log = {
                "calls": calls,
                "completed": completed,
                "result_volumes": (
                    _solid_volumes(namespace.get("result")) if completed else None
                ),
            }
            with suppress(Exception):
                Path(PROGRAM_LOG).write_text(json.dumps(log), encoding="utf-8")

    if "result" not in namespace:
        print("`result` not defined", file=sys.stderr)
        return 1
    try:
        _export(namespace["result"], RESULT_STEP)
    except Exception as error:  # noqa: BLE001 - reported below as an exit code
        print("Failed to export `result` to STEP:", file=sys.stderr)
        return _report(error)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
