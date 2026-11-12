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
from collections.abc import Callable, Sequence
from contextlib import suppress
from functools import wraps
from pathlib import Path
from typing import Any

RESULT_STEP = "output.step"
# Written when the program ran to its end and left `result`: the booleans it
# called (see `_record_booleans`) and the volume of each solid in `result`.
PROGRAM_LOG = "_program_log.json"
_BOOLEANS = ("cut", "fuse", "intersect")


def _report(error: BaseException) -> int:
    """Print the traceback without this runner's frames.

    They name a file the coder never wrote: the entry before the program, and
    the boolean recorder between the program and CadQuery.
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


def _record_booleans(program: Path, log: list[dict[str, Any]]) -> Callable[[], None]:
    """Log each cut, fuse and intersect the program calls; return the undo.

    Only the outermost of nested calls is logged, as the one call the program
    made. A call needs a tool shape and a solid to be logged: the fuse that
    merges a compound's own parts, or a sketch's boolean on faces, has no
    volume change to compare. Logging never changes what a call returns or
    raises.
    """
    import cadquery as cq

    filename = str(program)
    originals = {
        (cls, name): cls.__dict__[name]
        for cls in (cq.Shape, cq.Compound, cq.Solid)
        for name in _BOOLEANS
        if name in cls.__dict__
    }
    depth = 0

    def logged(name: str, original: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(original)
        def call(self: Any, *tools: Any, **kwargs: Any) -> Any:
            nonlocal depth
            depth += 1
            try:
                result = original(self, *tools, **kwargs)
            finally:
                depth -= 1
            if depth == 0:
                with suppress(Exception):
                    shapes = [tool for tool in tools if isinstance(tool, cq.Shape)]
                    lines = _program_lines(filename) if shapes else []
                    if lines and any(shape.Solids() for shape in (self, *shapes)):
                        log.append(
                            {
                                "operation": name,
                                "line": lines[0],
                                "caller_line": lines[-1],
                                "volume_before": _volume(self),
                                "volume_after": _volume(result),
                                "body_box": _box(self),
                                "tool_box": _box(cq.Compound.makeCompound(shapes)),
                            }
                        )
            return result

        return call

    for (cls, name), original in originals.items():
        setattr(cls, name, logged(name, original))

    def undo() -> None:
        for (cls, name), original in originals.items():
            setattr(cls, name, original)

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
    booleans: list[dict[str, Any]] = []
    try:
        undo = _record_booleans(program, booleans)
    except Exception:  # noqa: BLE001 - the program then runs unlogged
        undo = None
    try:
        # The program leaves `result` in `namespace`.
        exec(code, namespace)  # noqa: S102 - the program is the input, and the sandbox is the guard
    except Exception as error:  # noqa: BLE001 - reported below as an exit code
        return _report(error)
    finally:
        if undo is not None:
            undo()

    if "result" not in namespace:
        print("`result` not defined", file=sys.stderr)
        return 1
    if undo is not None:
        log = {
            "booleans": booleans,
            "result_volumes": _solid_volumes(namespace["result"]),
        }
        with suppress(Exception):
            Path(PROGRAM_LOG).write_text(json.dumps(log), encoding="utf-8")
    try:
        _export(namespace["result"], RESULT_STEP)
    except Exception as error:  # noqa: BLE001 - reported below as an exit code
        print("Failed to export `result` to STEP:", file=sys.stderr)
        return _report(error)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
