"""Runs the coder's program inside the sandbox and exports what it built.

Staged into the workdir beside the program, because the sandbox binds nothing
of this repository: code that has to run in there arrives as a file. Called as
`python _run_program.py <program>`.

The host half is `run_cadquery.py`, which stages this and reads back what
lands in the workdir.
"""

import sys
import traceback
from collections.abc import Sequence
from pathlib import Path

RESULT_STEP = "output.step"


def _report(error: BaseException) -> int:
    """Print the traceback, starting at the program's own first frame.

    The runner's frames name a file the coder never wrote, so they go.
    """
    frames = error.__traceback__
    while frames is not None and frames.tb_frame.f_code.co_filename == __file__:
        frames = frames.tb_next
    traceback.print_exception(type(error), error, frames, file=sys.stderr)
    return 1


def _export(value: object, path: Path | str) -> None:
    import cadquery as cq

    cq.exporters.export(value, str(path), exportType="STEP")


def _run(program: Path, namespace: dict[str, object]) -> None:
    """Run the program in `namespace`, where it leaves `result`."""
    code = compile(program.read_text(encoding="utf-8"), str(program), "exec")
    exec(code, namespace)  # noqa: S102 - the program is the input, and the sandbox is the guard


def main(argv: Sequence[str]) -> int:
    program = Path(argv[1])
    namespace: dict[str, object] = {"__name__": "__main__", "__file__": str(program)}
    try:
        _run(program, namespace)
    except Exception as error:  # noqa: BLE001 - reported below as an exit code
        return _report(error)

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
