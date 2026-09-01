"""Runs the coder's program inside the sandbox and exports what it built.

Staged into the workdir beside the program, because the sandbox binds nothing
of this repository: code that has to run in there arrives as a file. Called as
`python _run_program.py <program> [<ret_name> ...]`.

The host half is `run_cadquery.py`, which stages this, names the outputs to
keep, and reads back what lands in the workdir.
"""

import sys
import traceback
from collections.abc import Sequence
from pathlib import Path

RESULT_STEP = "output.step"
INTERMEDIATE_RETURNS_DIR = "intermediate_returns"
ERROR_SUFFIX = ".err"


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


def _run(program: Path) -> dict[str, object]:
    """The globals the program leaves behind, as running it directly would."""
    code = compile(program.read_text(encoding="utf-8"), str(program), "exec")
    namespace: dict[str, object] = {
        "__name__": "__main__",
        "__file__": str(program),
    }
    exec(code, namespace)  # noqa: S102 - the program is the input, and the sandbox is the guard
    return namespace


def _keep(namespace: dict[str, object], names: Sequence[str]) -> None:
    """Export each named output, recording what will not export instead of raising.

    Every `ret_*` still holds what it was last given, so this runs once at the
    end rather than stepping through the program.
    """
    directory = Path(INTERMEDIATE_RETURNS_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        try:
            _export(namespace.get(name), directory / f"{name}.step")
        except Exception as error:  # noqa: BLE001 - a kept output must not fail the build
            (directory / f"{name}{ERROR_SUFFIX}").write_text(
                f"{type(error).__name__}: {error}", encoding="utf-8"
            )


def main(argv: Sequence[str]) -> int:
    program, names = Path(argv[1]), argv[2:]
    try:
        namespace = _run(program)
    except Exception as error:  # noqa: BLE001 - reported below as an exit code
        return _report(error)

    if names:
        _keep(namespace, names)

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
