"""Runs the coder's program inside the sandbox and exports what it built.

Staged into the workdir beside the program, because the sandbox binds nothing
of this repository: code that has to run in there arrives as a file. Called as
`python _run_program.py <program> [<ret_name> ...]`.

The host half is `run_cadquery.py`, which stages this, names the outputs to
keep, and reads back what lands in the workdir.
"""

import json
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path

RESULT_STEP = "output.step"
INTERMEDIATE_RETURNS_DIR = "intermediate_returns"
METADATA_SUFFIX = ".json"


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
    """Run the program in `namespace`, which keeps whatever it assigned before raising."""
    code = compile(program.read_text(encoding="utf-8"), str(program), "exec")
    exec(code, namespace)  # noqa: S102 - the program is the input, and the sandbox is the guard


def _validity(value: object) -> tuple[bool | None, str | None]:
    """Check the BRep before export, since a STEP export can repair invalid shapes."""
    import cadquery as cq

    try:
        shapes = value.vals() if isinstance(value, cq.Workplane) else [value]
        if not shapes:
            return None, "holds no shape"
        if strangers := [
            type(s).__name__ for s in shapes if not isinstance(s, cq.Shape)
        ]:
            return None, f"holds non-shape values: {', '.join(strangers)}"
        invalid = [i for i, s in enumerate(shapes, 1) if not s.isValid()]
    except Exception as error:  # noqa: BLE001 - unknown, never valid
        return None, f"{type(error).__name__}: {error}"
    if invalid:
        return (
            False,
            f"shape {', '.join(map(str, invalid))} of {len(shapes)} is invalid",
        )
    return True, None


def _keep_one(value: object, name: str, directory: Path) -> dict[str, object]:
    valid, validity_reason = _validity(value)
    metadata: dict[str, object] = {
        "valid": valid,
        "validity_reason": validity_reason,
        "export_error": None,
    }
    try:
        _export(value, directory / f"{name}.step")
    except Exception as error:  # noqa: BLE001 - a kept output must not fail the build
        metadata["export_error"] = f"{type(error).__name__}: {error}"
    return metadata


def _keep(namespace: dict[str, object], names: Sequence[str]) -> None:
    """Export each named output as the namespace holds it now, with its metadata.

    Runs once when the program ends or raises, so a `ret_*` reassigned later
    is kept as last assigned. One never assigned leaves no file.
    """
    directory = Path(INTERMEDIATE_RETURNS_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        if name not in namespace:
            continue
        metadata = _keep_one(namespace[name], name, directory)
        (directory / f"{name}{METADATA_SUFFIX}").write_text(
            json.dumps(metadata), encoding="utf-8"
        )


def main(argv: Sequence[str]) -> int:
    program, names = Path(argv[1]), argv[2:]
    namespace: dict[str, object] = {"__name__": "__main__", "__file__": str(program)}
    try:
        _run(program, namespace)
    except Exception as error:  # noqa: BLE001 - reported below as an exit code
        # Report first: a failure while keeping outputs must not hide this traceback.
        status = _report(error)
        if names:
            _keep(namespace, names)
        return status

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
