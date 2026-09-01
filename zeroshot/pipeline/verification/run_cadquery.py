import ast
import shutil
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from zeroshot.pipeline.sandbox import (
    SandboxRunner,
    SandboxStatus,
    SandboxWorkdir,
)
from zeroshot.pipeline.verification._run_program import (
    ERROR_SUFFIX,
    INTERMEDIATE_RETURNS_DIR,
)
from zeroshot.pipeline.verification.check_program import assigned_names
from zeroshot.pipeline.verification.shape_census import ShapeCensus, read_census

_RUNNER = "_run_program.py"


class ExecutionStatus(StrEnum):
    UNINITIALIZED = "UNINITIALIZED"
    VERIFIED = "VERIFIED"
    REJECTED = "REJECTED"  # syntax/validation error
    FAILED = "FAILED"  # execution error (no STEP generated)
    TIMEOUT = "TIMEOUT"  # CAD process timeout
    INFRA_ERROR = "INFRA_ERROR"  # sandbox infrastructure failure


@dataclass(frozen=True)
class IntermediateReturn:
    """One `ret_*` as it stood the moment the program assigned it."""

    name: str
    step_path: Path | None = None
    error: str | None = None
    census: ShapeCensus | None = None


@dataclass(frozen=True)
class CadQueryExecutionReport:
    source: str | None = None
    status: ExecutionStatus = ExecutionStatus.INFRA_ERROR
    executor_error: str | None = None
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    census: ShapeCensus | None = None
    # In program order, and empty unless `execute` was asked to keep them.
    intermediate_returns: tuple[IntermediateReturn, ...] = ()


def _rejection(error: SyntaxError | ValueError) -> str:
    """Say what is wrong with the source.

    A syntax error is given the offending line and the caret under it, which
    `str` leaves out.
    """
    if not isinstance(error, SyntaxError):
        return str(error)
    return "".join(traceback.format_exception_only(type(error), error)).strip()


def _returned_names(source: str, filename: str) -> list[str]:
    """List the `ret_*` names the program assigns, in the order it assigns them."""
    tree = ast.parse(source, filename=filename, mode="exec")
    names: list[str] = []
    for statement in tree.body:
        for name in sorted(assigned_names(statement)):
            if name.startswith("ret_") and name not in names:
                names.append(name)
    return names


def _read_returns(
    ret_names: Sequence[str],
    sandbox_temp_returns_dir: Path,
    host_dest_returns_dir: Path,
) -> tuple[IntermediateReturn, ...]:
    """Keep a STEP for each named output, or the reason the program wrote none.

    Empty when the program never got that far, which is every run that failed
    to build: what such a run needs said is in the traceback.
    """
    if not sandbox_temp_returns_dir.is_dir():
        return ()
    host_dest_returns_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[IntermediateReturn] = []
    for name in ret_names:
        built = sandbox_temp_returns_dir / f"{name}.step"
        if built.is_file():
            # One directory per return, laid out like the attempt that holds it.
            kept = host_dest_returns_dir / name / "output.step"
            kept.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(built, kept)
            outputs.append(
                IntermediateReturn(name, step_path=kept, census=read_census(kept))
            )
            continue
        reported = sandbox_temp_returns_dir / f"{name}{ERROR_SUFFIX}"
        outputs.append(
            IntermediateReturn(
                name,
                error=(
                    reported.read_text(encoding="utf-8")
                    if reported.is_file()
                    else "no step file was written"
                ),
            )
        )
    return tuple(outputs)


class CadQueryExecutor:
    def __init__(self, sandbox_runner: SandboxRunner) -> None:
        self.sandbox_runner = sandbox_runner

    def execute(
        self,
        model_path: Path,
        output_step_path: Path | None = None,
        intermediate_returns_dir: Path | None = None,
    ) -> CadQueryExecutionReport:
        """Run the program in the sandbox and report what it built.

        Both directories are host paths the caller keeps. The sandbox writes
        to `output.step` and `intermediate_returns/` in its own workdir, which
        is gone by the time this returns. Given a returns directory, one STEP
        file is kept there for every `ret_*` the program holds once it has run,
        so a program that fails to build leaves none.
        """
        # file read check
        try:
            source = model_path.read_text(encoding="utf-8")
        except UnicodeError:
            return CadQueryExecutionReport(
                status=ExecutionStatus.REJECTED,
                executor_error=f"{model_path.name} must be valid UTF-8",
            )
        except PermissionError:
            return CadQueryExecutionReport(
                status=ExecutionStatus.REJECTED,
                executor_error=f"{model_path.name} is not readable",
            )
        except OSError as error:
            reason = error.strerror or type(error).__name__
            return CadQueryExecutionReport(
                status=ExecutionStatus.INFRA_ERROR,
                executor_error=f"Failed to read {model_path.name}: {reason}",
            )

        # Static check (syntax and self-containedness)
        try:
            self.validate_source(source, filename=model_path.name)
        except (SyntaxError, ValueError) as e:
            return CadQueryExecutionReport(
                source=source,
                status=ExecutionStatus.REJECTED,
                executor_error=_rejection(e),
            )

        # Prepare workdir to bind into the sandbox
        with SandboxWorkdir() as workdir:
            # Copy the program
            (workdir.host_bind_dir / model_path.name).write_text(
                source,
                encoding="utf-8",
            )
            # Copy the runner that executes it
            shutil.copyfile(
                Path(__file__).with_name(_RUNNER),
                workdir.host_bind_dir / _RUNNER,
            )

            # Enumerate intermediate ret_xxx
            ret_names = (
                []
                if intermediate_returns_dir is None
                else _returned_names(source, model_path.name)
            )
            # Run
            sandbox_result = self.sandbox_runner.run(
                command=" ".join(
                    [
                        "python",
                        str(workdir.sandbox_bind_dir / _RUNNER),
                        str(workdir.sandbox_bind_dir / model_path.name),
                        *ret_names,
                    ]
                ),
                workdir=workdir,
            )

            intermediate_returns = (
                ()
                if intermediate_returns_dir is None
                else _read_returns(
                    ret_names,
                    workdir.host_bind_dir / INTERMEDIATE_RETURNS_DIR,
                    intermediate_returns_dir,
                )
            )

            if sandbox_result.status is SandboxStatus.TIMEOUT:
                return CadQueryExecutionReport(
                    source=source,
                    status=ExecutionStatus.TIMEOUT,
                    returncode=sandbox_result.returncode,
                    stdout=sandbox_result.stdout,
                    stderr=sandbox_result.stderr,
                    intermediate_returns=intermediate_returns,
                )

            if sandbox_result.status is SandboxStatus.INFRA_ERROR:
                return CadQueryExecutionReport(
                    source=source,
                    status=ExecutionStatus.INFRA_ERROR,
                    returncode=sandbox_result.returncode,
                    stdout=sandbox_result.stdout,
                    stderr=sandbox_result.stderr,
                    intermediate_returns=intermediate_returns,
                )

            if sandbox_result.returncode != 0:
                return CadQueryExecutionReport(
                    source=source,
                    status=ExecutionStatus.FAILED,
                    returncode=sandbox_result.returncode,
                    stdout=sandbox_result.stdout,
                    stderr=sandbox_result.stderr,
                    intermediate_returns=intermediate_returns,
                )

            # STEP file should be generated here
            tmp_step_path = workdir.host_bind_dir / "output.step"
            if not tmp_step_path.is_file():
                return CadQueryExecutionReport(
                    source=source,
                    status=ExecutionStatus.FAILED,
                    executor_error="output.step was not generated",
                    returncode=sandbox_result.returncode,
                    stdout=sandbox_result.stdout,
                    stderr=sandbox_result.stderr,
                    intermediate_returns=intermediate_returns,
                )

            # Check validity of STEP
            try:
                self.verify_step(tmp_step_path)
            except StepVerificationError as e:
                return CadQueryExecutionReport(
                    source=source,
                    status=ExecutionStatus.FAILED,
                    executor_error=str(e),
                    returncode=sandbox_result.returncode,
                    stdout=sandbox_result.stdout,
                    stderr=sandbox_result.stderr,
                    intermediate_returns=intermediate_returns,
                )

            # Read inside the block: the workdir is gone once it exits.
            census = read_census(tmp_step_path)

            # Copy STEP
            if output_step_path is not None:
                shutil.copyfile(tmp_step_path, output_step_path)

        return CadQueryExecutionReport(
            source=source,
            status=ExecutionStatus.VERIFIED,
            returncode=sandbox_result.returncode,
            stdout=sandbox_result.stdout,
            stderr=sandbox_result.stderr,
            intermediate_returns=intermediate_returns,
            census=census,
        )

    @staticmethod
    def validate_source(
        source: str,
        filename: str = "model.py",
        forbid_try_except: bool = True,
    ) -> None:
        """Reject invalid Python, references to an input DXF file, and caught exceptions."""
        ast_tree = ast.parse(source, filename=filename, mode="exec")
        for ast_node in ast.walk(ast_tree):
            if (
                isinstance(ast_node, ast.Constant)
                and isinstance(ast_node.value, str)
                and ".dxf" in ast_node.value.lower()
            ):
                raise ValueError(
                    "DXF file reading is not allowed. "
                    f"Modify '{filename}' to be self-contained."
                )

            if (
                forbid_try_except
                and isinstance(ast_node, ast.Try | ast.TryStar)
                and ast_node.handlers
            ):
                raise ValueError(
                    f"Try-except is not allowed ('{filename}' line {ast_node.lineno})"
                )

    @staticmethod
    def verify_step(step_path: Path) -> None:
        import cadquery as cq

        if not step_path.is_file():
            raise StepVerificationError(f"STEP file not found: {step_path}")
        if step_path.is_symlink():
            raise StepVerificationError(f"STEP file must not be a symlink: {step_path}")

        try:
            imported = cq.importers.importStep(str(step_path))
        except (OSError, ValueError, RuntimeError) as exc:
            raise StepVerificationError(f"Failed to import STEP: {exc}") from exc

        solids = imported.solids().vals()

        if len(solids) != 1:
            raise StepVerificationError(
                f"Expected exactly one solid, found {len(solids)}"
            )

        if not cq.Shape(solids[0].wrapped).isValid():  # type: ignore[union-attr]
            raise StepVerificationError("STEP contains an invalid solid")


class StepVerificationError(ValueError):
    pass
