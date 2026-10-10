import ast
import shutil
import traceback
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from zeroshot.pipeline.sandbox import (
    SandboxRunner,
    SandboxStatus,
    SandboxWorkdir,
)
from zeroshot.pipeline.verification.shape_census import ShapeCensus, read_census

_RUNNER = "_run_program.py"


class ExecutionStatus(StrEnum):
    UNINITIALIZED = "UNINITIALIZED"
    VERIFIED = "VERIFIED"
    REJECTED = "REJECTED"  # syntax/validation error
    FAILED = "FAILED"  # execution or STEP validation error
    TIMEOUT = "TIMEOUT"  # CAD process timeout
    INFRA_ERROR = "INFRA_ERROR"  # sandbox infrastructure failure


@dataclass(frozen=True)
class CadQueryExecutionReport:
    source: str | None = None
    status: ExecutionStatus = ExecutionStatus.INFRA_ERROR
    executor_error: str | None = None
    returncode: int | None = None
    # A kept result may be diagnostic only; status determines acceptance.
    step_path: Path | None = None
    stdout: str = ""
    stderr: str = ""
    census: ShapeCensus | None = None


def _rejection(error: SyntaxError | ValueError) -> str:
    """Say what is wrong with the source.

    A syntax error is given the offending line and the caret under it, which
    `str` leaves out.
    """
    if not isinstance(error, SyntaxError):
        return str(error)
    return "".join(traceback.format_exception_only(type(error), error)).strip()


class CadQueryExecutor:
    def __init__(self, sandbox_runner: SandboxRunner) -> None:
        self.sandbox_runner = sandbox_runner

    def execute(
        self,
        model_path: Path,
        output_step_path: Path | None = None,
    ) -> CadQueryExecutionReport:
        """Run the program in the sandbox and report what it built.

        `output_step_path` is a host path the caller keeps. The sandbox writes
        `output.step` in its own workdir, which is gone by the time this returns.
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

            # Run
            sandbox_result = self.sandbox_runner.run(
                command=" ".join(
                    [
                        "python",
                        str(workdir.sandbox_bind_dir / _RUNNER),
                        str(workdir.sandbox_bind_dir / model_path.name),
                    ]
                ),
                workdir=workdir,
            )

            if sandbox_result.status is SandboxStatus.TIMEOUT:
                return CadQueryExecutionReport(
                    source=source,
                    status=ExecutionStatus.TIMEOUT,
                    returncode=sandbox_result.returncode,
                    stdout=sandbox_result.stdout,
                    stderr=sandbox_result.stderr,
                )

            if sandbox_result.status is SandboxStatus.INFRA_ERROR:
                return CadQueryExecutionReport(
                    source=source,
                    status=ExecutionStatus.INFRA_ERROR,
                    returncode=sandbox_result.returncode,
                    stdout=sandbox_result.stdout,
                    stderr=sandbox_result.stderr,
                )

            if sandbox_result.returncode != 0:
                return CadQueryExecutionReport(
                    source=source,
                    status=ExecutionStatus.FAILED,
                    returncode=sandbox_result.returncode,
                    stdout=sandbox_result.stdout,
                    stderr=sandbox_result.stderr,
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
                )

            # Keep a readable result even when STEP validation rejects its geometry.
            step_error = None
            try:
                self.verify_step(tmp_step_path)
            except StepVerificationError as e:
                step_error = str(e)

            # Read inside the block: the workdir is gone once it exits.
            census = None if tmp_step_path.is_symlink() else read_census(tmp_step_path)

            kept_step_path = None
            if output_step_path is not None and (
                step_error is None or census is not None
            ):
                shutil.copyfile(tmp_step_path, output_step_path)
                kept_step_path = output_step_path

        return CadQueryExecutionReport(
            source=source,
            status=(
                ExecutionStatus.FAILED
                if step_error is not None
                else ExecutionStatus.VERIFIED
            ),
            executor_error=step_error,
            returncode=sandbox_result.returncode,
            step_path=kept_step_path,
            stdout=sandbox_result.stdout,
            stderr=sandbox_result.stderr,
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
