import ast
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from zeroshot.pipeline.sandbox import (
    SandboxRunner,
    SandboxStatus,
    SandboxWorkdir,
)


class ExecutionStatus(Enum):
    VERIFIED = "VERIFIED"
    REJECTED = "REJECTED"  # syntax/validation error
    FAILED = "FAILED"  # execution error (no STEP generated)
    TIMEOUT = "TIMEOUT"  # CAD process timeout
    INFRA_ERROR = "INFRA_ERROR"  # sandbox infrastructure failure


@dataclass(frozen=True)
class CadQueryExecutionReport:
    source: str | None = None
    status: ExecutionStatus = ExecutionStatus.INFRA_ERROR
    executor_error: str | None = None
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""


class CadQueryExecutor:
    def __init__(self, sandbox_runner: SandboxRunner) -> None:
        self.sandbox_runner = sandbox_runner

        self.EXPORT_EPILOGUE = """

# Added by trusted CadQueryExecutor.
import cadquery as _pipeline_cq
_pipeline_cq.exporters.export(
    result,
    "output.step",
    exportType="STEP",
)
"""

    def execute(
        self,
        model_path: Path,
        output_step_path: Path | None = None,
    ) -> CadQueryExecutionReport:
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

        # Static check
        try:
            self.validate_source(source)
        except SyntaxError as e:
            return CadQueryExecutionReport(
                source=source,
                status=ExecutionStatus.REJECTED,
                executor_error=str(e),
            )

        # Prepare workdir to bind into the sandbox
        with SandboxWorkdir() as workdir:
            exec_path = workdir.host_bind_dir / "model.py"
            exec_path.write_text(
                source + self.EXPORT_EPILOGUE,
                encoding="utf-8",
            )

            sandbox_result = self.sandbox_runner.run(
                command="python model.py",
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
                )

            # Copy STEP
            if output_step_path is not None:
                shutil.copyfile(tmp_step_path, output_step_path)

        return CadQueryExecutionReport(
            source=source,
            status=ExecutionStatus.VERIFIED,
            returncode=sandbox_result.returncode,
            stdout=sandbox_result.stdout,
            stderr=sandbox_result.stderr,
        )

    @staticmethod
    def validate_source(source: str) -> None:
        """Raise SyntaxError when *source* is not valid Python module code.

        This function deliberately does not enforce import, API, filesystem, or
        CAD-library policy. Those are controlled by the sandbox and runtime.
        """
        ast.parse(source, filename="model.py", mode="exec")

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
