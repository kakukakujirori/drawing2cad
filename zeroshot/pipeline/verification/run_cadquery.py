import ast
import shutil
import tempfile
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from zeroshot.pipeline.sandbox import SandboxRunner, SandboxStatus


class ExecutionStatus(Enum):
    VERIFIED = "VERIFIED"
    REJECTED = "REJECTED"  # syntax/validation error
    FAILED = "FAILED"  # execution error (no STEP generated)
    TIMEOUT = "TIMEOUT"  # CAD process timeout
    INFRA_ERROR = "INFRA_ERROR"  # sandbox infrastructure failure


@dataclass(frozen=True)
class CadQueryExecutionReport:
    exec_id: str
    source_path: Path | None
    source_sha256: str | None
    status: ExecutionStatus
    step_path: Path | None
    executor_error: str | None = None
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""


class CadQueryExecutor:
    def __init__(self, artifact_root: Path, sandbox_runner: SandboxRunner) -> None:
        self.artifact_root = artifact_root
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

    def execute(self, source: str) -> CadQueryExecutionReport:
        # Issue an ID and save source
        exec_id = str(uuid4())
        exec_dir = self.artifact_root / exec_id
        source_path = exec_dir / "model.py"
        step_path = exec_dir / "output.step"

        exec_dir.mkdir(parents=True, exist_ok=False)
        source_path.write_text(source, encoding="utf-8")
        source_sha256 = sha256(source.encode("utf-8")).hexdigest()

        # Static check
        try:
            self.validate_source(source)
        except SyntaxError as e:
            return CadQueryExecutionReport(
                exec_id=exec_id,
                source_path=source_path,
                source_sha256=source_sha256,
                step_path=None,
                status=ExecutionStatus.REJECTED,
                executor_error=str(e),
            )

        # Prepare workdir to bind into the sandbox
        with tempfile.TemporaryDirectory() as work_dir:
            work_dir = Path(work_dir)
            exec_path = work_dir / "model.py"
            exec_path.write_text(
                source + self.EXPORT_EPILOGUE,
                encoding="utf-8",
            )

            sandbox_result = self.sandbox_runner.run(
                command="python model.py",
                work_dir=work_dir,
            )

            if sandbox_result.status is SandboxStatus.TIMEOUT:
                return CadQueryExecutionReport(
                    exec_id=exec_id,
                    source_path=source_path,
                    source_sha256=source_sha256,
                    step_path=None,
                    status=ExecutionStatus.TIMEOUT,
                    returncode=sandbox_result.returncode,
                    stdout=sandbox_result.stdout,
                    stderr=sandbox_result.stderr,
                )

            if sandbox_result.status is SandboxStatus.INFRA_ERROR:
                return CadQueryExecutionReport(
                    exec_id=exec_id,
                    source_path=source_path,
                    source_sha256=source_sha256,
                    step_path=None,
                    status=ExecutionStatus.INFRA_ERROR,
                    returncode=sandbox_result.returncode,
                    stdout=sandbox_result.stdout,
                    stderr=sandbox_result.stderr,
                )

            if sandbox_result.returncode != 0:
                return CadQueryExecutionReport(
                    exec_id=exec_id,
                    source_path=source_path,
                    source_sha256=source_sha256,
                    step_path=None,
                    status=ExecutionStatus.FAILED,
                    returncode=sandbox_result.returncode,
                    stdout=sandbox_result.stdout,
                    stderr=sandbox_result.stderr,
                )

            # STEP file should be generated here
            tmp_step_path = work_dir / "output.step"
            if not tmp_step_path.is_file():
                return CadQueryExecutionReport(
                    exec_id=exec_id,
                    source_path=source_path,
                    source_sha256=source_sha256,
                    step_path=None,
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
                    exec_id=exec_id,
                    source_path=source_path,
                    source_sha256=source_sha256,
                    step_path=None,
                    status=ExecutionStatus.FAILED,
                    executor_error=str(e),
                    returncode=sandbox_result.returncode,
                    stdout=sandbox_result.stdout,
                    stderr=sandbox_result.stderr,
                )

            # Move STEP file to the artifact directory
            shutil.move(tmp_step_path, step_path)

        return CadQueryExecutionReport(
            exec_id=exec_id,
            source_path=source_path,
            source_sha256=source_sha256,
            status=ExecutionStatus.VERIFIED,
            step_path=step_path,
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
