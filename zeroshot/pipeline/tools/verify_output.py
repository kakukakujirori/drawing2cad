from dataclasses import dataclass, asdict
from inspect import cleandoc
from pathlib import Path, PurePosixPath
from typing import Self, TypeAlias, cast

from langchain_core.tools import BaseTool, tool

from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.verification.run_cadquery import (
    CadQueryExecutor,
    CadQueryExecutionReport,
)


VerifyOutputValue: TypeAlias = str | int | None


@dataclass(frozen=True)
class VerifyOutputResult:
    status: str = "UNINITIALIZED"
    execution_id: str | None = None
    source_sha256: str | None = None
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    executor_error: str | None = None

    @classmethod
    def import_from(cls, report: CadQueryExecutionReport) -> Self:
        return cls(
            status=report.status.value,
            execution_id=report.exec_id,
            source_sha256=report.source_sha256,
            returncode=report.returncode,
            stdout=report.stdout,
            stderr=report.stderr,
            executor_error=report.executor_error,
        )

    def serialize(self) -> dict[str, VerifyOutputValue]:
        return cast(dict[str, VerifyOutputValue], asdict(self))


def create_verify_output_tool(
    executor: CadQueryExecutor,
    workdir: SandboxWorkdir,
    render_views: bool = False,
    source_filename: str = "model.py",
) -> BaseTool:
    description = cleandoc(
        f"""
        Execute and validate the current CadQuery program at
        {workdir.sandbox_bind_dir}/{source_filename}.

        The file must be a self-contained Python program that assigns the final
        CadQuery object to a variable named `result`. Verification runs in a fresh
        isolated directory, so other files in {workdir.sandbox_bind_dir} are not
        available to the program. The tool exports `result` to STEP and accepts it
        only when the exported file contains exactly one valid solid.

        The result reports the execution status, execution ID, source hash,
        return code, stdout, stderr, and any executor error. A `VERIFIED` status
        means that the validated STEP artifact was saved by the pipeline. Other
        statuses and errors are observations for you to interpret. Calling this
        tool does not by itself finish the reconstruction task.
        """
    )

    # source_filename sanity check
    source_name = PurePosixPath(source_filename)
    if source_name.name != source_filename or source_name.name in {"", ".", ".."}:
        raise ValueError("source_filename must be a filename in the workdir root")

    def _verify(model_path: Path) -> VerifyOutputResult:
        # file existence check
        if model_path.is_symlink():
            report = VerifyOutputResult(
                status="REJECTED",
                executor_error=f"{source_filename} must not be a symlink",
            )
            return report

        if not model_path.is_file():
            report = VerifyOutputResult(
                status="REJECTED",
                executor_error=f"{source_filename} was not found",
            )
            return report

        # file read check
        try:
            source = model_path.read_text(encoding="utf-8")
        except UnicodeError:
            return VerifyOutputResult(
                status="REJECTED",
                executor_error=f"{source_filename} must be valid UTF-8",
            )
        except PermissionError:
            return VerifyOutputResult(
                status="REJECTED",
                executor_error=f"{source_filename} is not readable",
            )
        except OSError as error:
            reason = error.strerror or type(error).__name__
            return VerifyOutputResult(
                status="INFRA_ERROR",
                executor_error=f"Failed to read {source_filename}: {reason}",
            )

        # execute
        cq_report = executor.execute(source)

        report = VerifyOutputResult.import_from(cq_report)

        if not (report.status == "VERIFIED" and report.returncode == 0):
            return report

        # render 2D techdraw
        # TODO: IMPLEMENT

        # render perspective views
        # TODO: IMPLEMENT
        if render_views:
            raise NotImplementedError("View rendering not implemented yet")

        # wrap the results into FeedbackManifest
        # TODO: IMPLEMENT

        return report

    @tool("verify_output", description=description)
    def verify_output() -> dict[str, VerifyOutputValue]:
        model_path = workdir.host_bind_dir / source_filename
        return _verify(model_path).serialize()

    return verify_output
