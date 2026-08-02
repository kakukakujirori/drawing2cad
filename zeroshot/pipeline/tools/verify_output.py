from dataclasses import asdict, dataclass, replace
from inspect import cleandoc
from pathlib import Path, PurePosixPath
from typing import Self, cast

from langchain_core.tools import BaseTool, tool

from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.verification.run_cadquery import (
    CadQueryExecutionReport,
    CadQueryExecutor,
)

VerifyOutputValue: type = str | int | None


@dataclass(frozen=True)
class VerifyOutputResult:
    verification_id: str | None = None
    status: str = "UNINITIALIZED"
    source: str | None = None
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    executor_error: str | None = None

    def import_from(self, report: CadQueryExecutionReport) -> Self:
        return replace(
            self,
            status=report.status.value,
            source=report.source,
            returncode=report.returncode,
            stdout=report.stdout,
            stderr=report.stderr,
            executor_error=report.executor_error,
        )

    def serialize(self) -> dict[str, VerifyOutputValue]:
        ret = cast(dict[str, VerifyOutputValue], asdict(self))
        ret.pop("source")
        return ret


def create_verify_output_tool(
    executor: CadQueryExecutor,
    workdir: SandboxWorkdir,
    render_views: bool = False,
    source_filename: str = "model.py",
    output_dirname: PurePosixPath = PurePosixPath("attempts"),
    serialize_output: bool = True,
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

        The result reports the execution status, return code, stdout, stderr, and
        any executor error. Generated STEP and its rendered views are saved in
        {workdir.sandbox_bind_dir}/{output_dirname}/<verification_id>/.
        """
    )

    # source_filename sanity check
    source_name = PurePosixPath(source_filename)
    if source_name.name != source_filename or source_name.name in {"", ".", ".."}:
        raise ValueError("source_filename must be a filename in the workdir root")

    # create output dir
    if (
        output_dirname.is_absolute()
        or len(output_dirname.parts) != 1
        or output_dirname.name in {"", ".", ".."}
    ):
        raise ValueError("output_dirname must be a directory basename")

    host_outdir = workdir.host_bind_dir / output_dirname
    if host_outdir.is_symlink():
        raise ValueError("output directory must not be a symlink")
    host_outdir.mkdir(parents=True, exist_ok=True)

    # set output_dirname read-only
    if output_dirname not in workdir.read_only_subdirs:
        workdir.read_only_subdirs.append(output_dirname)

    def _issue_verification_id_and_dir() -> tuple[str, Path]:
        host_outdir = workdir.host_bind_dir / output_dirname
        # issue an id
        existing = [int(p.name) for p in host_outdir.iterdir() if p.name.isdigit()]
        verification_id = f"{(max(existing, default=-1) + 1):03d}"
        # create dir
        host_verification_dir = host_outdir / verification_id
        host_verification_dir.mkdir(parents=True, exist_ok=False)
        return verification_id, host_verification_dir

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

        # prepare artifact save dir and report
        verification_id, host_verification_dir = _issue_verification_id_and_dir()
        report = VerifyOutputResult(verification_id=verification_id)
        output_model_path = host_verification_dir / source_filename
        output_step_path = host_verification_dir / "output.step"

        # execute
        cq_report = executor.execute(model_path, output_step_path)

        # copy source code to output dir
        if cq_report.source is not None:
            output_model_path.write_text(
                cq_report.source,
                encoding="utf-8",
            )

        # update report
        report = report.import_from(cq_report)

        # if verification failed, return early
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
    def verify_output() -> VerifyOutputResult | dict[str, VerifyOutputValue]:
        model_path = workdir.host_bind_dir / source_filename
        ret = _verify(model_path)
        return ret.serialize() if serialize_output else ret

    return verify_output
