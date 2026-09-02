import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Self, cast

from langchain_core.messages.content import ContentBlock, create_text_block

from zeroshot.pipeline.messages import ArtifactPresenter, FeedbackManifest
from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.verification._run_program import INTERMEDIATE_RETURNS_DIR
from zeroshot.pipeline.verification.render.constants import (
    Render3dPaths,
    TechdrawPaths,
)
from zeroshot.pipeline.verification.run_cadquery import (
    CadQueryExecutionReport,
    CadQueryExecutor,
    ExecutionStatus,
    IntermediateReturn,
)
from zeroshot.pipeline.verification.run_render import RenderReport, StepRenderer
from zeroshot.pipeline.verification.shape_census import ShapeCensus

VerifyOutputValue: type = str | int | None


def _census_table(returns: Sequence[IntermediateReturn]) -> str:
    """Lay out what each `ret_*` held, one line each, in program order.

    Every line after the first states its change from the line above, so an
    operation that built nothing and an operation that undid the one before it
    both show on the face of the table. A return that never reached a STEP
    carries the reason instead, and the line after it compares against the last
    return that did.
    """
    width = max((len(output.name) for output in returns), default=0)
    lines: list[str] = []
    previous: ShapeCensus | None = None
    for output in returns:
        if output.census is None:
            lines.append(f"{output.name:<{width}}  not exported: {output.error}")
            continue
        lines.append(
            f"{output.name:<{width}}  "
            + (
                output.census.describe()
                if previous is None
                else output.census.describe_change_from(previous)
            )
        )
        previous = output.census
    return "\n".join(lines)


def _describe_returns(
    intermediate_returns: Sequence[IntermediateReturn],
    intermediate_renders: Mapping[str, RenderReport],
    sandbox_returns_dir: PurePosixPath,
) -> str:
    """Say what every `ret_*` came out as, and where its views were written.

    One sentence for the layout rather than four paths per return: every
    directory holds the same file names, so listing them all would spend
    tokens on a convention the reader can apply once.
    """
    if not intermediate_returns:
        return ""
    failures = [
        f"{name}: {reason}"
        for name, report in intermediate_renders.items()
        for reason in (
            *report.techdraw_errors.values(),
            *report.render3d_errors.values(),
        )
    ]
    return "\n".join(
        [
            (
                "Intermediate returns, in program order. Each line is the solid "
                "the operation left behind, and how it differs from the line above."
            ),
            "",
            _census_table(intermediate_returns),
            "",
            (
                f"Each is written to {sandbox_returns_dir}/<name>/ as output.step, "
                "techdraw.dxf and render_3d/<style>.png."
            ),
            *(["", "Views that could not be drawn:", *failures] if failures else []),
        ]
    )


@dataclass(frozen=True)
class VerifyOutputResult:
    verification_id: str | None = None
    status: ExecutionStatus = ExecutionStatus.UNINITIALIZED
    source: str | None = None
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    executor_error: str | None = None
    shape: str = ""
    # Host paths never reach here, so this is written in sandbox paths.
    intermediate_returns: str = ""

    def import_from(self, report: CadQueryExecutionReport) -> Self:
        return replace(
            self,
            status=report.status,
            source=report.source,
            returncode=report.returncode,
            stdout=report.stdout,
            stderr=report.stderr,
            executor_error=report.executor_error,
            shape=report.census.describe() if report.census else "",
        )

    def serialize(self) -> dict[str, VerifyOutputValue]:
        ret = cast(dict[str, VerifyOutputValue], asdict(self))
        ret.pop("source")
        # Carried as its own block, because JSON escaping makes a table
        # of many lines unreadable.
        ret.pop("intermediate_returns")
        return ret


class OutputVerifier:
    """Build the current program and report what it produced.

    Shared by the middleware that checks a coder turn and by the workflow's own
    final verification, so both number their attempts from the same directory.
    """

    def __init__(
        self,
        executor: CadQueryExecutor,
        workdir: SandboxWorkdir,
        renderer: StepRenderer,
        artifact_presenter: ArtifactPresenter | None,
        source_filename: str = "model.py",
        output_dirname: PurePosixPath = PurePosixPath("attempts"),
        show_intermediate_returns: bool = True,
    ) -> None:
        source_path = PurePosixPath(source_filename)
        if (
            source_path.is_absolute()
            or len(source_path.parts) != 1
            or source_path.suffix != ".py"
        ):
            raise ValueError("source_filename must be a Python file basename")
        if (
            output_dirname.is_absolute()
            or len(output_dirname.parts) != 1
            or output_dirname.name in {"", ".", ".."}
        ):
            raise ValueError("output_dirname must be a directory basename")

        self.executor = executor
        self.workdir = workdir
        self.renderer = renderer
        self.artifact_presenter = artifact_presenter
        self.source_filename = source_filename
        self.output_dirname = output_dirname
        self.show_intermediate_returns = show_intermediate_returns

        host_outdir = workdir.host_bind_dir / output_dirname
        if host_outdir.is_symlink():
            raise ValueError("output directory must not be a symlink")
        host_outdir.mkdir(parents=True, exist_ok=True)

        # set output_dirname read-only
        if output_dirname not in workdir.read_only_subdirs:
            workdir.read_only_subdirs.append(output_dirname)

        self._last_feedback_report: VerifyOutputResult | None = None

    @property
    def source_path(self) -> Path:
        """The program this verifier builds, on the host side of the sandbox."""
        return self.workdir.host_bind_dir / self.source_filename

    def _issue_verification_id_and_dir(self) -> tuple[str, Path]:
        host_outdir = self.workdir.host_bind_dir / self.output_dirname
        # issue an id
        existing = [int(p.name) for p in host_outdir.iterdir() if p.name.isdigit()]
        verification_id = f"{(max(existing, default=-1) + 1):03d}"
        # create dir
        host_verification_dir = host_outdir / verification_id
        host_verification_dir.mkdir(parents=True, exist_ok=False)
        return verification_id, host_verification_dir

    def verify(self) -> tuple[VerifyOutputResult, FeedbackManifest | None]:
        """Verify the program and, when it yields a solid, render its views.

        The manifest stays out of the report because it carries host paths, and
        only the report is msgpack-serialisable enough to reach graph state.
        """
        model_path = self.source_path

        # file existence check
        if model_path.is_symlink():
            report = VerifyOutputResult(
                status=ExecutionStatus.REJECTED,
                executor_error=f"{self.source_filename} must not be a symlink",
            )
            return report, None

        if not model_path.is_file():
            report = VerifyOutputResult(
                status=ExecutionStatus.REJECTED,
                executor_error=f"{self.source_filename} was not found",
            )
            return report, None

        # prepare artifact save dir and report
        verification_id, host_verification_dir = self._issue_verification_id_and_dir()
        report = VerifyOutputResult(verification_id=verification_id)
        output_model_path = host_verification_dir / self.source_filename
        output_step_path = host_verification_dir / "output.step"

        # execute, keeping the solid each planned operation left behind (ret_xxx)
        cq_report = self.executor.execute(
            model_path,
            output_step_path,
            intermediate_returns_dir=(
                host_verification_dir / INTERMEDIATE_RETURNS_DIR
                if self.show_intermediate_returns
                else None
            ),
        )

        # copy source code to output dir
        if cq_report.source is not None:
            output_model_path.write_text(
                cq_report.source,
                encoding="utf-8",
            )

        # update report
        report = report.import_from(cq_report)

        # Draw and describe every ret_xxx the program left behind. A program
        # that ran leaves these whether or not `result` passed, and a result
        # that failed is when they are most worth reading.
        host_returns_dir = host_verification_dir / INTERMEDIATE_RETURNS_DIR
        renders = {
            output.name: self._render(output.step_path, host_returns_dir / output.name)
            for output in cq_report.intermediate_returns
            if output.step_path is not None
        }
        report = replace(
            report,
            intermediate_returns=_describe_returns(
                cq_report.intermediate_returns,
                renders,
                self.workdir.sandbox_bind_dir
                / self.output_dirname
                / verification_id
                / INTERMEDIATE_RETURNS_DIR,
            ),
        )

        # A result that did not build has no STEP of its own to draw.
        if not (report.status == ExecutionStatus.VERIFIED and report.returncode == 0):
            return report, None

        # render the three-view DXF and the perspective PNGs
        render_report = self._render(output_step_path, host_verification_dir)

        manifest = FeedbackManifest(
            verification_id=verification_id,
            dxf_path=render_report.techdraw_paths.dxf,
            dxf_error=render_report.techdraw_errors.get("dxf"),
            render3d_paths=render_report.render3d_paths.as_mapping(),
            render3d_errors=render_report.render3d_errors,
        )
        return report, manifest

    def _render(self, step_path: Path, verification_dir: Path) -> RenderReport:
        """Render feedback artifacts into the verification directory."""
        # Flat: an attempt holds one drawing and one set of renders, and the
        # model has already been shown the inputs under the same convention.
        techdraw_paths = TechdrawPaths.flat(verification_dir / "techdraw")
        render3d_paths = Render3dPaths.flat(verification_dir / "render_3d")
        # The renderer leaves directory layout to its caller.
        for path in (
            *techdraw_paths.as_mapping().values(),
            *render3d_paths.as_mapping().values(),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)

        return self.renderer.render(step_path, techdraw_paths, render3d_paths)

    @property
    def confirmed_a_solid(self) -> bool:
        """Whether the most recent `feedback` build yielded a usable STEP.

        False before the first build, so a program never built cannot pass for
        one that did.
        """
        report = self._last_feedback_report
        if report is None:
            return False
        return report.status is ExecutionStatus.VERIFIED and report.returncode == 0

    def feedback(self) -> list[ContentBlock]:
        """Verify, and say what happened in blocks a message can carry."""
        report, manifest = self.verify()
        self._last_feedback_report = report
        sandbox_source = self.workdir.sandbox_bind_dir / self.source_filename
        blocks: list[ContentBlock] = [
            create_text_block(
                f"{sandbox_source} has been executed, and upon successful STEP "
                "file generation, its rendering images are exported:"
            ),
            create_text_block(json.dumps(report.serialize(), indent=2)),
        ]
        if report.intermediate_returns:
            blocks.append(create_text_block(report.intermediate_returns))
        if manifest and self.artifact_presenter:
            blocks.extend(
                self.artifact_presenter.build_feedback_message_blocks(
                    manifest, self.workdir
                )
            )
        return blocks
