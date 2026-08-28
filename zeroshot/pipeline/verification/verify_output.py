import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Self, cast

from langchain_core.messages.content import ContentBlock, create_text_block

from zeroshot.pipeline.messages import ArtifactPresenter, FeedbackManifest
from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.verification.render.constants import (
    Render3dPaths,
    TechdrawPaths,
)
from zeroshot.pipeline.verification.run_cadquery import (
    CadQueryExecutionReport,
    CadQueryExecutor,
)
from zeroshot.pipeline.verification.run_render import RenderReport, StepRenderer

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
    shape: str = ""

    def import_from(self, report: CadQueryExecutionReport) -> Self:
        return replace(
            self,
            status=report.status.value,
            source=report.source,
            returncode=report.returncode,
            stdout=report.stdout,
            stderr=report.stderr,
            executor_error=report.executor_error,
            shape=report.shape,
        )

    def serialize(self) -> dict[str, VerifyOutputValue]:
        ret = cast(dict[str, VerifyOutputValue], asdict(self))
        ret.pop("source")
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

        host_outdir = workdir.host_bind_dir / output_dirname
        if host_outdir.is_symlink():
            raise ValueError("output directory must not be a symlink")
        host_outdir.mkdir(parents=True, exist_ok=True)

        # set output_dirname read-only
        if output_dirname not in workdir.read_only_subdirs:
            workdir.read_only_subdirs.append(output_dirname)

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
                status="REJECTED",
                executor_error=f"{self.source_filename} must not be a symlink",
            )
            return report, None

        if not model_path.is_file():
            report = VerifyOutputResult(
                status="REJECTED",
                executor_error=f"{self.source_filename} was not found",
            )
            return report, None

        # prepare artifact save dir and report
        verification_id, host_verification_dir = self._issue_verification_id_and_dir()
        report = VerifyOutputResult(verification_id=verification_id)
        output_model_path = host_verification_dir / self.source_filename
        output_step_path = host_verification_dir / "output.step"

        # execute
        cq_report = self.executor.execute(model_path, output_step_path)

        # copy source code to output dir
        if cq_report.source is not None:
            output_model_path.write_text(
                cq_report.source,
                encoding="utf-8",
            )

        # update report
        report = report.import_from(cq_report)

        # if verification failed, return early. No STEP means nothing to draw.
        if not (report.status == "VERIFIED" and report.returncode == 0):
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

    def feedback(self) -> list[ContentBlock]:
        """Verify, and say what happened in blocks a message can carry."""
        report, manifest = self.verify()
        sandbox_source = self.workdir.sandbox_bind_dir / self.source_filename
        blocks: list[ContentBlock] = [
            create_text_block(
                f"{sandbox_source} has been executed, and upon successful STEP "
                "file generation, its rendering images are exported:"
            ),
            create_text_block(json.dumps(report.serialize(), indent=2)),
        ]
        if manifest and self.artifact_presenter:
            blocks.extend(
                self.artifact_presenter.build_feedback_message_blocks(
                    manifest, self.workdir
                )
            )
        return blocks
