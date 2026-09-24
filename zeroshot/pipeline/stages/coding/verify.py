import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from inspect import cleandoc
from pathlib import Path, PurePosixPath
from typing import Literal

from langchain_core.messages.content import ContentBlock, create_text_block

from zeroshot.pipeline.messages.artifact import (
    View,
    build_feedback_message_blocks,
)
from zeroshot.pipeline.messages.manifest import FeedbackManifest, register_view
from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.stages.interpretation.contracts import (
    Axis,
    DrawingInterpretation,
    DrawingView,
)
from zeroshot.pipeline.stages.operations.contracts import OperationPlan
from zeroshot.pipeline.verification._run_program import INTERMEDIATE_RETURNS_DIR
from zeroshot.pipeline.verification.attempts import AttemptStore
from zeroshot.pipeline.verification.check_program import check_program
from zeroshot.pipeline.verification.render.constants import (
    ProjectionPaths,
    Render3dPaths,
)
from zeroshot.pipeline.verification.run_cadquery import (
    CadQueryExecutionReport,
    CadQueryExecutor,
    ExecutionStatus,
    IntermediateReturn,
)
from zeroshot.pipeline.verification.run_drawing_diff import (
    DrawingDiffExecutor,
    DrawingDiffReport,
)
from zeroshot.pipeline.verification.run_render import (
    RenderReport,
    RenderRequest,
    StepRenderer,
)
from zeroshot.pipeline.verification.shape_census import ShapeCensus

# Build outcomes a second attempt at the same bytes could come out of
# differently, because they turn on how loaded the machine was.
_TRANSIENT_OUTCOMES = frozenset({ExecutionStatus.TIMEOUT, ExecutionStatus.INFRA_ERROR})

# The renderer draws three styles of the one perspective, and only this one is
# offered: it is the line art of `hlg_perspective` over a faint copy of the
# shaded pass, so it says which side is material as well as where the edges
# are. Three pictures of one camera would spend a message saying it three times.
FEEDBACK_PICTORIAL = "hlg_translucent_faces_perspective"

# model.py's final result variable name
RESULT_NAME = "result"


@dataclass(frozen=True)
class VerifyOutputResult:
    verification_id: str | None = None
    host_verification_dir: Path | None = None
    sandbox_verification_dir: str | None = (
        None  # PurePosixPath fails to save in LangGraph
    )

    exec_report: CadQueryExecutionReport | None = None
    render_report: dict[str, RenderReport] | None = None
    drawing_diff_report: dict[str, DrawingDiffReport] | None = None


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
        diff_drawer: DrawingDiffExecutor | None,
        feedback_presentation_mode: Literal["none", "path", "image"],
        attempt_store: AttemptStore,
        source_filename: str = "model.py",
        show_intermediate_returns: bool = True,
    ) -> None:
        source_path = PurePosixPath(source_filename)
        if (
            source_path.is_absolute()
            or len(source_path.parts) != 1
            or source_path.suffix != ".py"
        ):
            raise ValueError("source_filename must be a Python file basename")
        self.executor = executor
        self.workdir = workdir
        self.renderer = renderer
        self.diff_drawer = diff_drawer
        self.feedback_presentation_mode = feedback_presentation_mode
        # The previous stages' deliverables must be available to the verifier
        self.interpretation: DrawingInterpretation | None = None
        self.operations: OperationPlan | None = None
        self.source_filename = source_filename
        self.attempt_store = attempt_store
        self.show_intermediate_returns = show_intermediate_returns

        self._last_feedback_report: VerifyOutputResult | None = None
        # What the last build returned, and the digest of the program it ran
        # on. Assigned together, so one is never read against the other. See
        # `verify`.
        self._built: VerifyOutputResult | None = None
        self._built_from_digest: str | None = None

    @property
    def source_path(self) -> Path:
        """The program this verifier builds, on the host side of the sandbox."""
        return self.workdir.host_bind_dir / self.source_filename

    def source_digest(self) -> str | None:
        """What the program is right now, or nothing when there is no program."""
        path = self.source_path
        if path.is_symlink() or not path.is_file():
            return None
        return sha256(path.read_bytes()).hexdigest()

    def reset(self) -> None:
        """Forget build state before a new coding-stage invocation."""
        self._built = None
        self._built_from_digest = None
        self._last_feedback_report = None

    def verify(self) -> VerifyOutputResult:
        """Verify the program and, when it yields a solid, render its views.

        Within a coding-stage invocation, reuse a non-transient result while
        model.py is unchanged. Middleware and stage integration can both call
        this method; reset() clears the cache before the next invocation, when
        the drawing context may differ.
        """
        digest = self.source_digest()
        if self._built is not None and digest == self._built_from_digest:
            return self._built

        report = self._build()
        assert report.exec_report is not None

        if digest is not None and report.exec_report.status not in _TRANSIENT_OUTCOMES:
            self._built = report
            self._built_from_digest = digest

        return report

    def _build(self) -> VerifyOutputResult:

        # file existence check
        if self.source_path.is_symlink():
            return VerifyOutputResult(
                exec_report=CadQueryExecutionReport(
                    status=ExecutionStatus.REJECTED,
                    executor_error=f"{self.source_filename} must not be a symlink",
                )
            )

        if not self.source_path.is_file():
            return VerifyOutputResult(
                exec_report=CadQueryExecutionReport(
                    status=ExecutionStatus.REJECTED,
                    executor_error=f"{self.source_filename} was not found",
                )
            )

        # prepare artifact save dir
        verification_id, host_verification_dir, sandbox_verification_dir = (
            self.attempt_store.issue("coding")
        )

        # execute -> render -> draw_diff
        cq_report = self._execute(host_verification_dir)
        render_report = self._render(cq_report, host_verification_dir)
        final_render_report = render_report.get(RESULT_NAME)
        diff_report = (
            self._draw_diffs(final_render_report) if final_render_report else None
        )

        # compile the reports and the manifest.
        return VerifyOutputResult(
            verification_id=verification_id,
            host_verification_dir=host_verification_dir,
            sandbox_verification_dir=str(sandbox_verification_dir),
            exec_report=cq_report,
            render_report=render_report,
            drawing_diff_report=diff_report,
        )

    def _execute(self, host_verification_dir: Path) -> CadQueryExecutionReport:
        output_model_path = host_verification_dir / self.source_filename
        output_step_path = host_verification_dir / "output.step"

        cq_report = self.executor.execute(
            self.source_path,
            output_step_path,
            intermediate_returns_dir=(
                host_verification_dir / INTERMEDIATE_RETURNS_DIR
                if self.show_intermediate_returns
                else None
            ),
        )
        if cq_report.source is not None:
            output_model_path.write_text(
                cq_report.source,
                encoding="utf-8",
            )

        return cq_report

    def _issue_render_request(
        self, step_path: Path, verification_dir: Path
    ) -> RenderRequest:
        """Name the feedback artifacts one STEP is to be drawn into."""
        # Flat: one file per view and per style, named as the inputs are, so
        # the model does not have to guess a second convention.
        assert self.interpretation is not None
        view_frames = self.interpretation.view_frames()

        projection_paths = ProjectionPaths.flat(
            verification_dir / "projection", view_frames
        )
        render3d_paths = Render3dPaths.flat(verification_dir / "render_3d")
        # The renderer leaves directory layout to its caller.
        for path in (
            *projection_paths.as_mapping().values(),
            *render3d_paths.as_mapping().values(),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)

        return RenderRequest(step_path, projection_paths, render3d_paths, view_frames)

    def _render(
        self,
        cq_report: CadQueryExecutionReport,
        host_verification_dir: Path,
    ) -> dict[str, RenderReport]:
        """Draw and describe every `ret_xxx` and `result` the program left behind.

        A program that ran leaves the returns whether or not `result` passed, and
        a result that failed is when they are most worth reading; a result that
        passed adds one more drawing of the same kind, so all of them are drawn in one batch.
        """
        host_returns_dir = host_verification_dir / INTERMEDIATE_RETURNS_DIR
        built_steps = [
            (output.name, output.step_path)
            for output in cq_report.intermediate_returns
            if output.step_path is not None
        ]
        render_requests = [
            self._issue_render_request(step_path, host_returns_dir / name)
            for name, step_path in built_steps
        ]

        # if the final result built a STEP, draw it too here.
        if cq_report.status == ExecutionStatus.VERIFIED and cq_report.returncode == 0:
            if cq_report.step_path is None or not cq_report.step_path.is_file():
                raise ValueError("result built but no STEP path was returned?")
            assert RESULT_NAME not in (name for name, _ in built_steps)
            built_steps.append((RESULT_NAME, cq_report.step_path))
            render_requests.append(
                self._issue_render_request(cq_report.step_path, host_verification_dir)
            )

        # render all the requests in one batch.
        return {
            name: ret
            for (name, _), ret in zip(
                built_steps,
                self.renderer.render_many(render_requests),
                strict=True,
            )
        }

    def _draw_diffs(
        self,
        render_report: RenderReport,
    ) -> dict[str, DrawingDiffReport] | None:
        if self.diff_drawer is None:
            return None
        assert self.interpretation is not None

        view_frames = self.interpretation.view_frames()
        projections = render_report.projection_paths.as_mapping()
        workspace = self.workdir.host_bind_dir.resolve()
        selected: list[tuple[DrawingView, Path, Path]] = []
        skipped: list[tuple[DrawingView, Path, Path | None, str]] = []
        seen_roles: set[View] = set()

        # pick up comparable views
        for view in self.interpretation.views:
            # input output paths
            drawing_path = self.workdir.sandbox_to_host_path(view.file)
            proj_dxf_path = projections.get(view.role.value)
            proj_png_path = (
                proj_dxf_path.with_suffix(".png") if proj_dxf_path is not None else None
            )

            # sanity checks
            if view.role not in view_frames:  # e.g., FULL_PAGE
                continue
            if view.role in seen_roles:
                skipped.append(
                    (
                        view,
                        drawing_path,
                        proj_png_path,
                        "another registered view already uses this role",
                    )
                )
                continue
            seen_roles.add(view.role)

            if proj_dxf_path is None:
                skipped.append(
                    (view, drawing_path, proj_png_path, "projection unavailable")
                )
                continue

            if proj_png_path is None or not proj_png_path.is_file():
                skipped.append(
                    (view, drawing_path, proj_png_path, "projection PNG unavailable")
                )
                continue

            if not (drawing_path.is_file() and drawing_path.is_relative_to(workspace)):
                skipped.append(
                    (
                        view,
                        drawing_path,
                        proj_png_path,
                        "input drawing unavailable or outside workspace",
                    )
                )
                continue

            if drawing_path.suffix.lower() == ".dxf":
                skipped.append(
                    (
                        view,
                        drawing_path,
                        proj_png_path,
                        "native DXF input has no raster image to align",
                    )
                )
                continue

            selected.append((view, drawing_path, proj_png_path))

        # run the diff on the selected pairs
        raw_reports = self.diff_drawer.execute(
            [(drawing, projection) for _, drawing, projection in selected]
        )

        compared: dict[str, DrawingDiffReport] = {
            view.name: report
            for (view, _, _), report in zip(selected, raw_reports, strict=True)
        } | {
            view.name: DrawingDiffReport(
                drawing_path=drawing_path,
                projection_path=projection_path,
                error=why,
            )
            for view, drawing_path, projection_path, why in skipped
        }

        return compared

    def _make_manifest(
        self, render_report: RenderReport, verification_id: str
    ) -> FeedbackManifest:
        pictorial = render_report.render3d_paths.as_mapping().get(FEEDBACK_PICTORIAL)
        sheets: list[DrawingView] = []
        failed = dict(render_report.projection_errors)

        def offer(
            name: str,
            role: View,
            path: Path,
            axes: tuple[Axis, Axis] | None = None,
        ) -> None:
            """Announce a drawing, or explain it: an unreadable one is not fatal."""
            try:
                sheets.append(register_view(_projected(name), role, path, axes))
            except Exception as why:  # noqa: BLE001 - report it where it would have been
                failed[name] = f"{type(why).__name__}: {why}"

        # A projection is written at 1:1 in model millimetres, which is the
        # frame a region measured on it is read in.
        assert self.interpretation is not None
        view_frames = self.interpretation.view_frames()
        for view, path in render_report.projection_paths.as_mapping().items():
            offer(view, View(view), path, view_frames[View(view)])
        if pictorial:
            offer(FEEDBACK_PICTORIAL, View.PERSPECTIVE, pictorial)
        if why := render_report.render3d_errors.get(FEEDBACK_PICTORIAL):
            failed[FEEDBACK_PICTORIAL] = why
        return FeedbackManifest(
            verification_id=verification_id,
            drawing=sheets,
            errors={_projected(name): why for name, why in failed.items()},
        )

    @property
    def confirmed(self) -> bool:
        """Whether the most recent `feedback` build implements the operations.

        False before the first build, so a program never built cannot pass for
        one that did.
        """
        report = self._last_feedback_report
        if report is None or report.exec_report is None or self._program_faults(report):
            return False
        return (
            report.exec_report.status is ExecutionStatus.VERIFIED
            and report.exec_report.returncode == 0
        )

    @property
    def accepted_source(self) -> str | None:
        """The program of the most recent `feedback` build, once confirmed."""
        report = self._last_feedback_report
        exec_report = report.exec_report if report is not None else None
        return (
            exec_report.source if exec_report is not None and self.confirmed else None
        )

    def feedback(self) -> list[ContentBlock]:
        """Verify, and say what happened in blocks a message can carry."""
        report = self.verify()
        self._last_feedback_report = report

        exec_report = report.exec_report
        render_report = report.render_report
        drawing_diff_report = report.drawing_diff_report
        assert exec_report is not None, (
            "shouldn't happen: verify() always returns an exec_report"
        )

        # create manifest
        final_render = (render_report or {}).get(RESULT_NAME)
        manifest = None
        if final_render is not None:
            assert report.verification_id is not None
            manifest = self._make_manifest(final_render, report.verification_id)

        # build feedback messages
        sandbox_source = self.workdir.sandbox_bind_dir / self.source_filename
        exec_report_dict = {
            "verification_id": report.verification_id,
            "status": exec_report.status.value,
            "returncode": exec_report.returncode,
            "stdout": exec_report.stdout,
            "stderr": exec_report.stderr,
            "executor_error": exec_report.executor_error,
            "shape": exec_report.census.describe() if exec_report.census else "",
        }
        blocks: list[ContentBlock] = [
            create_text_block(
                f"{sandbox_source} has been executed, and upon successful STEP "
                "file generation, its rendering images are exported:"
            ),
            create_text_block(json.dumps(exec_report_dict, indent=2)),
        ]

        # render report
        if exec_report.intermediate_returns:
            assert report.sandbox_verification_dir is not None

            intermediate_renders = {
                output.name: render_report[output.name]
                for output in exec_report.intermediate_returns
                if render_report and output.name in render_report
            }

            blocks.append(
                create_text_block(
                    describe_intermediates(
                        exec_report.intermediate_returns,
                        intermediate_renders,
                        PurePosixPath(report.sandbox_verification_dir)
                        / INTERMEDIATE_RETURNS_DIR,
                    )
                )
            )

        # drawing diff report
        if drawing_diff_text := describe_drawing_diffs(
            drawing_diff_report, self.workdir
        ):
            blocks.append(create_text_block(drawing_diff_text))

        # output paths and feedback manifest
        if manifest:
            # Offer the same ordinary projections in both A/B conditions, also
            # when automatic image attachments are disabled.
            projections = [
                self.workdir.host_to_sandbox_path(png)
                for sheet in manifest.drawing
                if (path := Path(sheet.file)).suffix.lower() == ".dxf"
                and (png := path.with_suffix(".png")).is_file()
            ]
            if projections:
                blocks.append(
                    create_text_block(
                        "[Orthographic PNGs]\nOpen these with `load_image` and compare "
                        "each view against the input drawing:\n"
                        + "\n".join(f"- {path}" for path in projections)
                    )
                )
            blocks.extend(
                build_feedback_message_blocks(
                    manifest,
                    self.workdir,
                    mode=self.feedback_presentation_mode,
                    heading="[Projected drawing]",
                )
            )
        if faults := self._program_faults(report):
            blocks.append(create_text_block("\n".join(faults)))
        return blocks

    def _program_faults(self, report: VerifyOutputResult) -> tuple[str, ...]:
        exec_report = report.exec_report
        if self.operations is None or exec_report is None or exec_report.source is None:
            return ()
        try:
            return check_program(exec_report.source, self.operations).faults
        except SyntaxError:
            # The build already reports it.
            return ()


def _projected(view: str) -> str:
    """Name a drawing of the built solid apart from the views of the input."""
    return f"view_projected_{view}"


def _validity(output: IntermediateReturn) -> str:
    """Flag a BRep that was not valid before export; its STEP may have been repaired."""
    if output.valid is None:
        return f"BRep validity unknown ({output.validity_reason or 'not checked'}); "
    if not output.valid:
        return f"BRep INVALID before export ({output.validity_reason}); "
    return ""


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
        head = f"{output.name:<{width}}  {_validity(output)}"
        if output.census is None:
            reason = (
                "STEP exported; shape census unavailable"
                if output.step_path is not None
                else f"not exported: {output.error}"
            )
            lines.append(head + reason)
            continue
        lines.append(
            head
            + (
                output.census.describe()
                if previous is None
                else output.census.describe_change_from(previous)
            )
        )
        previous = output.census
    return "\n".join(lines)


def describe_intermediates(
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
        f"- {name}: {reason}"
        for name, report in intermediate_renders.items()
        for reason in (
            *report.projection_errors.values(),
            *report.render3d_errors.values(),
        )
    ]

    msg = cleandoc(f"""
        [Intermediate results]
        Each line describes a `ret_*` solid in the program in a built order.
        The first line is the first solid built, and each line after it is the difference from the line above.
        Refer to these values for debugging.

        {_census_table(intermediate_returns)}

        Corresponding artifacts are saved in {sandbox_returns_dir}/<name>/ with the following files:
        output.step, projection/<view>.dxf, projection/<view>.png, render_3d/<style>.png.
        Inspect relevant views with load_image.
    """)
    if failures:
        msg += "\n\nRender failures:\n" + "\n".join(failures)

    return msg


def describe_drawing_diffs(
    diff_reports: Mapping[str, DrawingDiffReport] | None,
    workdir: SandboxWorkdir,
) -> str:
    """Format comparison results without executing comparisons or writing files."""
    if not diff_reports:
        return ""

    # The complete layout of one view.
    view_template = cleandoc("""
        {name}
        input: {input_path}
        projection: {projection_path}
        overlay: {overlay}
        residual: {residual}{notes}
    """)

    view_blocks: list[str] = []
    for name, report in diff_reports.items():
        # Preserve errors and warnings, including failures without images.
        notes = [f"error: {report.error}"] if report.error else []
        notes.extend(f"warning: {warning}" for warning in report.warnings)
        notes_text = "\n" + "\n".join(notes) if notes else ""

        view_blocks.append(
            view_template.format(
                name=name,
                input_path=workdir.host_to_sandbox_path(report.drawing_path),
                projection_path=(
                    workdir.host_to_sandbox_path(report.projection_path)
                    if report.projection_path is not None
                    else "unavailable"
                ),
                overlay=(
                    workdir.host_to_sandbox_path(report.paths.get("overlay_path"))
                    if report.paths.get("overlay_path") is not None
                    else "unavailable"
                ),
                residual=(
                    workdir.host_to_sandbox_path(report.paths.get("residual_path"))
                    if report.paths.get("residual_path") is not None
                    else "unavailable"
                ),
                notes=notes_text,
            )
        )

    # The complete message layout, with shared explanations stated once.
    return cleandoc("""
        [Drawing comparison]
        - input: original drawing crop for this view.
        - projection: orthographic line rendering of the generated STEP.

        For available comparison images, the input is transformed into
        projection pixel coordinates to help locate possible mismatches.
        Refer to the resulting images for refining your deliverables.

        - overlay: aligned input in pale gray; projection lines colored blue→red
          by increasing distance to the nearest aligned input line.
          Medium-gray projection lines fall outside the transformed input crop;
          their mismatch is not evaluated.
        - residual: aligned input lines, darker where farther from the nearest projection line.
          Dark regions can indicate missing or misplaced CAD features; drawing annotations also contribute.

        Note that alignment can be wrong or hide size errors.

        {views}

        Open images with load_image; confirm mismatches against originals.
    """).format(views="\n\n".join(view_blocks))
