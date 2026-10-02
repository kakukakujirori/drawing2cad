import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from inspect import cleandoc
from pathlib import Path, PurePosixPath
from statistics import fmean
from typing import Any, Literal

from langchain_core.messages.content import ContentBlock, create_text_block

from zeroshot.pipeline.messages.artifact import (
    ArtifactPresenter,
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
from zeroshot.pipeline.verification.render.orthographic import (
    STANDARD_VIEW_FRAMES,
    ViewFrames,
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
        artifact_presenter: ArtifactPresenter,
        attempt_store: AttemptStore,
        source_filename: str = "model.py",
        projection_view_mode: Literal["interpreted", "standard"] = "interpreted",
    ) -> None:
        source_path = PurePosixPath(source_filename)
        if (
            source_path.is_absolute()
            or len(source_path.parts) != 1
            or source_path.suffix != ".py"
        ):
            raise ValueError("source_filename must be a Python file basename")
        if projection_view_mode not in ("interpreted", "standard"):
            raise ValueError(f"unknown projection_view_mode: {projection_view_mode!r}")
        if projection_view_mode == "standard" and diff_drawer is not None:
            raise ValueError("diff_drawer requires projection_view_mode='interpreted'")
        self.executor = executor
        self.workdir = workdir
        self.renderer = renderer
        self.diff_drawer = diff_drawer
        self.artifact_presenter = artifact_presenter
        # The previous stages' deliverables must be available to the verifier
        self.interpretation: DrawingInterpretation | None = None
        self.operations: OperationPlan | None = None
        self.source_filename = source_filename
        self.attempt_store = attempt_store
        self.projection_view_mode = projection_view_mode

        self._last_feedback_report: VerifyOutputResult | None = None
        self._last_feedback_digest: str | None = None
        # The last two builds with a chamfer distance, so feedback shows the change.
        self._scored: VerifyOutputResult | None = None
        self._scored_before: VerifyOutputResult | None = None
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
        self._last_feedback_digest = None
        self._scored = None
        self._scored_before = None

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
                if self.artifact_presenter.intermediates != "none"
                else None
            ),
        )
        if cq_report.source is not None:
            output_model_path.write_text(
                cq_report.source,
                encoding="utf-8",
            )

        return cq_report

    def _projection_frames(self) -> ViewFrames:
        if self.projection_view_mode == "standard":
            return STANDARD_VIEW_FRAMES
        assert self.interpretation is not None
        return self.interpretation.view_frames()

    def _issue_render_request(
        self, step_path: Path, verification_dir: Path
    ) -> RenderRequest:
        """Name the feedback artifacts one STEP is to be drawn into."""
        # Flat: one file per view and per style, named as the inputs are, so
        # the model does not have to guess a second convention.
        view_frames = self._projection_frames()

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
            [(drawing, projection) for _, drawing, projection in selected],
            drawing_scales=[view.scale for view, _, _ in selected],
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

    @property
    def confirmed(self) -> bool:
        return not self.blockers

    @property
    def blockers(self) -> list[str]:
        """Why the file as it is now cannot be the answer, by its latest verification."""
        report = self._last_feedback_report
        if (
            report is None
            or report.exec_report is None
            or self.source_digest() != self._last_feedback_digest
        ):
            return [f"{self.source_filename} has not been verified as it is now"]
        exec_report = report.exec_report
        if exec_report.status is not ExecutionStatus.VERIFIED or exec_report.returncode:
            status = exec_report.status.value
            return [f"its build ended with status {status}; the build report says why"]
        return list(self._program_faults(report))

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
        self._last_feedback_digest = self.source_digest()
        if report is not self._scored and chamfers(report.drawing_diff_report):
            self._scored_before, self._scored = self._scored, report

        return build_verification_feedback(
            report,
            self.workdir,
            self.artifact_presenter,
            self._projection_frames() if report.render_report else {},
            previous=self._scored_before if report is self._scored else None,
        )

    def _program_faults(self, report: VerifyOutputResult) -> tuple[str, ...]:
        exec_report = report.exec_report
        if self.operations is None or exec_report is None or exec_report.source is None:
            return ()
        try:
            return check_program(exec_report.source, self.operations).faults
        except SyntaxError:
            # The build already reports it.
            return ()


def build_verification_feedback(
    report: VerifyOutputResult,
    workdir: SandboxWorkdir,
    presenter: ArtifactPresenter,
    view_frames: ViewFrames,
    *,
    previous: VerifyOutputResult | None = None,
) -> list[ContentBlock]:
    """Format saved verification reports in execution, render, comparison order.

    This owns the report text and its attachments together. Stage instructions
    and the HumanMessage carrying them belong to the caller.
    """
    exec_report = report.exec_report
    render_report = report.render_report or {}
    drawing_diff_report = report.drawing_diff_report

    # 1. Execution outcome, including failures that produced no STEP.
    blocks: list[ContentBlock] = [create_text_block("[Execution result]")]
    if exec_report is not None:
        blocks.append(
            create_text_block(
                json.dumps(
                    {
                        "verification_id": report.verification_id,
                        "status": exec_report.status.value,
                        "returncode": exec_report.returncode,
                        "stdout": exec_report.stdout,
                        "stderr": exec_report.stderr,
                        "executor_error": exec_report.executor_error,
                        "shape": exec_report.census.describe()
                        if exec_report.census
                        else "",
                    },
                    separators=(",", ":"),
                )
            )
        )
    else:
        blocks.append(create_text_block("No execution report was recorded."))

    # 2a. Final renders.
    if presenter.output_renders != "none" and RESULT_NAME in render_report:
        blocks.extend(
            build_feedback_message_blocks(
                _render_manifest(
                    render_report[RESULT_NAME],
                    report.verification_id or "",
                    view_frames,
                ),
                workdir,
                mode=presenter.output_renders,
            )
        )

    # 2b. Intermediate renders, including the intermediate census.
    if presenter.intermediates != "none" and exec_report:
        blocks.extend(
            describe_intermediates(
                exec_report.intermediate_returns,
                {
                    item.name: render_report[item.name]
                    for item in exec_report.intermediate_returns
                    if item.name in render_report
                },
                view_frames=view_frames,
                sandbox_workdir=workdir,
                sandbox_verification_dir=report.sandbox_verification_dir,
                verification_id=report.verification_id,
                presenter=presenter,
            )
        )

    # 3. Drawing comparison, including image legends and attachments.
    blocks.extend(
        describe_drawing_diffs(
            drawing_diff_report, workdir, previous, presenter=presenter
        )
    )
    return blocks


def _render_manifest(
    render_report: RenderReport, verification_id: str, view_frames: ViewFrames
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
    view_frames: ViewFrames,
    sandbox_workdir: SandboxWorkdir,
    sandbox_verification_dir: str | None,
    verification_id: str | None = None,
    *,
    presenter: ArtifactPresenter,
) -> list[ContentBlock]:
    """Say what every `ret_*` came out as, and where its views were written.

    One sentence for the layout rather than four paths per return: every
    directory holds the same file names, so listing them all would spend
    tokens on a convention the reader can apply once.
    """
    if not intermediate_returns:
        return []

    failures = [
        f"- {name}: {reason}"
        for name, report in intermediate_renders.items()
        for reason in (
            *report.projection_errors.values(),
            *report.render3d_errors.values(),
        )
    ]

    assert sandbox_verification_dir is not None, "intermediate renders need a sandbox dir"

    msg = cleandoc("""
        [Intermediate results]
        ret_* in execution order: first absolute, then changes vs last measured return.
        {table}
        Files are under {sandbox_returns_dir}/<name>/:
        output.step, projection/<view>.dxf, projection/<view>.png, render_3d/<style>.png.
    """).format(
        table=_census_table(intermediate_returns),
        sandbox_returns_dir=PurePosixPath(sandbox_verification_dir) / INTERMEDIATE_RETURNS_DIR,
    )
    if failures:
        msg += "\n\nRender failures:\n" + "\n".join(failures)

    blocks: list[ContentBlock] = [create_text_block(msg)]
    if presenter.intermediates == "image":
        for name in intermediate_renders:
            blocks.extend(
                build_feedback_message_blocks(
                    _render_manifest(
                        intermediate_renders[name],
                        verification_id or "",
                        view_frames,
                    ),
                    sandbox_workdir,
                    mode="image",
                    heading=f"[Intermediate {name}]",
                )
            )

    return blocks


def chamfers(diff_reports: Mapping[str, DrawingDiffReport] | None) -> dict[str, float]:
    """The chamfer distance of each view that could be measured."""
    return {
        name: chamfer
        for name, report in (diff_reports or {}).items()
        if (chamfer := report.stats.get("chamfer_drawing_px")) is not None
    }


def _view_unmatched(name: str, report: DrawingDiffReport) -> dict[str, dict[str, Any]]:
    return {
        f"drawing_diff.{name}.{number}": item
        for number, item in enumerate(report.stats.get("unmatched", []), 1)
    }


def unmatched_items(
    diff_reports: Mapping[str, DrawingDiffReport] | None,
) -> dict[str, dict[str, Any]]:
    """Each listed unmatched line group, keyed as the audit answers it."""
    return {
        key: item
        for name, report in (diff_reports or {}).items()
        for key, item in _view_unmatched(name, report).items()
    }


def _describe_unmatched(key: str, item: Mapping[str, Any]) -> str:
    unit = "skeleton pixels" if item["kind"] == "lines" else "px²"
    return (
        f"{key} ({item['color']}): {item['direction']} {item['kind']}, "
        f"{item['size_px']} {unit}, box {item['box_px']}"
    )


def _describe_chamfer(current: float, previous: float | None) -> str:
    change = f" ({current - previous:+.2f})" if previous is not None else ""
    return f"{current:.2f} px{change}"


def _describe_mean_chamfer(
    current: Mapping[str, float],
    previous: Mapping[str, float],
    previous_id: str | None,
) -> str:
    comparable = bool(previous) and previous.keys() == current.keys()
    text = f"Mean chamfer over {len(current)} views: " + _describe_chamfer(
        fmean(current.values()), fmean(previous.values()) if comparable else None
    )
    if previous:
        text += f"\nChanges in parentheses are against verification {previous_id}."
    return text


def describe_drawing_diffs(
    diff_reports: Mapping[str, DrawingDiffReport] | None,
    workdir: SandboxWorkdir,
    previous: VerifyOutputResult | None = None,
    *,
    presenter: ArtifactPresenter,
) -> list[ContentBlock]:
    """Build the complete comparison feedback: results, legends, paths and images.

    Scores are compared with `previous`, the build the reader saw before.
    """
    if not diff_reports:
        return []
    current_chamfer_by_view = chamfers(diff_reports)
    previous_chamfer_by_view = (
        chamfers(previous.drawing_diff_report) if previous else {}
    )

    view_chamfer_scores = []
    comparison_diagnostics = []
    for name, report in diff_reports.items():
        if name in current_chamfer_by_view:
            view_chamfer_scores.append(
                f"{name} chamfer: "
                + _describe_chamfer(
                    current_chamfer_by_view[name], previous_chamfer_by_view.get(name)
                )
            )
        if report.error:
            comparison_diagnostics.append(f"{name} error: {report.error}")
        comparison_diagnostics.extend(
            f"{name} warning: {warning}" for warning in report.warnings
        )

    mismatch_clusters = "\n".join(
        _describe_unmatched(key, item)
        for key, item in unmatched_items(diff_reports).items()
    )
    input_view_paths = "\n".join(
        f"{name} input: {workdir.host_to_sandbox_path(report.drawing_path)}"
        for name, report in diff_reports.items()
    )
    mean_chamfer_summary = (
        _describe_mean_chamfer(
            current_chamfer_by_view,
            previous_chamfer_by_view,
            previous.verification_id if previous else None,
        )
        if current_chamfer_by_view
        else ""
    )
    overlay_paths = {
        f"{name} overlay": report.paths["overlay_path"]
        for name, report in diff_reports.items()
        if "overlay_path" in report.paths
    }
    unmatched_paths = {
        f"{name} unmatched": report.paths["unmatched_path"]
        for name, report in diff_reports.items()
        if "unmatched_path" in report.paths
    }

    prompt = cleandoc("""
        [Drawing comparison]
        Projections were spatially aligned to DrawingViews, with differences highlighted in Overlay and Unmatched.
        Check Chamfer distances for quantitative deviation extent.
        NOTE: Alignment may be wrong or hide size errors.

        {comparison_diagnostics}

        Input views:
        {input_view_paths}

        Chamfer: mean line distance after alignment (input px); lower is better.
        {view_chamfer_scores}
        {mean_chamfer_summary}
    """).format(
        comparison_diagnostics=(
            "\n".join(["Comparison warnings/errors:", *comparison_diagnostics])
            if comparison_diagnostics
            else ""
        ),
        input_view_paths=input_view_paths,
        view_chamfer_scores="\n".join(view_chamfer_scores),
        mean_chamfer_summary=mean_chamfer_summary,
    )
    blocks: list[ContentBlock] = [create_text_block(prompt)]

    # overlay
    blocks += build_feedback_message_blocks(
        overlay_paths,
        workdir,
        mode=presenter.overlay,
        heading=cleandoc("""Overlay images:
        The input is moved onto the projection's pixels (pale gray).
        Blue=near, red=far from the input lines. Note that blue doesn't ensure correct matching, only that the input is near the projection.
        """)
    )

    # unmatched
    blocks += build_feedback_message_blocks(
        unmatched_paths,
        workdir,
        mode=presenter.unmatched,
        heading=cleandoc("""Unmatched images:
        The input is shown in gray, the projection in light blue, and their overlap in blue.
        Bands match cluster colors and ID suffix numbers; a material mismatch is also hatched.
        - Missing = input-only; extra = projection-only; material = silhouette difference.
        - Missing dimension/leader/text lines aren't defects.

        Mismatch clusters (bboxes in input-view pixels):
        {mismatch_clusters}
        """).format(mismatch_clusters=mismatch_clusters or "No clusters reported.")
    )
    return blocks