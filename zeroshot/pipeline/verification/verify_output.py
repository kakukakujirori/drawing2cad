import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Literal, Self, cast

from langchain_core.messages.content import ContentBlock, create_text_block

from zeroshot.pipeline.messages import (
    DrawingSource,
    FeedbackManifest,
    View,
    unread_sheet,
)
from zeroshot.pipeline.messages.artifact import build_feedback_message_blocks
from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.verification._run_program import INTERMEDIATE_RETURNS_DIR
from zeroshot.pipeline.verification.attempts import AttemptStore
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
from zeroshot.pipeline.verification.run_render import (
    RenderReport,
    RenderRequest,
    StepRenderer,
)
from zeroshot.pipeline.verification.shape_census import ShapeCensus

VerifyOutputValue: type = str | int | None

# Build outcomes a second attempt at the same bytes could come out of
# differently, because they turn on how loaded the machine was.
_TRANSIENT_OUTCOMES = frozenset({ExecutionStatus.TIMEOUT, ExecutionStatus.INFRA_ERROR})

# The renderer draws three styles of the one perspective, and only this one is
# offered: it is the line art of `hlg_perspective` over a faint copy of the
# shaded pass, so it says which side is material as well as where the edges
# are. Three pictures of one camera would spend a message saying it three times.
FEEDBACK_PICTORIAL = "hlg_translucent_faces_perspective"


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
            *report.projection_errors.values(),
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
                "projection/<view>.dxf with a .png of it alongside, and "
                "render_3d/<style>.png. Open them with `run_shell` and "
                "`load_image` to see whether an operation built what it was "
                "meant to."
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
        feedback_presentation_mode: Literal["none", "path", "image"],
        attempt_store: AttemptStore,
        views: Sequence[View] = (),
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
        self.feedback_presentation_mode = feedback_presentation_mode
        # The orthographic views to redraw the solid in. A caller that reads a
        # drawing sets this per round; nothing is drawn until one does, because
        # a guessed view has nothing to be compared against.
        self.views: Sequence[View] = tuple(views)
        self.source_filename = source_filename
        self.attempt_store = attempt_store
        self.show_intermediate_returns = show_intermediate_returns

        self._last_feedback_report: VerifyOutputResult | None = None
        # What the last build returned, and the digest of the program it ran
        # on. Assigned together, so one is never read against the other. See
        # `verify`.
        self._built: tuple[VerifyOutputResult, FeedbackManifest | None] | None = None
        self._built_from_digest: str | None = None

    @property
    def source_path(self) -> Path:
        """The program this verifier builds, on the host side of the sandbox."""
        return self.workdir.host_bind_dir / self.source_filename

    def _source_digest(self) -> str | None:
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

    def verify(self) -> tuple[VerifyOutputResult, FeedbackManifest | None]:
        """Verify the program and, when it yields a solid, render its views.

        A program byte-identical to the one the last build ran on is not built
        again: it would land in a second attempt directory holding the same
        STEP and the same twenty renders. The coder's turn is checked by
        middleware, and the same program reaches the graph's own verification a
        moment later, so without this every coding stage paid for its build
        twice. A transient outcome is the exception: it is not kept, and the
        same bytes are built again.

        The manifest stays out of the report because it carries host paths, and
        only the report is msgpack-serialisable enough to reach graph state.
        """
        digest = self._source_digest()
        if self._built is not None and digest == self._built_from_digest:
            return self._built

        report, manifest = self._build()
        if digest is not None and report.status not in _TRANSIENT_OUTCOMES:
            self._built, self._built_from_digest = (report, manifest), digest
        return report, manifest

    def _build(self) -> tuple[VerifyOutputResult, FeedbackManifest | None]:
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
        verification_id, host_verification_dir, sandbox_verification_dir = (
            self.attempt_store.issue("coding")
        )
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

        # Draw and describe every ret_xxx the program left behind, and the
        # result beside them. A program that ran leaves the returns whether or
        # not `result` passed, and a result that failed is when they are most
        # worth reading; a result that passed adds one more drawing of the same
        # kind, so all of them are drawn in one batch.
        host_returns_dir = host_verification_dir / INTERMEDIATE_RETURNS_DIR
        built = [
            (output.name, output.step_path)
            for output in cq_report.intermediate_returns
            if output.step_path is not None
        ]
        result_built = (
            report.status == ExecutionStatus.VERIFIED and report.returncode == 0
        )
        requests = [
            self._request(step_path, host_returns_dir / name)
            for name, step_path in built
        ]
        if result_built:
            requests.append(self._request(output_step_path, host_verification_dir))

        results = self.renderer.render_many(requests)
        renders = {name: result for (name, _), result in zip(built, results)}
        report = replace(
            report,
            intermediate_returns=_describe_returns(
                cq_report.intermediate_returns,
                renders,
                sandbox_verification_dir / INTERMEDIATE_RETURNS_DIR,
            ),
        )

        # A result that did not build has no STEP of its own to draw.
        if not result_built:
            return report, None

        render_report = results[-1]
        # The projected drawing is announced the way the input was: one sheet
        # per view, and one pictorial. A view is named by its role, which is
        # also the field the renderer wrote it under.
        pictorial = render_report.render3d_paths.as_mapping().get(FEEDBACK_PICTORIAL)
        sheets = [
            *(
                unread_sheet(f"sheet_{view}", View(view), path)
                for view, path in render_report.projection_paths.as_mapping().items()
            ),
            *(
                [
                    unread_sheet(
                        f"sheet_{FEEDBACK_PICTORIAL}", View.PERSPECTIVE, pictorial
                    )
                ]
                if pictorial
                else []
            ),
        ]
        failed = dict(render_report.projection_errors)
        if why := render_report.render3d_errors.get(FEEDBACK_PICTORIAL):
            failed[FEEDBACK_PICTORIAL] = why
        manifest = FeedbackManifest(
            verification_id=verification_id,
            drawing=DrawingSource(sheets=sheets) if sheets else None,
            errors={f"sheet_{name}": why for name, why in failed.items()},
        )
        return report, manifest

    def _request(self, step_path: Path, verification_dir: Path) -> RenderRequest:
        """Name the feedback artifacts one STEP is to be drawn into."""
        # Flat: one file per view and per style, named as the inputs are, so
        # the model does not have to guess a second convention.
        projection_paths = ProjectionPaths.flat(
            verification_dir / "projection", self.views
        )
        render3d_paths = Render3dPaths.flat(verification_dir / "render_3d")
        # The renderer leaves directory layout to its caller.
        for path in (
            *projection_paths.as_mapping().values(),
            *render3d_paths.as_mapping().values(),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)

        return RenderRequest(step_path, projection_paths, render3d_paths)

    @property
    def confirmed(self) -> bool:
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
        if manifest:
            blocks.extend(
                build_feedback_message_blocks(
                    manifest,
                    self.workdir,
                    mode=self.feedback_presentation_mode,
                    heading="[Projected drawing]",
                )
            )
        return blocks
