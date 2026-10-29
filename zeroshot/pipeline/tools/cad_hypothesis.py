"""Build a trial script and compare it with the drawing, as a build of the program is."""

from collections.abc import Callable
from functools import partial
from inspect import cleandoc
from pathlib import Path, PurePosixPath
from typing import Annotated, Any

from langchain_core.tools import BaseTool, tool

from zeroshot.pipeline.messages.artifact import ArtifactPresenter
from zeroshot.pipeline.sandbox import SandboxRunner, SandboxWorkdir
from zeroshot.pipeline.stages.coding.progress import match_score
from zeroshot.pipeline.stages.coding.verify import (
    TrialVerifier,
    VerifyOutputResult,
    build_verification_feedback,
)
from zeroshot.pipeline.stages.interpretation.contracts import DrawingInterpretation
from zeroshot.pipeline.tools.errors import ToolFeedbackError
from zeroshot.pipeline.verification import (
    AttemptStore,
    CadQueryExecutor,
    DrawingDiffExecutor,
    StepRenderer,
)
from zeroshot.pipeline.verification.attempts import AttemptStage

TRIALS_DIRNAME = PurePosixPath("trials")
_TRIAL_SOURCE = "hypothesis.py"

# Trial output never carries images: the model opens the ones it needs.
_PATHS_ONLY = ArtifactPresenter(input="path")

_PURPOSE = cleandoc(
    """This tool packages the pipeline's automatic program execution and rendering;
    you can test other programs without changing the main program to be submitted.
    This also means that a trial never reaches the submission: if you are responsible
    for the `coding` stage, remember to update the main program once a trial confirms
    a promising alternative."""
)

_DESCRIPTION = cleandoc(
    """Build a trial CadQuery script and compare it with the input drawing.

    {purpose}

    The script needs to store its trial shape in `result` and must not read
    workspace files. Like a build of the main program, the trial is drawn in
    the views the interpretation registers, compared with them and given a
    match score. The result shows that score together with the match score of
    the latest build of the main program.

    A trial may hold the whole model, such as a copy of the program with one
    change, or only the part in question with enough surrounding geometry to
    check its extent and connections. A whole-model trial's score is directly
    comparable with that of the main program. A partial trial also gets a
    score, but everything it omits counts as mismatch; ignore that score and
    inspect its projections around the part.

    Each call writes a new directory under {root} with the script (saved as
    {source}), its STEP, projections and comparison images. The result lists
    their paths; open PNGs with load_image or read DXFs with ezdxf. Audit
    evidence may cite a trial's projections; the coder can read the script
    that trial built there.

    Trial-specific mismatch clusters are named trial_diff.*; they are not the
    drawing_diff items of a build.
    """
)
# Without a drawing comparison a trial is only drawn, and says no more.
_DRAWN_DESCRIPTION = cleandoc(
    """Build a trial CadQuery script and draw it as the main program is drawn.

    {purpose}

    The script needs to store its trial shape in `result` and must not read
    workspace files. Like a build of the main program, the trial is drawn in
    the views the interpretation registers.

    A trial may hold the whole model, such as a copy of the program with one
    change, or only the part in question with enough surrounding geometry to
    check its extent and connections.

    Each call writes a new directory under {root} with the script (saved as
    {source}), its STEP and projections. The result lists their paths; open
    PNGs with load_image or read DXFs with ezdxf. Audit evidence may cite a
    trial's projections; the coder can read the script that trial built there.
    """
)

_RESULT = cleandoc(
    """[Trial build]
    Script: {script}, kept as {kept}"""
)
_SCORE = "Match score: {score:.2f} px; lower is better."
_UNSCORED = "Match score unavailable: {reason}."


def create_cad_hypothesis_tool(
    *,
    workdir: SandboxWorkdir,
    sandbox_runner: SandboxRunner,
    renderer: StepRenderer,
    diff_drawer_config: dict[str, Any] | None,
    trial_store: AttemptStore,
    stage: AttemptStage,
    interpretation: Callable[[], DrawingInterpretation | None],
    reference: Callable[[], VerifyOutputResult | None],
) -> BaseTool:
    """`reference` gives the build a trial is weighed against."""
    trial_verifier = partial(
        TrialVerifier,
        executor=CadQueryExecutor(sandbox_runner=sandbox_runner),
        workdir=workdir,
        renderer=renderer,
        diff_drawer=(
            DrawingDiffExecutor(**diff_drawer_config)
            if diff_drawer_config is not None
            else None
        ),
        artifact_presenter=_PATHS_ONLY,
        attempt_store=trial_store,
        source_filename=_TRIAL_SOURCE,
        attempt_stage=stage,
    )
    # Only a comparison with the drawing gives a trial its score.
    compares_with_drawing = diff_drawer_config is not None
    description = (
        _DESCRIPTION if compares_with_drawing else _DRAWN_DESCRIPTION
    ).format(purpose=_PURPOSE, source=_TRIAL_SOURCE, root=trial_store.sandbox_root)

    @tool("evaluate_cad_hypothesis", description=description)
    def evaluate_cad_hypothesis(
        source_path: Annotated[str, "A CadQuery script in the workspace."],
    ) -> str:
        views = interpretation()
        if views is None:
            raise ToolFeedbackError(
                "Trials are drawn in the views the interpretation registers; "
                "it has not validated yet."
            )
        script = _workspace_file(workdir, source_path)
        # A verifier per call, so parallel calls never share a script.
        report = trial_verifier(script, views).verify()
        score, reason = match_score(report, views, workdir)
        assert report.sandbox_verification_dir is not None
        lines = build_verification_feedback(
            report,
            workdir,
            _PATHS_ONLY,
            views.view_frames() if report.render_report else {},
            cluster_prefix="trial_diff",
        )
        return "\n".join(
            [
                _RESULT.format(
                    script=source_path,
                    kept=PurePosixPath(report.sandbox_verification_dir) / _TRIAL_SOURCE,
                ),
                *(
                    [
                        _SCORE.format(score=score)
                        if score is not None
                        else _UNSCORED.format(reason=reason),
                        _reference(reference(), views, workdir),
                    ]
                    if compares_with_drawing
                    else []
                ),
                *(block["text"] for block in lines if block["type"] == "text"),
            ]
        )

    return evaluate_cad_hypothesis


def _reference(
    report: VerifyOutputResult | None,
    views: DrawingInterpretation,
    workdir: SandboxWorkdir,
) -> str:
    if report is None or report.sandbox_verification_dir is None:
        return "No build to compare with yet."
    score, reason = match_score(report, views, workdir)
    shown = f"{score:.2f} px" if score is not None else f"unavailable ({reason})"
    return f"Build {report.sandbox_verification_dir}: match score {shown}."


def _workspace_file(workdir: SandboxWorkdir, sandbox_path: str) -> Path:
    try:
        path = workdir.sandbox_to_host_path(sandbox_path).resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        raise ToolFeedbackError(f"Cannot read workspace file: {sandbox_path}") from None
    if not path.is_relative_to(workdir.host_bind_dir.resolve()) or not path.is_file():
        raise ToolFeedbackError(f"Cannot read workspace file: {sandbox_path}")
    return path
