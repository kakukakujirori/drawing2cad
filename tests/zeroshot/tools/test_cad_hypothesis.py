"""A trial is built, compared and scored as the program is, and never submitted."""

import re
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path, PurePosixPath

import pytest
from PIL import Image

from tests.zeroshot.contracts import interpretation, view
from zeroshot.pipeline.sandbox import SandboxRunner, SandboxWorkdir
from zeroshot.pipeline.stages.coding.verify import TrialVerifier, VerifyOutputResult
from zeroshot.pipeline.tools.cad_hypothesis import (
    TRIALS_DIRNAME,
    _comparison_feedback,
    create_cad_hypothesis_tool,
)
from zeroshot.pipeline.tools.errors import ToolFeedbackError
from zeroshot.pipeline.verification import AttemptStore
from zeroshot.pipeline.verification.attempts import AttemptStage
from zeroshot.pipeline.verification.drawing_diff.align import AlignmentResult
from zeroshot.pipeline.verification.render.constants import (
    ProjectionPaths,
    Render3dPaths,
)
from zeroshot.pipeline.verification.run_cadquery import (
    CadQueryExecutionReport,
    CadQueryExecutor,
    ExecutionStatus,
)
from zeroshot.pipeline.verification.run_drawing_diff import DrawingDiffReport
from zeroshot.pipeline.verification.run_render import RenderReport, RenderStatus
from zeroshot.pipeline.verification.shape_census import ShapeCensus, read_census

ROLES = ("front", "top", "right")
EXTRA = {
    "direction": "extra",
    "kind": "lines",
    "unit": "px",
    "size_px": 60,
    "box_px": [1, 2, 3, 4],
    "color": "red",
}


def _views():
    return interpretation(
        views=[view(role, scale=0.1, file=f"inputs/{role}.png") for role in ROLES]
    )


def _diffs(root: Path, score: float) -> dict[str, DrawingDiffReport]:
    return {
        f"view_{role}": DrawingDiffReport(
            drawing_path=root / "inputs" / f"{role}.png",
            projection_path=None,
            alignment=AlignmentResult(
                backend="directional_chamfer",
                model="similarity",
                status="ok",
                H_drawing_to_projection=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                diagnostics={"scale_calibration": {"status": "applied"}},
            ),
            paths={"unmatched_path": root / f"{role}_unmatched.png"},
            stats={
                "match_score": score,
                "bounded_chamfer_drawing_px": score,
                "comparison_status": "ok",
                "provisional": False,
                "outside_count": 0,
                "material_error": {"status": "ok"},
                "unmatched": [EXTRA],
            },
        )
        for role in ROLES
    }


def _census(warnings=()):
    return ShapeCensus(
        solids=1,
        volume=1,
        extent=(1, 1, 1),
        faces=Counter(),
        edges=Counter(),
        warnings=warnings,
    )


def _scored_report(root: Path, name: str, score: float, warning_count: int | None):
    step = root / f"{name}.step"
    step.write_text("STEP")
    return VerifyOutputResult(
        sandbox_verification_dir=f"/work/{name}",
        exec_report=CadQueryExecutionReport(
            status=ExecutionStatus.VERIFIED,
            returncode=0,
            step_path=step,
            census=(
                _census(tuple(f"warning {i}" for i in range(warning_count)))
                if warning_count is not None
                else None
            ),
        ),
        drawing_diff_report=_diffs(root, score / len(ROLES)),
    )


class _Renderer:
    def render_many(self, requests):
        return [
            RenderReport(RenderStatus.OK, ProjectionPaths(), Render3dPaths())
            for _ in requests
        ]


@pytest.fixture
def trial(tmp_path, monkeypatch):
    """A tool whose build executes nothing: the script's text is the program."""

    def execute(self, source, step):
        step.write_text("STEP")
        return CadQueryExecutionReport(
            status=ExecutionStatus.VERIFIED,
            source=source.read_text(),
            returncode=0,
            step_path=step,
            census=_census(),
        )

    monkeypatch.setattr(CadQueryExecutor, "execute", execute)
    real_draw_diffs = TrialVerifier._draw_diffs
    # With a comparator, every trial compares as if scored 0.7 px per view.
    monkeypatch.setattr(
        TrialVerifier,
        "_draw_diffs",
        lambda self, render: (
            _diffs(tmp_path, 0.7)
            if self.diff_drawer is not None
            else real_draw_diffs(self, render)
        ),
    )
    workdir = SandboxWorkdir(tmp_path)
    for role in ROLES:
        Image.new("RGB", (4, 4), "white").save(tmp_path / f"{role}_unmatched.png")
    (built := tmp_path / "built.step").write_text("STEP")
    reference = VerifyOutputResult(
        sandbox_verification_dir="/work/attempts/round_000/coding/003",
        exec_report=CadQueryExecutionReport(
            status=ExecutionStatus.VERIFIED,
            returncode=0,
            step_path=built,
            census=_census(),
        ),
        drawing_diff_report=_diffs(tmp_path, 7.0),
    )

    def make(stage: AttemptStage = "coding", views=_views, compares=True, margin=0.5):
        return workdir, create_cad_hypothesis_tool(
            workdir=workdir,
            sandbox_runner=SandboxRunner(Path(sys.executable), 10),
            renderer=_Renderer(),  # type: ignore[arg-type]
            # Never run: `_draw_diffs` above stands in for it.
            diff_drawer_config={} if compares else None,
            trial_store=AttemptStore(workdir, lambda: 0, root_dirname=TRIALS_DIRNAME),
            stage=stage,
            interpretation=views,
            reference=lambda: reference,
            match_margin_px=margin,
        )

    return make


@pytest.mark.parametrize("stage", ["coding", "audit"])
def test_a_trial_is_scored_against_a_reference_and_kept_apart(trial, stage):
    workdir, tool = trial(stage)
    root = workdir.host_bind_dir
    (root / "trial.py").write_text("result = 'box'\n")
    (root / "model.py").write_text("result = 'program'\n")

    first = tool.invoke({"source_path": "/work/trial.py"})
    second = tool.invoke({"source_path": "/work/trial.py"})

    kept = f"/work/trials/round_000/{stage}/000/hypothesis.py"
    assert first.startswith(f"[Trial build]\nScript: /work/trial.py, kept as {kept}")
    assert "Match score: 2.10 px; lower is better." in first
    assert "Build /work/attempts/round_000/coding/003: match score 21.00 px." in first
    assert "Projection match improved by 18.90 px" in first
    assert "No geometry warnings were reported." in first
    assert "Open the trial projections" in first
    assert ("update the main program" in first) == (stage == "coding")
    # Its clusters are not the drawing_diff items an audit answers.
    assert "trial_diff.view_front.1" in first and "drawing_diff." not in first
    assert f"round_000/{stage}/001/hypothesis.py" in second
    assert (root / f"trials/round_000/{stage}/000/hypothesis.py").read_text() == (
        "result = 'box'\n"
    )
    assert (root / "model.py").read_text() == "result = 'program'\n"
    assert not (root / "attempts").exists()
    assert PurePosixPath("trials") in workdir.read_only_subdirs


@pytest.mark.parametrize(
    "reference_score,score,reference_warnings,warnings,margin,encouraged",
    [
        (5, 4, 0, 0, 0.5, True),
        (5, 4, 2, 0, 0.5, True),
        (5, 4, 2, 1, 0.5, True),
        (5, 4, 1, 1, 0.5, False),
        (5, 4, 1, 2, 0.5, False),
        (5, 4, 0, 1, 0.5, False),
        (5, 6, 2, 0, 0.5, False),  # An omitted feature can worsen a partial trial.
        (5, 5, 1, 0, 0.5, False),
        (5, 4.6, 1, 0, 0.5, False),
        (5, 4.5, 1, 0, 0.5, True),
        (5, 4.5, 1, 0, 1.0, False),
        (5, 5, 0, 0, 0.0, False),
        (5, 4, None, 0, 0.5, True),
        (5, 4, None, 1, 0.5, False),
        (5, 4, 0, None, 0.5, False),
    ],
)
def test_only_qualifying_improvements_get_advice(
    tmp_path, reference_score, score, reference_warnings, warnings, margin, encouraged
):
    workdir = SandboxWorkdir(tmp_path)
    baseline = _scored_report(tmp_path, "baseline", reference_score, reference_warnings)
    candidate = _scored_report(tmp_path, "candidate", score, warnings)
    text = "\n".join(
        _comparison_feedback(
            candidate,
            baseline,
            _views(),
            workdir,
            stage="coding",
            match_margin_px=margin,
        )
    )

    assert f"Match score: {score:.2f} px" in text
    assert f"Build /work/baseline: match score {reference_score:.2f} px." in text
    assert ("Projection match improved" in text) == encouraged
    assert ("Open the trial projections" in text) == encouraged
    assert ("For a whole-model trial" in text) == encouraged
    assert "worse" not in text.lower() and "previous was better" not in text.lower()
    if encouraged and warnings:
        assert f"decreased from {reference_warnings} to {warnings}" in text
        assert "Review the remaining geometry warnings." in text
    elif encouraged:
        assert "No geometry warnings were reported." in text


@pytest.mark.parametrize("reference_warnings", [0, 2])
def test_warning_check_failure_does_not_encourage_a_trial(
    tmp_path, monkeypatch, reference_warnings
):
    import cadquery as cq

    from zeroshot.pipeline.verification import shape_warnings

    workdir = SandboxWorkdir(tmp_path)
    baseline = _scored_report(tmp_path, "baseline", 5, reference_warnings)
    candidate = _scored_report(tmp_path, "candidate", 4, 0)
    cq.exporters.export(
        cq.Solid.makeBox(10, 20, 30), str(candidate.exec_report.step_path)
    )

    def broken_review(*args, **kwargs):
        raise RuntimeError("Review failed")

    monkeypatch.setattr(shape_warnings, "find_shape_warnings", broken_review)
    census = read_census(candidate.exec_report.step_path)
    assert census is not None and census.volume == pytest.approx(6000)
    candidate = replace(
        candidate, exec_report=replace(candidate.exec_report, census=census)
    )

    text = "\n".join(
        _comparison_feedback(
            candidate, baseline, _views(), workdir, stage="coding", match_margin_px=0.5
        )
    )

    assert "Match score: 4.00 px" in text
    assert "Projection match improved" not in text
    assert "No geometry warnings were reported." not in text
    assert "update the main program" not in text


@pytest.mark.parametrize("reference_state", ["absent", "unavailable"])
def test_an_unscored_reference_does_not_encourage_a_trial(tmp_path, reference_state):
    workdir = SandboxWorkdir(tmp_path)
    candidate = _scored_report(tmp_path, "candidate", 1, 0)
    baseline = None
    if reference_state == "unavailable":
        baseline = _scored_report(tmp_path, "baseline", 5, 0)
        baseline.drawing_diff_report["view_front"].stats["provisional"] = True
    text = "\n".join(
        _comparison_feedback(
            candidate, baseline, _views(), workdir, stage="coding", match_margin_px=0.5
        )
    )

    assert "Projection match improved" not in text
    assert "Open the trial projections" not in text


@pytest.mark.parametrize("margin", [-1, float("nan"), float("inf")])
def test_invalid_improvement_margins_are_rejected(trial, margin):
    with pytest.raises(ValueError, match="match_margin_px"):
        trial(margin=margin)


def test_without_drawing_comparison_a_trial_is_drawn_and_claims_no_score(trial):
    workdir, tool = trial(compares=False)
    (workdir.host_bind_dir / "trial.py").write_text("result = 'box'\n")

    result = tool.invoke({"source_path": "/work/trial.py"})

    kept = "/work/trials/round_000/coding/000/hypothesis.py"
    assert result.startswith(
        f"[Trial build]\nScript: /work/trial.py, kept as {kept}\n[Execution result]"
    )
    for claim in ("score", "compar", "trial_diff", "cluster"):
        assert claim not in result.lower()
        assert claim not in tool.description.lower()
    _, compared = trial()
    assert tool.description != compared.description
    assert "is drawn in\nthe views the interpretation registers." in tool.description


def test_a_trial_needs_registered_views_and_a_workspace_script(trial):
    workdir, unregistered = trial(views=lambda: None)
    (workdir.host_bind_dir / "trial.py").write_text("result = 'box'\n")
    with pytest.raises(ToolFeedbackError, match="has not validated yet"):
        unregistered.invoke({"source_path": "/work/trial.py"})

    _, tool = trial()
    for path in ("/work/absent.py", "/etc/hostname"):
        with pytest.raises(ToolFeedbackError, match="Cannot read workspace file"):
            tool.invoke({"source_path": path})


def test_parallel_trials_build_their_own_scripts(trial, monkeypatch):
    workdir, tool = trial()
    for name in "ab":
        (workdir.host_bind_dir / f"{name}.py").write_text(f"result = '{name}'\n")
    # Both calls reach the build before either runs it, as parallel calls can.
    barrier = threading.Barrier(2)
    original = AttemptStore.issue

    def issue(self, stage):
        barrier.wait()
        return original(self, stage)

    monkeypatch.setattr(AttemptStore, "issue", issue)

    with ThreadPoolExecutor(2) as pool:
        results = pool.map(
            tool.invoke, [{"source_path": f"/work/{n}.py"} for n in "ab"]
        )

    for name, text in zip("ab", results, strict=True):
        kept = re.search(r"kept as /work/(\S+)", text)
        assert kept and (workdir.host_bind_dir / kept[1]).read_text() == (
            f"result = '{name}'\n"
        )
