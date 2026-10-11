"""A trial is built, compared and scored as the program is, and never submitted."""

import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import pytest
from PIL import Image

from tests.zeroshot.contracts import interpretation, view
from zeroshot.pipeline.sandbox import SandboxRunner, SandboxWorkdir
from zeroshot.pipeline.stages.coding.verify import TrialVerifier, VerifyOutputResult
from zeroshot.pipeline.tools.cad_hypothesis import (
    TRIALS_DIRNAME,
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
        )

    monkeypatch.setattr(CadQueryExecutor, "execute", execute)
    monkeypatch.setattr(
        TrialVerifier, "_draw_diffs", lambda self, render: _diffs(tmp_path, 0.7)
    )
    workdir = SandboxWorkdir(tmp_path)
    for role in ROLES:
        Image.new("RGB", (4, 4), "white").save(tmp_path / f"{role}_unmatched.png")
    (built := tmp_path / "built.step").write_text("STEP")
    reference = VerifyOutputResult(
        sandbox_verification_dir="/work/attempts/round_000/coding/003",
        exec_report=CadQueryExecutionReport(
            status=ExecutionStatus.VERIFIED, returncode=0, step_path=built
        ),
        drawing_diff_report=_diffs(tmp_path, 7.0),
    )

    def make(stage: AttemptStage = "coding", views=_views):
        return workdir, create_cad_hypothesis_tool(
            workdir=workdir,
            sandbox_runner=SandboxRunner(Path(sys.executable), 10),
            renderer=_Renderer(),  # type: ignore[arg-type]
            diff_drawer_config=None,
            trial_store=AttemptStore(workdir, lambda: 0, root_dirname=TRIALS_DIRNAME),
            stage=stage,
            interpretation=views,
            reference=lambda: reference,
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
    assert "Match score: 0.70 px; lower is better." in first
    assert "Build /work/attempts/round_000/coding/003: match score 7.00 px." in first
    # Its clusters are not the drawing_diff items an audit answers.
    assert "trial_diff.view_front.1" in first and "drawing_diff." not in first
    assert f"round_000/{stage}/001/hypothesis.py" in second
    assert (root / f"trials/round_000/{stage}/000/hypothesis.py").read_text() == (
        "result = 'box'\n"
    )
    assert (root / "model.py").read_text() == "result = 'program'\n"
    assert not (root / "attempts").exists()
    assert PurePosixPath("trials") in workdir.read_only_subdirs


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
