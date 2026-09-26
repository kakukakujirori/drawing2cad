"""Text shown to coder and auditor from the current in-memory reports."""

from pathlib import Path

import numpy as np
import pytest

from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.stages.coding.verify import describe_drawing_diffs
from zeroshot.pipeline.verification.drawing_diff.align import AlignmentResult
from zeroshot.pipeline.verification.run_drawing_diff import DrawingDiffReport


@pytest.mark.parametrize("reports", [None, {}])
def test_no_comparison_produces_no_message(tmp_path, reports):
    assert describe_drawing_diffs(reports, SandboxWorkdir(tmp_path)) == ""


def test_feedback_lists_sandbox_images_and_keeps_failure_reasons(tmp_path):
    workdir = SandboxWorkdir(tmp_path)
    reports = {
        "view_front": DrawingDiffReport(
            drawing_path=tmp_path / "inputs/front.png",
            projection_path=tmp_path / "projection/front.png",
            alignment=AlignmentResult(
                "directional_chamfer", "similarity", "uncertain", np.eye(3).tolist(), {}
            ),
            stats={"red_distance_px": 12, "output_to_input": {"p95_px": 5}},
            paths={
                "overlay_path": tmp_path / "projection/front_overlay.png",
                "residual_path": tmp_path / "projection/front_residual.png",
            },
            warnings=("ambiguous alignment",),
        ),
        "view_top": DrawingDiffReport(
            drawing_path=tmp_path / "inputs/top.png",
            projection_path=None,
            error="projection unavailable",
        ),
    }
    files_before = set(tmp_path.rglob("*"))

    text = describe_drawing_diffs(reports, workdir)

    assert (
        """view_front
input: /work/inputs/front.png
projection: /work/projection/front.png
overlay: /work/projection/front_overlay.png
residual: /work/projection/front_residual.png
warning: ambiguous alignment"""
        in text
    )
    assert (
        """view_top
input: /work/inputs/top.png
projection: unavailable
overlay: unavailable
residual: unavailable
error: projection unavailable"""
        in text
    )
    assert text.count("[Drawing comparison]") == 1
    assert "original drawing crop" in text
    assert "orthographic line rendering of the generated STEP" in text
    assert "annotations" in text
    assert "increasingly red" in text
    assert "residual: projection in pale gray" in text
    assert "Blue only means a nearby input line" in text
    assert text.lower().count("alignment is heuristic") == 1
    assert "Open available overlay and residual with load_image" in text
    assert "input drawing crop and original STEP projection" in text
    for unwanted in (str(tmp_path), "/work/unavailable", "p95_px", "red_distance_px"):
        assert unwanted not in text
    assert set(tmp_path.rglob("*")) == files_before


def test_failed_alignment_can_have_warnings_without_an_error_or_images(tmp_path):
    report = DrawingDiffReport(
        drawing_path=tmp_path / "crop.png",
        projection_path=tmp_path / "front.png",
        alignment=AlignmentResult(
            "directional_chamfer", "similarity", "failed", None, {}
        ),
        warnings=("Drawing residuals unavailable: no usable alignment",),
    )

    text = describe_drawing_diffs({"view_front": report}, SandboxWorkdir(tmp_path))

    assert "projection: /work/front.png" in text
    assert "overlay: unavailable\nresidual: unavailable" in text
    assert "warning: Drawing residuals unavailable: no usable alignment" in text
    assert "error:" not in text


@pytest.mark.parametrize("stage", ["coding", "audit"])
def test_fixed_instructions_do_not_require_optional_comparison_images(stage):
    stages = Path(__file__).parents[3] / "zeroshot/pipeline/stages"
    text = (stages / stage / "prompts/round.md").read_text()

    assert "load_image" in text
    assert "overlay" not in text
    assert "residual" not in text
    assert "distance scales" not in text
    assert "linked JSON" not in text


def test_coding_can_correct_geometry_without_overwriting_upstream_artifacts():
    stages = Path(__file__).parents[3] / "zeroshot/pipeline/stages"
    coding = (stages / "coding/prompts/round.md").read_text()

    assert "source drawing takes precedence" in coding
    assert "Leave upstream JSON files unchanged" in coding
    assert "state the unresolved mismatch" in coding
