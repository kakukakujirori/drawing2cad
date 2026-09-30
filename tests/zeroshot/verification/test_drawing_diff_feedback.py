"""Text shown to coder and auditor from the current in-memory reports."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from zeroshot.pipeline.messages.artifact import ArtifactPresenter
from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.stages.coding.verify import (
    VerifyOutputResult,
    describe_drawing_diffs,
    unmatched_items,
)
from zeroshot.pipeline.verification.drawing_diff.align import AlignmentResult
from zeroshot.pipeline.verification.run_drawing_diff import DrawingDiffReport

PRESENTER = ArtifactPresenter(input="path")


def _text(blocks):
    return "\n".join(block["text"] for block in blocks if block["type"] == "text")


@pytest.mark.parametrize("reports", [None, {}])
def test_no_comparison_produces_no_message(tmp_path, reports):
    assert (
        describe_drawing_diffs(reports, SandboxWorkdir(tmp_path), presenter=PRESENTER)
        == []
    )


def test_feedback_lists_input_and_keeps_failure_reasons(tmp_path):
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
            },
            warnings=("ambiguous alignment",),
        ),
        "view_top": DrawingDiffReport(
            drawing_path=tmp_path / "inputs/top.png",
            projection_path=None,
            error="projection unavailable",
        ),
    }
    overlay = reports["view_front"].paths["overlay_path"]
    overlay.parent.mkdir(parents=True)
    Image.new("RGB", (4, 4), "white").save(overlay)
    files_before = set(tmp_path.rglob("*"))

    text = _text(describe_drawing_diffs(reports, workdir, presenter=PRESENTER))

    assert "view_front input: /work/inputs/front.png" in text
    assert "view_front warning: ambiguous alignment" in text
    assert "view_top input: /work/inputs/top.png" in text
    assert "view_top error: projection unavailable" in text
    assert text.count("[Drawing comparison]") == 1
    assert "Alignment may be wrong or hide size errors" in text
    assert "Overlay images:" in text
    assert "view_front overlay: /work/projection/front_overlay.png" in text
    assert "Blue=near, red=far" in text
    assert "blue doesn't ensure correct matching" in text
    assert "Unmatched images:" not in text
    assert "Mismatch clusters" not in text
    for unwanted in (str(tmp_path), "/work/unavailable", "p95_px", "red_distance_px"):
        assert unwanted not in text
    assert set(tmp_path.rglob("*")) == files_before


def _scored(tmp_path, chamfers):
    return {
        name: DrawingDiffReport(
            drawing_path=tmp_path / f"{name}.png",
            projection_path=tmp_path / f"projected_{name}.png",
            stats={"chamfer_drawing_px": chamfer},
        )
        for name, chamfer in chamfers.items()
    }


def test_scores_show_changes_against_the_previous_build(tmp_path):
    previous = VerifyOutputResult(
        verification_id="002",
        drawing_diff_report=_scored(tmp_path, {"view_front": 8.0, "view_top": 4.0}),
    )
    current = _scored(tmp_path, {"view_front": 6.0, "view_top": 5.0})

    text = _text(
        describe_drawing_diffs(
            current, SandboxWorkdir(tmp_path), previous, presenter=PRESENTER
        )
    )

    assert "chamfer: 6.00 px (-2.00)" in text
    assert "Mean chamfer over 2 views: 5.50 px (-0.50)" in text
    assert "against verification 002" in text


def test_only_views_measured_both_times_are_compared(tmp_path):
    previous = VerifyOutputResult(
        verification_id="002",
        drawing_diff_report=_scored(tmp_path, {"view_front": 8.0}),
    )
    current = _scored(tmp_path, {"view_front": 6.0, "view_top": 5.0})

    text = _text(
        describe_drawing_diffs(
            current, SandboxWorkdir(tmp_path), previous, presenter=PRESENTER
        )
    )

    assert "chamfer: 6.00 px (-2.00)" in text
    assert "view_top chamfer: 5.00 px" in text.splitlines()
    assert "Mean chamfer over 2 views: 5.50 px" in text.splitlines()


@pytest.mark.parametrize("mode", ["path", "image"])
def test_unmatched_groups_are_listed_under_their_view_by_audit_key(tmp_path, mode):
    group = {
        "direction": "missing",
        "kind": "lines",
        "size_px": 361,
        "box_px": [465, 358, 673, 440],
    }
    reports = {
        "view_top": DrawingDiffReport(
            drawing_path=tmp_path / "top.png",
            projection_path=tmp_path / "projection/top.png",
            stats={
                "chamfer_drawing_px": 6.0,
                "unmatched": [
                    group | {"color": "red"},
                    group | {"direction": "extra", "color": "yellow"},
                    group
                    | {"kind": "material", "direction": "extra", "color": "purple"},
                ],
            },
            paths={"unmatched_path": tmp_path / "projection/top_unmatched.png"},
        ),
    }

    unmatched = reports["view_top"].paths["unmatched_path"]
    unmatched.parent.mkdir(parents=True)
    Image.new("RGB", (700, 500), "white").save(unmatched)
    blocks = describe_drawing_diffs(
        reports,
        SandboxWorkdir(tmp_path),
        presenter=ArtifactPresenter(input="path", unmatched=mode),
    )
    text = _text(blocks)

    assert "view_top input: /work/top.png" in text
    assert "view_top chamfer: 6.00 px" in text
    assert "Mismatch clusters (bboxes in input-view pixels):" in text
    assert (
        "Missing = input-only; extra = projection-only; material = silhouette difference"
        in text
    )
    assert "Missing dimension/leader/text lines aren't defects" in text
    assert (
        "The input is shown in gray, the projection in light blue, and their overlap in blue"
        in text
    )
    assert "Bands match cluster colors and ID suffix numbers" in text
    assert "a material mismatch is also hatched" in text
    assert (
        """drawing_diff.view_top.1 (red): missing lines, 361 skeleton pixels, box [465, 358, 673, 440]
drawing_diff.view_top.2 (yellow): extra lines, 361 skeleton pixels, box [465, 358, 673, 440]
drawing_diff.view_top.3 (purple): extra material, 361 px², box [465, 358, 673, 440]"""
        in text
    )
    label = next(
        i
        for i, block in enumerate(blocks)
        if block["type"] == "text" and block["text"].startswith("view_top unmatched:")
    )
    assert text.index("drawing_diff.view_top.1") < text.index("view_top unmatched:")
    assert sum(block["type"] == "image" for block in blocks) == int(mode == "image")
    if mode == "image":
        assert blocks[label + 1]["type"] == "image"
    assert list(unmatched_items(reports)) == [
        "drawing_diff.view_top.1",
        "drawing_diff.view_top.2",
        "drawing_diff.view_top.3",
    ]


def test_failed_alignment_can_have_warnings_without_an_error_or_images(tmp_path):
    report = DrawingDiffReport(
        drawing_path=tmp_path / "crop.png",
        projection_path=tmp_path / "front.png",
        alignment=AlignmentResult(
            "directional_chamfer", "similarity", "failed", None, {}
        ),
        warnings=("Drawing comparison unavailable: no usable alignment",),
    )

    text = _text(
        describe_drawing_diffs(
            {"view_front": report}, SandboxWorkdir(tmp_path), presenter=PRESENTER
        )
    )

    assert "input: /work/crop.png" in text
    assert "/work/front.png" not in text
    assert "warning: Drawing comparison unavailable: no usable alignment" in text
    assert "error:" not in text
    assert "Overlay images:" not in text
    assert "Unmatched images:" not in text


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
    assert "Leaving the feature out is one of these hypotheses" in coding
