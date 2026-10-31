"""Text shown to coder and auditor from the current in-memory reports."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from zeroshot.pipeline.messages.artifact import ArtifactPresenter
from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.stages.coding.verify import (
    VerifyOutputResult,
    build_verification_feedback,
    describe_drawing_diffs,
    unmatched_items,
)
from zeroshot.pipeline.stages.interpretation.contracts import View
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

    front, top = text.split("view_top (", 1)
    top = "view_top (" + top
    assert "/work/inputs/front.png" in front
    assert "warning: ambiguous alignment" in front
    assert "/work/inputs/top.png" in top
    assert "error: projection unavailable" in top
    assert text.count("[Drawing comparison]") == 1
    assert "Alignment log:" not in text
    assert "Alignment may be wrong or hide size errors" in text
    assert "[Overlay images]" in text
    assert "view_front overlay: /work/projection/front_overlay.png" in text
    assert "blue near input; red far away" in text
    assert "Blue does not prove a correct match" in text
    assert "[Unmatched images]" not in text
    assert "Mismatch clusters" not in text
    for unwanted in (str(tmp_path), "/work/unavailable", "p95_px", "red_distance_px"):
        assert unwanted not in text
    assert set(tmp_path.rglob("*")) == files_before


def _scored(tmp_path, chamfers):
    return {
        name: DrawingDiffReport(
            drawing_path=tmp_path / f"{name}.png",
            projection_path=tmp_path / f"projected_{name}.png",
            stats={"bounded_chamfer_drawing_px": chamfer},
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

    assert "line distance: 6.00 px (-2.00)" in text
    assert "Mean line distance over 2 views: 5.50 px (-0.50)" in text
    assert "against verification 002" in text
    assert "warning" not in text
    assert "Alignment log:" not in text


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

    assert "line distance: 6.00 px (-2.00)" in text
    assert (
        "view_top (/work/view_top.png)\n- line distance: 5.00 px\n- silhouette" in text
    )
    assert "Mean line distance over 2 views: 5.50 px" in text.splitlines()


@pytest.mark.parametrize("mode", ["path", "image"])
def test_unmatched_groups_are_listed_under_their_view_by_audit_key(tmp_path, mode):
    group = {
        "direction": "missing",
        "kind": "lines",
        "unit": "px",
        "size_px": 361,
        "box_px": [465, 358, 673, 440],
    }
    reports = {
        "view_top": DrawingDiffReport(
            drawing_path=tmp_path / "top.png",
            projection_path=tmp_path / "projection/top.png",
            stats={
                "bounded_chamfer_drawing_px": 6.0,
                "unmatched": [
                    group | {"color": "red"},
                    group
                    | {
                        "direction": "extra",
                        "color": "yellow",
                    },
                    group
                    | {
                        "kind": "material",
                        "unit": "px²",
                        "direction": "extra",
                        "color": "purple",
                    },
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

    assert "view_top (/work/top.png)\n- line distance: 6.00 px" in text
    assert "[Mismatch clusters — bboxes in input-view pixels]" in text
    assert "Missing: input only. Extra: projection only" in text
    assert "Missing dimension, leader and text lines are not defects" in text
    assert "Input: gray; projection: light blue; overlap: blue" in text
    assert "Colored bands and IDs: mismatch clusters" in text
    assert "hatching: silhouette differences" in text
    assert (
        """- drawing_diff.view_top.1 (red): missing lines, 361 px
  bbox [465, 358, 673, 440]
- drawing_diff.view_top.2 (yellow): extra lines, 361 px
  bbox [465, 358, 673, 440]
- drawing_diff.view_top.3 (purple): extra material, 361 px²
  bbox [465, 358, 673, 440]"""
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

    assert "/work/crop.png" in text
    assert "/work/front.png" not in text
    assert "warning: Drawing comparison unavailable: no usable alignment" in text
    assert "error:" not in text
    assert "[Overlay images]" not in text
    assert "[Unmatched images]" not in text


@pytest.mark.parametrize("stage", ["coding", "audit"])
def test_fixed_instructions_do_not_require_optional_comparison_images(stage):
    stages = Path(__file__).parents[3] / "zeroshot/pipeline/stages"
    text = (stages / stage / "prompts/round.md").read_text()

    assert "load_image" in text
    assert "overlay" not in text
    assert "residual" not in text
    assert "distance scales" not in text
    assert "linked JSON" not in text


def test_coding_corrects_its_reading_with_the_drawing_as_authority():
    stages = Path(__file__).parents[3] / "zeroshot/pipeline/stages"
    coding = (stages / "coding/prompts/round.md").read_text()

    assert "The source drawing is the authority" in coding
    assert "correct the interpretation as well as the program" in coding
    assert "Leaving the feature out is one of these hypotheses" in coding


def test_region_error_is_explained_and_combined_scores_stay_internal(tmp_path):
    reports = _scored(tmp_path, {"view_front": 5.0, "view_top": 6.0})
    for name, ratio, score in (("view_front", 0.1, 0.35), ("view_top", 0.2, 0.5)):
        reports[name].stats.update(
            {
                "material_error": {
                    "status": "ok",
                    "ratio": ratio,
                    "missing_px2": 0,
                    "extra_px2": 1000,
                },
                "match_score": score,
            }
        )

    text = _text(
        describe_drawing_diffs(reports, SandboxWorkdir(tmp_path), presenter=PRESENTER)
    )

    assert "silhouette mismatch: 10.00% (missing 0 px², extra 1000 px²)" in text
    assert "silhouette mismatch: 20.00%" in text
    assert "holes and inner lines are excluded" in text
    assert "S=" not in text and "C/20" not in text
    assert "match score 0.35" not in text and "match score 0.50" not in text
    assert reports["view_front"].stats["match_score"] == 0.35
    assert "Best" not in text and "improved" not in text

    reports["view_front"].stats["material_error"].update(
        {"ratio": 0.000001, "extra_px2": 1}
    )
    text = _text(
        describe_drawing_diffs(reports, SandboxWorkdir(tmp_path), presenter=PRESENTER)
    )
    assert "silhouette mismatch: <0.01%" in text

    reports["view_top"].stats.update(
        {
            "material_error": {
                "status": "unavailable",
                "ratio": None,
                "reason": "input part area unavailable",
            },
            "match_score": None,
        }
    )
    text = _text(
        describe_drawing_diffs(reports, SandboxWorkdir(tmp_path), presenter=PRESENTER)
    )
    assert "silhouette mismatch: unavailable: input part area unavailable" in text
    assert "match score 0.35" not in text and "match score 0.50" not in text


def test_shared_feedback_uses_registered_signed_axes_for_arbitrary_view_names(tmp_path):
    path = tmp_path / "right_unmatched.png"
    Image.new("RGB", (4, 4), "white").save(path)
    report = DrawingDiffReport(
        drawing_path=tmp_path / "crop.png",
        projection_path=tmp_path / "right.png",
        stats={
            "model_uv_mapping": {"status": "provisional"},
            "unmatched": [
                {
                    "direction": "extra",
                    "kind": "material",
                    "size_px": 2000,
                    "unit": "px²",
                    "color": "red",
                    "box_px": [10, 20, 30, 40],
                    "box_model_uv": [6, 10, 8, 14],
                }
            ],
        },
        paths={"unmatched_path": path},
    )

    text = _text(
        build_verification_feedback(
            VerifyOutputResult(drawing_diff_report={"view_rotated_profile": report}),
            SandboxWorkdir(tmp_path),
            PRESENTER,
            {View.RIGHT: ("-z", "+y")},
        )
    )

    assert "U=-Z, V=+Y, X depth hidden" in text
    assert "CAD bounds: Z≈-8.00..-6.00, Y≈10.00..14.00 mm" in text
    assert "alignment uncertain" in text


def test_a_view_without_scale_says_its_size_was_not_compared(tmp_path):
    report = DrawingDiffReport(
        drawing_path=tmp_path / "crop.png",
        projection_path=tmp_path / "front.png",
        alignment=AlignmentResult(
            backend="directional_chamfer",
            model="similarity",
            status="ok",
            H_drawing_to_projection=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            diagnostics={
                "scale_calibration": {
                    "status": "unavailable",
                    "reason": "input scale unavailable",
                }
            },
        ),
        stats={"model_uv_mapping": {"status": "provisional"}},
    )

    text = _text(
        describe_drawing_diffs(
            {"view_front": report}, SandboxWorkdir(tmp_path), presenter=PRESENTER
        )
    )

    assert "size not compared: input scale unavailable" in text
    assert "alignment uncertain" not in text


def test_unavailable_cad_coordinates_keep_the_reason_in_shared_feedback(tmp_path):
    report = DrawingDiffReport(
        drawing_path=tmp_path / "crop.png",
        projection_path=tmp_path / "front.png",
        stats={
            "model_uv_mapping": {
                "status": "unavailable",
                "reason": "projection DXF unavailable",
            }
        },
    )
    text = _text(
        describe_drawing_diffs(
            {"view_front": report}, SandboxWorkdir(tmp_path), presenter=PRESENTER
        )
    )

    assert "CAD coordinates unavailable: projection DXF unavailable" in text


@pytest.mark.parametrize(
    "input_style,output_style,shown",
    [
        ("visible", "hidden", "input solid / CAD dashed"),
        ("hidden", "visible", "input dashed / CAD solid"),
    ],
)
def test_line_style_lengths_and_cluster_styles_are_reported(
    tmp_path, input_style, output_style, shown
):
    reports = _with_style_lengths(tmp_path)
    previous_reports = _with_style_lengths(tmp_path, mismatch=200)
    reports["front"].stats["line_style"]["mismatch_count"] = 2
    reports["front"].stats["unmatched"] = [
        {
            "direction": "line_style",
            "kind": "lines",
            "input_style": input_style,
            "output_style": output_style,
            "size_px": 30,
            "unit": "px",
            "color": "red",
            "box_px": [10, 10, 20, 20],
        }
    ]
    path = tmp_path / "unmatched.png"
    Image.new("RGB", (4, 4), "white").save(path)
    reports["front"].paths["unmatched_path"] = path
    text = _text(
        describe_drawing_diffs(
            reports,
            SandboxWorkdir(tmp_path),
            VerifyOutputResult(
                verification_id="001", drawing_diff_report=previous_reports
            ),
            presenter=PRESENTER,
        )
    )
    assert "solid/dashed mismatch length: ≈126 px (-74 vs previous)" in text
    assert f"drawing_diff.front.1 (red): {shown}\n  bbox [10, 10, 20, 20]" in text
    assert "Solid/dashed mismatch: overlapping lines with different styles." in text
    assert "1 labeled" not in text and "2 labeled" not in text
    assert "Check images:" not in text


def _with_style_lengths(tmp_path, mismatch=126, checked=840, unknown=210):
    reports = _scored(tmp_path, {"front": 0.0})
    reports["front"].stats["line_style"] = {
        "status": "ok",
        "mismatch_count": int(mismatch > 0),
        "displayed_count": int(mismatch > 0),
        "checked_curves": int(checked > 0),
        "unknown_curves": int(unknown > 0),
        "mismatch_length_px": mismatch,
        "checked_length_px": checked,
        "unknown_length_px": unknown,
    }
    return reports


def test_first_style_summary_shows_only_the_mismatch_length(tmp_path):
    text = _text(
        describe_drawing_diffs(
            _with_style_lengths(tmp_path),
            SandboxWorkdir(tmp_path),
            presenter=PRESENTER,
        )
    )
    assert "solid/dashed mismatch length: ≈126 px" in text
    assert "total length of detected mismatches, in input pixels" in text
    assert "excluded from the geometric score" in text
    assert "840" not in text and "210" not in text
    assert "classified" not in text and "unknown" not in text
    assert "vs previous" not in text


@pytest.mark.parametrize("checked,unknown", [(756, 210), (840, 294)])
def test_reduced_mismatches_and_changed_coverage_are_not_reported_as_a_fix(
    tmp_path, checked, unknown
):
    previous = VerifyOutputResult(
        verification_id="001",
        drawing_diff_report=_with_style_lengths(tmp_path),
    )
    text = _text(
        describe_drawing_diffs(
            _with_style_lengths(
                tmp_path, mismatch=42, checked=checked, unknown=unknown
            ),
            SandboxWorkdir(tmp_path),
            previous,
            presenter=PRESENTER,
        )
    )
    assert "≈42 px (-84 vs previous)" in text
    assert "Check images: this decrease may reflect unassessable lines" in text


def test_unclassified_view_does_not_claim_zero_mismatches(tmp_path):
    previous = VerifyOutputResult(
        verification_id="001",
        drawing_diff_report=_with_style_lengths(tmp_path),
    )
    text = _text(
        describe_drawing_diffs(
            _with_style_lengths(tmp_path, mismatch=0, checked=0, unknown=1050),
            SandboxWorkdir(tmp_path),
            previous,
            presenter=PRESENTER,
        )
    )
    assert "solid/dashed comparison unavailable" in text
    assert "solid/dashed mismatch length: ≈0" not in text


@pytest.mark.parametrize("incomparable", ["different_crop", "provisional", "legacy"])
def test_style_length_deltas_require_comparable_measurements(tmp_path, incomparable):
    reports = _with_style_lengths(tmp_path)
    before = _with_style_lengths(tmp_path, mismatch=200)
    if incomparable == "different_crop":
        before["front"] = replace(before["front"], drawing_path=tmp_path / "other.png")
    elif incomparable == "provisional":
        before["front"].stats["provisional"] = True
    else:
        before["front"].stats["line_style"] = {
            "status": "ok",
            "mismatch_count": 1,
        }
    text = _text(
        describe_drawing_diffs(
            reports,
            SandboxWorkdir(tmp_path),
            VerifyOutputResult(verification_id="001", drawing_diff_report=before),
            presenter=PRESENTER,
        )
    )
    assert "≈126 px" in text
    assert "vs previous" not in text
