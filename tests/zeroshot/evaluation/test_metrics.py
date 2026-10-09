"""STEP-level checks without pipeline imports or benchmark-data dependencies."""

import json
from dataclasses import replace

import cadquery as cq
import numpy as np
import pytest

from zeroshot.evaluation.metrics.eccv import score_eccv
from zeroshot.evaluation.metrics.itercad import aggregate_itercad
from zeroshot.evaluation.metrics.ortho2cad import score_ortho2cad
from zeroshot.evaluation.metrics.surface_distance import score_surface_distance
from zeroshot.evaluation.preprocess import (
    ORIENTATIONS,
    AlignmentError,
    EvaluationError,
    InvalidPredictionError,
    PreprocessConfig,
    prepare_pair,
    read_solid,
    shape_bbox,
)
from zeroshot.evaluation.score_pair import score_pair


def _step(tmp_path, name, shape):
    path = tmp_path / f"{name}.step"
    shape.exportStep(str(path))
    return path


def _asymmetric_shape():
    return (
        cq.Workplane("XY")
        .box(30, 20, 10)
        .faces(">Z")
        .workplane()
        .center(7, 3)
        .hole(4)
        .val()
    )


def test_24_rotation_alignment_preserves_geometry_and_analytic_faces(tmp_path):
    shape = _asymmetric_shape()
    gt = _step(tmp_path, "gt", shape)
    pred = _step(
        tmp_path, "pred", shape.rotate((0, 0, 0), (0, 1, 0), 90).translate((13, -7, 4))
    )
    pair = prepare_pair(pred, gt)
    assert len(ORIENTATIONS) == len(pair.metadata["orientation_ious"]) == 24
    assert all(np.linalg.det(rotation) == pytest.approx(1) for rotation in ORIENTATIONS)
    assert pair.iou == pytest.approx(1, abs=1e-8)
    assert pair.iou >= max(pair.metadata["orientation_ious"]) - 1e-10
    assert max(shape_bbox(pair.gt_shape)[1]) == pytest.approx(1.8)
    assert shape_bbox(pair.pred_shape)[0] == pytest.approx([0, 0, 0], abs=1e-7)
    assert sorted(face.geomType() for face in pair.pred_shape.Faces()) == sorted(
        face.geomType() for face in shape.Faces()
    )
    scores = score_eccv(pair)
    for kind in ["surface", "edge", "vertex", "topology"]:
        assert scores[f"eccv_{kind}_f1"] == pytest.approx(1, abs=2e-6)


def test_eccv_takes_the_best_of_the_iou_tied_poses(tmp_path):
    shape = cq.Workplane("XY").box(30, 20, 10).faces(">Z").workplane().hole(4).val()
    gt = _step(tmp_path, "symmetric_hole", shape)
    pair = prepare_pair(gt, gt)
    # The box with a centred hole maps onto itself under four rotations.
    assert len(pair.tied_shapes) == 4
    assert any(np.allclose(ORIENTATIONS[i], np.eye(3)) for i in pair.tied_shapes)
    scores = score_eccv(pair)
    singles = [
        score_eccv(replace(pair, tied_shapes={i: shape}))["eccv_mean_f1"]
        for i, shape in pair.tied_shapes.items()
    ]
    assert scores["eccv_mean_f1"] == max(singles)
    assert scores["eccv_mean_f1"] == pytest.approx(1, abs=2e-6)
    assert scores["eccv_rotation_index"] in pair.tied_shapes


def test_shared_scaling_keeps_dimension_errors_even_below_one_mm(tmp_path):
    distances = []
    for unit_scale in [0.001, 1, 1000]:
        gt_shape = cq.Workplane("XY").box(30, 20, 10).val().scale(unit_scale)
        pred_shape = gt_shape.scale(2).translate((3 * unit_scale, 8 * unit_scale, 0))
        gt = _step(tmp_path, f"gt_{unit_scale}", gt_shape)
        pred = _step(tmp_path, f"pred_{unit_scale}", pred_shape)
        pair = prepare_pair(pred, gt)
        assert pair.iou == pytest.approx(1 / 8)
        assert max(shape_bbox(pair.gt_shape)[1]) == pytest.approx(1.8)
        assert max(shape_bbox(pair.pred_shape)[1]) == pytest.approx(3.6)
        scores = score_surface_distance(pair, sample_points=2048, seed=12)
        distances.append(scores["chamfer_diag"])
    assert distances == pytest.approx([distances[0]] * 3, rel=1e-6)
    assert summarize_auc(distances) == pytest.approx([summarize_auc(distances)[0]] * 3)


def summarize_auc(distances):
    return [
        aggregate_itercad([{"build_valid": True, "metrics": {"chamfer_diag": cd}}])[
            "auc_tr"
        ]
        for cd in distances
    ]


def test_reference_extent_is_configurable_and_distances_are_squared(tmp_path):
    gt = _step(tmp_path, "gt", cq.Workplane("XY").box(30, 20, 10).val())
    pred = _step(tmp_path, "pred", cq.Workplane("XY").box(30, 20, 12).val())
    pair1 = prepare_pair(pred, gt, PreprocessConfig(reference_extent=1.8))
    pair2 = prepare_pair(pred, gt, PreprocessConfig(reference_extent=3.6))
    assert pair1.iou == pytest.approx(pair2.iou)
    cd1 = score_surface_distance(pair1, seed=11)
    cd2 = score_surface_distance(pair2, seed=11)
    assert cd2["chamfer"] == pytest.approx(4 * cd1["chamfer"])
    assert cd2["hausdorff"] == pytest.approx(4 * cd1["hausdorff"])
    assert cd2["chamfer_diag"] == pytest.approx(cd1["chamfer_diag"])


def test_multipart_overlap_uses_union_and_cavity_is_preserved(tmp_path):
    left = cq.Workplane("XY").box(2, 2, 2).val()
    right = left.translate((1, 0, 0))
    compound = cq.Compound.makeCompound([left, right])
    pred = _step(tmp_path, "overlapping_parts", compound)
    gt = _step(tmp_path, "union", left.fuse(right))
    assert prepare_pair(pred, gt).iou == pytest.approx(1)
    shell = cq.Workplane("XY").box(2, 2, 2).cut(cq.Workplane("XY").box(1, 1, 1)).val()
    solid_gt = _step(tmp_path, "solid", left)
    hollow_pred = _step(tmp_path, "hollow", shell)
    assert prepare_pair(hollow_pred, solid_gt).iou == pytest.approx(7 / 8)


def test_eccv_does_not_erase_pred_size_error(tmp_path):
    gt = _step(tmp_path, "gt", cq.Workplane("XY").box(30, 20, 10).val())
    pred = _step(tmp_path, "pred", cq.Workplane("XY").box(60, 40, 20).val())
    scores = score_eccv(prepare_pair(pred, gt))
    assert scores["eccv_mean_f1"] < 0.5
    # The literature comparator deliberately independently normalizes both.
    assert score_ortho2cad(pred, gt)["ortho2cad_iou"] == pytest.approx(1)


def test_ortho2cad_official_inertia_alignment_and_self_iou(tmp_path):
    shape = cq.Workplane("XY").box(30, 20, 10).val()
    gt = _step(tmp_path, "gt", shape)
    pred = _step(
        tmp_path,
        "pred",
        shape.scale(3).rotate((0, 0, 0), (1, 2, 3), 37).translate((21, -5, 11)),
    )
    assert score_ortho2cad(gt, gt)["ortho2cad_iou"] == pytest.approx(1)
    assert score_ortho2cad(pred, gt)["ortho2cad_iou"] == pytest.approx(1, abs=1e-6)
    wrong = _step(tmp_path, "wrong", cq.Workplane("XY").box(30, 20, 15).val())
    # Golden value from upstream cq_align_shapes in its pinned CadQuery 2.5.2.
    assert score_ortho2cad(wrong, gt)["ortho2cad_iou"] == pytest.approx(
        0.6550950128826313, abs=1e-8
    )


def test_missing_or_invalid_prediction_differs_from_invalid_gt(tmp_path):
    gt = _step(tmp_path, "gt", cq.Workplane("XY").box(30, 20, 10).val())
    missing = tmp_path / "missing.step"
    with pytest.raises(InvalidPredictionError):
        prepare_pair(missing, gt)
    invalid = tmp_path / "invalid.step"
    invalid.write_text("not STEP")
    with pytest.raises(InvalidPredictionError):
        prepare_pair(invalid, gt)
    with pytest.raises(EvaluationError):
        prepare_pair(gt, invalid)
    report = score_pair(missing, gt)
    assert report["build_valid"] is False
    assert report["errors"]["prediction"]["kind"] == "generation_failure"
    report = score_pair(gt, missing)
    assert report["build_valid"] is None
    assert report["errors"]["preprocess"]["kind"] == "evaluation_error"


def test_alignment_failure_never_scores_unaligned_distances(tmp_path, monkeypatch):
    gt = _step(tmp_path, "gt", cq.Workplane("XY").box(30, 20, 10).val())

    def broken_alignment(*args):
        raise AlignmentError("test Boolean failure")

    monkeypatch.setattr(
        "zeroshot.evaluation.preprocess.orientation_ious", broken_alignment
    )
    report = score_pair(gt, gt)
    assert report["status"] == "PARTIAL"
    # Ortho2CAD does not depend on the shared alignment.
    assert report["metrics"].keys() == {"ortho2cad_iou"}
    assert report["errors"]["alignment"]["kind"] == "evaluation_error"
    assert aggregate_itercad([report])["auc_tr"] is None


def test_native_pose_error_is_an_alignment_error(tmp_path, monkeypatch):
    gt = _step(tmp_path, "gt", cq.Workplane("XY").box(30, 20, 10).val())

    def failed_boolean(*args):
        raise ValueError("native intersection failed")

    monkeypatch.setattr("zeroshot.evaluation.preprocess.manifold_iou", failed_boolean)
    with pytest.raises(AlignmentError):
        prepare_pair(gt, gt)


def test_metric_failure_keeps_valid_cd_and_its_error_kind(tmp_path, monkeypatch):
    gt = _step(tmp_path, "gt", cq.Workplane("XY").box(30, 20, 10).val())

    def failed_eccv(*args, **kwargs):
        raise RuntimeError("sampling failed")

    monkeypatch.setattr("zeroshot.evaluation.metrics.eccv.score_eccv", failed_eccv)
    report = score_pair(gt, gt, include_ortho2cad=False)
    assert report["status"] == "PARTIAL"
    assert report["build_valid"] is True
    assert report["metrics"]["chamfer"] >= 0
    assert report["errors"]["eccv"]["kind"] == "evaluation_error"
    assert aggregate_itercad([report])["auc_tr"] is not None


def test_standalone_report_is_deterministic_and_json_serializable(tmp_path):
    gt = _step(tmp_path, "gt", cq.Workplane("XY").box(30, 20, 10).val())
    options = {"include_eccv": False, "include_ortho2cad": False, "sample_points": 2048}
    report = score_pair(gt, gt, **options)
    assert report == score_pair(gt, gt, **options)
    assert report["status"] == "OK"
    assert report["metrics"]["mesh_iou"] == pytest.approx(1)
    # Independent surface samples are deliberately not forced to zero CD.
    assert report["metrics"]["chamfer"] > 0
    json.dumps(report, allow_nan=False)


def test_eccv_leads_with_the_mean_of_the_four_official_axes(solids):
    scores = score_eccv(prepare_pair(solids["sphere"], solids["box"]))
    axes = ("surface", "edge", "vertex", "topology")
    assert next(iter(scores)) == "eccv_mean_f1"
    assert scores["eccv_mean_f1"] == pytest.approx(
        sum(scores[f"eccv_{axis}_f1"] for axis in axes) / 4
    )
    assert scores["eccv_mean_f1"] < 0.9


def test_bbox_is_not_fooled_by_a_spline_control_hull(solids):
    """An untriangulated ``Add`` bounds splines by their hull and shrinks the scale."""
    assert max(shape_bbox(read_solid(solids["loft"]))[1]) == pytest.approx(
        20.0, rel=1e-3
    )
