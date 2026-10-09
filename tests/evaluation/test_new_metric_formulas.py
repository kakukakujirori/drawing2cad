"""Analytic checks of the new metric definitions and failure accounting."""

import numpy as np
import pytest
import trimesh

from zeroshot.evaluation.metrics.itercad import aggregate_itercad, summarize_cds
from zeroshot.evaluation.metrics.mesh_iou import mesh_iou, to_manifold
from zeroshot.evaluation.metrics.surface_distance import point_distances
from zeroshot.evaluation.preprocess import PreprocessConfig, close_sliver_holes


def test_squared_distances_have_no_chamfer_half_factor():
    scores = point_distances(np.array([[0, 0, 0]]), np.array([[3, 4, 0]]))
    assert scores == {"chamfer": 50.0, "hausdorff": 25.0}
    # Asymmetric nearest neighbours distinguish Hausdorff from a mean.
    scores = point_distances(np.array([[0, 0, 0], [2, 0, 0]]), np.array([[0, 0, 0]]))
    assert scores == {"chamfer": 2.0, "hausdorff": 4.0}
    assert point_distances(np.zeros((1, 3)), np.zeros((1, 3)))["chamfer"] == 0


@pytest.mark.parametrize("bad", [np.empty((0, 3)), np.zeros((2, 2)), [[np.nan, 0, 0]]])
def test_distances_reject_invalid_clouds(bad):
    with pytest.raises(ValueError):
        point_distances(bad, np.zeros((1, 3)))


def test_mesh_iou_analytic_overlap_and_disjoint():
    box = trimesh.creation.box(extents=(2, 2, 2))
    assert mesh_iou(box, box) == pytest.approx(1)
    shifted = box.copy().apply_translation((1, 0, 0))
    assert mesh_iou(box, shifted) == pytest.approx(1 / 3)
    disjoint = box.copy().apply_translation((3, 0, 0))
    assert mesh_iou(box, disjoint) == 0
    larger = box.copy().apply_scale(2)
    assert mesh_iou(box, larger) == pytest.approx(1 / 8)


def test_sliver_holes_close_and_real_holes_stay_open():
    box = trimesh.creation.box()
    # Split one triangle at an edge midpoint, leaving a collinear three-edge hole.
    a, b, c = box.faces[0]
    m = len(box.vertices)
    vertices = np.vstack([box.vertices, box.vertices[[a, b]].mean(axis=0)])
    faces = np.vstack([box.faces[1:], [[a, m, c], [m, b, c]]])
    split = trimesh.Trimesh(vertices, faces, process=False)
    assert to_manifold(close_sliver_holes(split)).volume() == pytest.approx(1)
    holed = trimesh.Trimesh(box.vertices, box.faces[1:], process=False)
    assert close_sliver_holes(holed) is holed
    with pytest.raises(ValueError, match="Manifold"):
        to_manifold(holed)


def test_mesh_iou_rejects_open_surfaces():
    triangle = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]]
    )
    with pytest.raises(ValueError, match="Manifold"):
        mesh_iou(triangle, trimesh.creation.box())


def test_auc_needs_all_valid_and_all_cd_below_strictest_threshold():
    assert summarize_cds([0, 1e-5])["auc_tr"] == pytest.approx(1)
    assert summarize_cds([0, None])["auc_tr"] == pytest.approx(0.5)
    assert summarize_cds([1, 2])["auc_tr"] == 0  # both builds valid, both inaccurate
    summary = summarize_cds([1e-3, 2e-3, None, np.inf, np.nan, -1])
    assert summary["valid_cd_samples"] == 2
    assert summary["invalid_predictions"] == 4
    assert summary["mean_cd"] == pytest.approx(1.5e-3)
    assert summary["median_cd"] == pytest.approx(1.5e-3)
    assert 0 < summary["auc_tr"] < 2 / 6
    assert summarize_cds([])["mean_cd"] is None
    assert summarize_cds([None])["auc_tr"] == 0


def test_auc_matches_official_log_trapezoid_and_scales_quadratically():
    # Interior log-grid threshold with a tolerance plateau on either side.
    cds = [0, 2e-4, 0.3, None]
    xs = np.linspace(1, 5, 401)
    recalls = [sum(cd is not None and cd <= 10 ** (-x) for cd in cds) / 4 for x in xs]
    expected = (
        sum(
            (xs[i] - xs[i - 1]) * (recalls[i] + recalls[i - 1]) / 2
            for i in range(1, 401)
        )
        / 4
    )
    assert summarize_cds(cds)["auc_tr"] == pytest.approx(expected)
    scaled = [None if cd is None else cd * 4 for cd in cds]
    assert summarize_cds(scaled, min_cd=4e-5, max_cd=0.4)["auc_tr"] == pytest.approx(
        expected
    )


def test_evaluation_errors_never_become_failed_generation_or_shrink_denominator():
    good = {"build_valid": True, "metrics": {"chamfer_diag": 0}, "errors": {}}
    invalid = {
        "build_valid": False,
        "metrics": {},
        "errors": {"prediction": "bad build"},
    }
    broken = {
        "build_valid": True,
        "metrics": {},
        "errors": {"alignment": "Boolean failure"},
    }
    summary = aggregate_itercad([good, invalid])
    assert summary["auc_tr"] == pytest.approx(0.5)
    assert summary["mean_cd"] == 0
    summary = aggregate_itercad([good, invalid, broken])
    assert summary["total_samples"] == 3
    assert summary["invalid_predictions"] == 1
    assert summary["cd_evaluation_errors"] == 1
    assert summary["auc_tr"] is None
    assert summary["mean_cd"] is None
    assert summary["median_cd"] is None
    # A different metric's evaluator error does not erase a valid CD.
    partial = {**good, "errors": {"eccv": "sampler failure"}}
    assert aggregate_itercad([partial])["auc_tr"] == pytest.approx(1)


def test_aggregation_rejects_mixed_preprocessing():
    records = [{"protocol": {"reference_extent": r}} for r in [1.8, 1.0]]
    with pytest.raises(ValueError, match="protocol"):
        aggregate_itercad(records)


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf])
def test_preprocess_config_rejects_bad_extent(value):
    with pytest.raises(ValueError):
        PreprocessConfig(reference_extent=value)


@pytest.mark.parametrize(
    "options", [{"min_cd": 0}, {"min_cd": 1}, {"max_cd": np.nan}, {"num_points": 1}]
)
def test_auc_rejects_bad_threshold_configuration(options):
    with pytest.raises(ValueError):
        summarize_cds([0], **options)
