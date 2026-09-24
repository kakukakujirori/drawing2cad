from importlib import import_module
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

alignment = import_module("zeroshot.pipeline.verification.drawing_diff.align")
chamfer = import_module(
    "zeroshot.pipeline.verification.drawing_diff.directional_chamfer"
)
matchanything = import_module(
    "zeroshot.pipeline.verification.drawing_diff.match_anything"
)


def _drawing():
    image = np.full((180, 240, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (25, 20), (145, 120), (0, 0, 0), 2)
    cv2.circle(image, (55, 45), 12, (0, 0, 0), 2)
    cv2.line(image, (100, 120), (100, 155), (0, 0, 0), 2)
    image[0, 0] = (255, 0, 0)
    return image


def _transform(matrix, points):
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ np.asarray(matrix).T
    return homogeneous[:, :2] / homogeneous[:, 2:]


def _stub_pairs(monkeypatch, points0, points1):
    pairs = SimpleNamespace(
        points0=points0,
        points1=points1,
        confidence=np.ones(len(points0)),
        in_bounds=np.ones(len(points0), dtype=bool),
        diagnostics={},
        debug={},
    )
    seen = []

    def infer_pairs(runtime, rgb0, rgb1):
        seen.append((runtime, rgb0, rgb1))
        return pairs

    monkeypatch.setattr(matchanything, "infer_pairs", infer_pairs)
    return seen


@pytest.mark.parametrize(
    ("model", "matrix"),
    [
        ("similarity", [[1.12, -0.18, 24], [0.18, 1.12, 8], [0, 0, 1]]),
        ("affine", [[1.05, 0.22, 18], [-0.06, 0.87, 18], [0, 0, 1]]),
        ("homography", [[1.05, 0.17, 18], [-0.04, 0.94, 14], [0.0012, -0.0007, 1]]),
    ],
)
def test_matchanything_fits_each_model_with_outliers(monkeypatch, model, matrix):
    rng = np.random.default_rng(17)
    points = rng.uniform((12, 12), (110, 120), (40, 2))
    targets = _transform(np.asarray(matrix), points)
    targets[-8:] = rng.uniform((150, 20), (220, 160), (8, 2))
    points = np.vstack((points, [[np.nan, 20], [240, 40]]))
    targets = np.vstack((targets, [[30, 30], [40, 40]]))
    seen = _stub_pairs(monkeypatch, points, targets)
    drawing, projection = _drawing(), _drawing()
    projection[0, 0] = (0, 0, 255)
    runtime = SimpleNamespace(provenance={})
    result = alignment.align(
        drawing, projection, backend="match_anything", model=model, runtime=runtime
    )

    assert seen[0][0] is runtime
    np.testing.assert_array_equal(seen[0][1], drawing)
    np.testing.assert_array_equal(seen[0][2], projection)
    assert (result.backend, result.model) == ("match_anything", model)
    assert result.status != "failed"
    assert result.diagnostics["valid_match_count"] == 40
    assert result.diagnostics["inlier_count"] == 32
    # Landmarks are pixel centers in the public boundary coordinate system.
    mapped = _transform(result.H_drawing_to_projection, points[:32] + 0.5)
    np.testing.assert_allclose(mapped, targets[:32] + 0.5, atol=1e-3)
    mapped_cv = _transform(
        alignment.opencv_transform(result.H_drawing_to_projection), points[:32]
    )
    np.testing.assert_allclose(mapped_cv, targets[:32], atol=1e-3)


def test_similarity_does_not_report_the_affine_prefilter_as_its_inliers(monkeypatch):
    x, y = np.meshgrid(np.linspace(20, 110, 7), np.linspace(20, 110, 7))
    points = np.column_stack((x.ravel(), y.ravel()))
    targets = _transform(np.diag([1.2, 0.8, 1]), points)
    _stub_pairs(monkeypatch, points, targets)
    result = alignment.align(
        _drawing(),
        _drawing(),
        backend="match_anything",
        runtime=SimpleNamespace(provenance={}),
    )
    assert result.diagnostics["prefilter_inlier_count"] == len(points)
    assert result.diagnostics["inlier_count"] < len(points) // 2


@pytest.mark.parametrize("model", ["similarity", "affine", "homography"])
def test_chamfer_keeps_its_objective_direction_and_inverts_the_result(
    monkeypatch, model
):
    drawing, projection = _drawing(), _drawing()[::2, ::2].copy()
    projection[0, 0] = (0, 0, 255)
    matrix = np.array([[1.5, -0.2, 7], [0.2, 1.5, 11], [0, 0, 1]])

    def register(source_gray, drawing_gray, cfg):
        np.testing.assert_array_equal(
            source_gray, cv2.cvtColor(projection, cv2.COLOR_RGB2GRAY)
        )
        np.testing.assert_array_equal(
            drawing_gray, cv2.cvtColor(drawing, cv2.COLOR_RGB2GRAY)
        )
        assert cfg.model == model and cfg.maxiter == 5
        return {
            "H_source_to_drawing": matrix.tolist(),
            "config": {"model": model},
            "warnings": ["test warning"],
            "hypotheses": [],
            "optimizer_runs": [{"success": True}],
        }

    monkeypatch.setattr(chamfer, "register", register)
    result = alignment.align(drawing, projection, model=model, options={"maxiter": 5})
    assert result.status != "failed"
    assert "test warning" in result.warnings
    projection_points = np.array([[5.0, 7.0], [22.0, 41.0], [81.0, 61.0]])
    drawing_points = _transform(matrix, projection_points)
    actual = _transform(result.H_drawing_to_projection, drawing_points + 0.5)
    np.testing.assert_allclose(actual, projection_points + 0.5, atol=1e-10)


@pytest.mark.parametrize("points", [np.empty((0, 2)), np.array([[20, 20]] * 8)])
def test_missing_or_degenerate_matches_fail_without_a_matrix(monkeypatch, points):
    _stub_pairs(monkeypatch, points, points.copy())
    result = alignment.align(
        _drawing(),
        _drawing(),
        backend="match_anything",
        runtime=SimpleNamespace(provenance={}),
    )
    assert result.status == "failed"
    assert result.H_drawing_to_projection is None
    assert result.diagnostics["error"]


def test_blank_input_and_invalid_api_arguments():
    image = _drawing()
    result = alignment.align(np.full_like(image, 255), image)
    assert result.status == "failed" and result.H_drawing_to_projection is None
    for bad in (image[..., 0], image.astype(float), image[:0]):
        with pytest.raises(ValueError):
            alignment.align(bad, image)
    for arguments in (
        {"backend": "unknown"},
        {"model": "rigid"},
        {"options": {"unknown": True}},
        {
            "backend": "match_anything",
            "runtime": object(),
            "options": {"unknown": True},
        },
    ):
        with pytest.raises(ValueError):
            alignment.align(image, image, **arguments)


def test_chamfer_optimizer_recovers_a_transformed_line_drawing():
    projection = np.full((96, 128, 3), 255, np.uint8)
    outline = np.array([[15, 12], [95, 12], [95, 55], [75, 55], [75, 75], [15, 75]])
    cv2.polylines(projection, [outline], True, (0, 0, 0), 1)
    cv2.circle(projection, (37, 31), 10, (0, 0, 0), 1)
    angle = np.deg2rad(3)
    a, b = 1.35 * np.cos(angle), 1.35 * np.sin(angle)
    matrix = np.array([[a, -b, 20], [b, a, 10], [0, 0, 1]])
    drawing = cv2.warpPerspective(
        projection, matrix, (190, 140), borderValue=(255, 255, 255)
    )
    result = alignment.align(
        drawing,
        projection,
        options={"maxiter": 60, "popsize": 10, "restarts": 1, "top_k": 2},
    )
    assert result.status != "failed", result.diagnostics
    # General caveats are in the feedback legend, not repeated per view.
    assert result.diagnostics["chamfer"]["warnings"] == []
    points = np.array([[15, 12], [95, 12], [75, 75], [37, 31]])
    restored = _transform(
        result.H_drawing_to_projection, _transform(matrix, points) + 0.5
    )
    assert np.max(np.linalg.norm(restored - (points + 0.5), axis=1)) < 1


def test_rgb_loader_composites_alpha_without_exif_rotation(tmp_path):
    pixels = np.array(
        [[[255, 0, 0, 255], [0, 0, 0, 0], [0, 0, 255, 128]]], dtype=np.uint8
    )
    image = Image.fromarray(pixels)
    exif = Image.Exif()
    exif[274] = 6
    path = tmp_path / "drawing.png"
    image.save(path, exif=exif)
    loaded = alignment.read_rgb(path)
    np.testing.assert_array_equal(
        loaded, [[[255, 0, 0], [255, 255, 255], [127, 127, 255]]]
    )
    assert loaded.dtype == np.uint8
