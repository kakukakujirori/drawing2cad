import cv2
import numpy as np
import pytest

from zeroshot.pipeline.verification.drawing_diff.align import AlignmentResult
from zeroshot.pipeline.verification.drawing_diff.diff import compute_diff


def _white(size=64):
    return np.full((size, size, 3), 255, dtype=np.uint8)


def _alignment(matrix=None, status="ok"):
    return AlignmentResult(
        "directional_chamfer",
        "similarity",
        status,
        (np.eye(3) if matrix is None else matrix).tolist(),
        {"test": np.float64(1), "unavailable": float("inf")},
    )


def test_every_shifted_line_pixel_is_colored():
    drawing, output = _white(), _white()
    drawing[10:55, 20] = 0
    output[10:55, 24] = 0
    diff = compute_diff(drawing, output, _alignment())

    assert diff.stats["output_to_input"] == {
        "count": 45,
        "median_px": 4.0,
        "p95_px": 4.0,
        "max_px": 4.0,
    }
    assert np.all(diff.overlay[10:55, 24] == [0, 212, 255])
    assert np.all(diff.overlay[10:55, 20] == 225)
    assert np.all(diff.residual[10:55, 20] == 150)
    assert diff.stats["outside_fraction"] == 0


def test_missing_hole_only_appears_in_reverse_residual():
    drawing, output = _white(100), _white(100)
    cv2.rectangle(output, (10, 10), (90, 90), (0, 0, 0), 1)
    drawing[:] = output
    cv2.circle(drawing, (50, 50), 10, (0, 0, 0), 1)
    diff = compute_diff(drawing, output, _alignment())

    assert diff.stats["output_to_input"]["max_px"] == 0
    assert diff.stats["input_to_output"]["max_px"] >= 30
    np.testing.assert_array_equal(diff.overlay[10, 50], [0, 0, 128])
    assert diff.residual[10, 50] == 225
    assert diff.residual[40, 50] == 0


def test_output_outside_input_is_neutral_and_excluded():
    drawing, output = _white(32), _white()
    drawing[8:24, 10] = 0
    output[18:34, 20] = 0
    output[18:34, 2] = 0
    matrix = np.array([[1, 0, 10], [0, 1, 10], [0, 0, 1.0]])
    diff = compute_diff(drawing, output, _alignment(matrix))

    assert diff.stats["outside_fraction"] == 0.5
    assert diff.stats["outside_count"] == 16
    assert diff.stats["output_to_input"]["count"] == 16
    assert diff.stats["output_to_input"]["max_px"] == 0
    assert np.all(diff.overlay[18:34, 2] == 128)
    assert np.all(diff.overlay[18:34, 20] == [0, 0, 128])
    assert any("outside" in warning for warning in diff.warnings)


def test_public_scale_uses_half_pixel_coordinates():
    drawing = _white(32)
    drawing[8:24, 10] = 0
    # Boundary-frame scale of 3 is x' = 3*x + 1 in integer-center coordinates.
    output = cv2.warpPerspective(
        drawing,
        np.array([[3.0, 0, 1], [0, 3, 1], [0, 0, 1]]),
        (96, 96),
        flags=cv2.INTER_NEAREST,
        borderValue=(255, 255, 255),
    )
    diff = compute_diff(drawing, output, _alignment(np.diag([3.0, 3.0, 1.0])))
    assert diff.stats["output_to_input"]["max_px"] == 0
    assert diff.stats["input_to_output"]["max_px"] == 0


def test_failed_alignment_returns_no_fake_residual():
    drawing, output = _white(32), _white()
    drawing[8:24, 10] = 0
    output[18:34, 20] = 0
    failed = AlignmentResult(
        "directional_chamfer", "similarity", "failed", None, {}, ("no transform",)
    )
    diff = compute_diff(drawing, output, failed)
    assert diff.overlay is None and diff.residual is None
    assert diff.stats["comparison_status"] == "failed"
    assert "output_to_input" not in diff.stats
    assert diff.warnings[0] == "no transform"
    assert "Drawing residuals unavailable" in diff.warnings[1]
    with pytest.raises(ValueError, match="distance_clip_px"):
        compute_diff(drawing, output, failed, distance_clip_px=0)


def test_automatic_range_preserves_gradient_and_raw_distances():
    drawing, output = _white(), _white()
    drawing[10:55, 20] = 0
    output[10:55, 22] = 0
    output[10:55, 26] = 0
    automatic = compute_diff(drawing, output, _alignment(), distance_clip_px=None)
    fixed = compute_diff(drawing, output, _alignment())
    clipped = compute_diff(drawing, output, _alignment(), distance_clip_px=1)

    assert automatic.stats["color_normalization"] == "image_max"
    assert automatic.stats["red_distance_px"] == 6
    assert automatic.stats["black_distance_px"] == 2
    assert automatic.stats["output_to_input"] == fixed.stats["output_to_input"]
    assert automatic.stats["input_to_output"] == fixed.stats["input_to_output"]
    assert np.all(automatic.overlay[10:55, 22] == [0, 212, 255])
    assert np.all(automatic.overlay[10:55, 26] == [128, 0, 0])
    assert np.all(fixed.overlay[10:55, 22] == [0, 40, 255])
    assert np.all(clipped.overlay[10:55, [22, 26]] == [128, 0, 0])
    assert clipped.stats["output_to_input"] == fixed.stats["output_to_input"]
    assert np.all(automatic.residual[10:55, 20] == 0)
    assert automatic.warnings == fixed.warnings == ()


def test_automatic_zero_range_and_reverse_range_are_independent():
    drawing, output = _white(100), _white(100)
    cv2.rectangle(output, (10, 10), (90, 90), (0, 0, 0), 1)
    drawing[:] = output
    identical = compute_diff(drawing, output, _alignment(), distance_clip_px=None)
    assert (
        identical.stats["red_distance_px"] == identical.stats["black_distance_px"] == 0
    )
    np.testing.assert_array_equal(identical.overlay[10, 50], [0, 0, 128])
    assert identical.residual[10, 50] == 225

    cv2.circle(drawing, (50, 50), 10, (0, 0, 0), 1)
    missing_hole = compute_diff(drawing, output, _alignment(), distance_clip_px=None)
    assert missing_hole.stats["red_distance_px"] == 0
    assert missing_hole.stats["black_distance_px"] >= 30
    assert missing_hole.residual.ndim == 2
    np.testing.assert_array_equal(missing_hole.overlay[10, 50], [0, 0, 128])
    assert missing_hole.residual[10, 50] == 225
    assert missing_hole.residual.min() == 0
    assert 0 < missing_hole.residual[40, 50] < 225


def test_automatic_range_is_unknown_when_no_output_lines_are_observed():
    drawing, output = _white(32), _white()
    drawing[8:24, 10] = 0
    output[8:24, 50] = 0
    diff = compute_diff(drawing, output, _alignment(), distance_clip_px=None)

    assert diff.stats["comparison_status"] == "uncertain"
    assert diff.stats["red_distance_px"] is None
    assert diff.stats["output_to_input"] == {
        "count": 0,
        "median_px": None,
        "p95_px": None,
        "max_px": None,
    }
    assert np.all(diff.overlay[8:24, 50] == 128)
    assert diff.stats["black_distance_px"] == 40
