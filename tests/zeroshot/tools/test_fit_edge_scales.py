"""Centre-line extraction at different stroke widths and pixel alignments."""

import numpy as np
import pytest

from zeroshot.pipeline.tools import fit_edge as fit


def sheet_from_gray(gray, line_width):
    ink = gray <= 128
    return fit.Sheet(
        gray.astype(np.float32),
        ink,
        np.minimum(fit._run_lengths(ink), fit._run_lengths(ink.T).T),
        128,
        line_width,
        np.zeros_like(ink),
    )


@pytest.mark.parametrize("scale", [1, 2, 3])
@pytest.mark.parametrize("vertical", [False, True])
def test_close_parallel_strokes_stay_separate(scale, vertical):
    gray = np.full((180, 240), 255, np.uint8)
    gray[79:81, 20:220] = 20
    gray[82:84, 20:220] = 20
    gray[120:127, 20:220] = 20
    gray = np.repeat(np.repeat(gray, scale, axis=0), scale, axis=1)
    if vertical:
        gray = gray.T
    sheet = sheet_from_gray(gray, 7 * scale)
    box = np.array([40, 75, 200, 90]) * scale
    if vertical:
        box = box[[1, 0, 3, 2]]
    finding = fit.find_strokes(sheet, tuple(box), "line")
    assert len(finding.strokes) == 2, finding
    centres = sorted(s.model.x if vertical else s.model.y for s in finding.strokes)
    assert centres == pytest.approx(np.array([80, 83]) * scale, abs=0.3)
    assert all(s.rms < 0.1 for s in finding.strokes)
    assert all(s.support >= 155 * scale for s in finding.strokes)


@pytest.mark.parametrize("angle", [0, 30, 45, 75, 90])
@pytest.mark.parametrize("width", [1, 2, 3, 7])
def test_isolated_line_has_one_centre_and_preserves_position(angle, width):
    # Integrate a straight strip over each pixel; its true centre is known.
    size, samples = 120, 8
    rows, cols = np.mgrid[: size * samples, : size * samples] / samples
    normal = np.array([-np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))])
    distance = (cols + 0.5 / samples - 60) * normal[0]
    distance += (rows + 0.5 / samples - 60) * normal[1]
    dark = (np.abs(distance) < width / 2).reshape(size, samples, size, samples)
    gray = np.round(255 * (1 - dark.mean(axis=(1, 3)))).astype(np.uint8)
    sheet = sheet_from_gray(gray, 7)
    finding = fit.find_strokes(sheet, (20, 20, 100, 100), "line")
    assert len(finding.strokes) == 1, finding
    [stroke] = finding.strokes
    assert abs(stroke.model.residual(np.array([[60, 60]]))[0]) < 0.2
    assert stroke.rms < 0.25
    points = fit.centre_points(sheet, (20, 20, 100, 100))
    assert np.max(np.abs((points.xy - 60) @ normal)) < 0.5


def test_close_concentric_strokes_remain_two_circles():
    size, samples = 140, 8
    rows, cols = (np.mgrid[: size * samples, : size * samples] + 0.5) / samples
    radius = np.hypot(cols - 70, rows - 70)
    dark = ((np.abs(radius - 40) < 1) | (np.abs(radius - 43) < 1)).reshape(
        size, samples, size, samples
    )
    gray = np.round(255 * (1 - dark.mean(axis=(1, 3)))).astype(np.uint8)
    finding = fit.find_strokes(sheet_from_gray(gray, 7), (20, 20, 120, 120), "circle")
    assert len(finding.strokes) == 2, finding
    assert sorted(s.model.r for s in finding.strokes) == pytest.approx(
        [40, 43], abs=0.3
    )


@pytest.mark.parametrize("fringes", [(153, 170), (160, 160)])
@pytest.mark.parametrize("vertical", [False, True])
def test_antialiased_two_pixel_line_does_not_produce_duplicate_centres(
    fringes, vertical
):
    gray = np.full((180, 240), 255, np.uint8)
    gray[78:82, 20:220] = np.array([fringes[0], 0, 0, fringes[1]])[:, None]
    if vertical:
        gray = gray.T
    sheet = sheet_from_gray(gray, 7)
    box = (75, 40, 90, 200) if vertical else (40, 75, 200, 90)
    points = fit.centre_points(sheet, box)
    assert len(points.xy) == 160
    assert np.max(np.abs(points.xy[:, 0 if vertical else 1] - 80)) < 0.1
    finding = fit.find_strokes(sheet, box, "line")
    assert len(finding.strokes) == 1, finding
