import cv2
import ezdxf
import numpy as np
import pytest

from zeroshot.pipeline.verification.drawing_diff.align import AlignmentResult
from zeroshot.pipeline.verification.drawing_diff.diff import compute_diff
from zeroshot.pipeline.verification.drawing_diff.image_ops import (
    annotation_mask,
    beside_part,
    drawing_area,
    foreground_mask,
    thin_mask,
)


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
    assert np.all(diff.overlay[10:55, 24] == [0, 168, 255])
    assert np.all(diff.overlay[20, 22:27] == [0, 168, 255])
    assert np.all(diff.overlay[20, [21, 27]] == 255)
    assert np.all(diff.overlay[10:55, 20] == 225)
    assert diff.stats["outside_fraction"] == 0
    assert diff.stats["material_error"]["status"] == "unavailable"
    # A line drawing has no fillable silhouette, so lines alone score it.
    assert diff.stats["match_score"] == pytest.approx(
        diff.stats["unbounded_chamfer_drawing_px"]
    )


def test_chamfer_caps_far_lines():
    drawing, output = _white(), _white()
    drawing[10:55, [20, 60]] = 0  # the line at 60 is 36px from any output line
    output[10:55, 24] = 0
    diff = compute_diff(drawing, output, _alignment())

    # input capped (4 + 20) / 2 = 12, output 4; uncapped input (4 + 36) / 2 = 20
    assert diff.stats["bounded_chamfer_drawing_px"] == 8.0
    assert diff.stats["unbounded_chamfer_drawing_px"] == 12.0


def test_chamfer_counts_drawing_lines_beyond_the_projection():
    drawing = np.full((64, 100, 3), 255, dtype=np.uint8)
    output = _white()
    drawing[10:55, [20, 90]] = 0  # x=90 lies outside the 64px projection
    output[10:55, 20] = 0
    diff = compute_diff(drawing, output, _alignment())

    # input (0 + capped 20) / 2 = 10, output 0
    assert diff.stats["bounded_chamfer_drawing_px"] == 5.0


def test_unmatched_groups_are_listed_in_drawing_pixels():
    drawing = np.full((160, 200, 3), 255, dtype=np.uint8)
    output = _white(160)
    drawing[20:140, [20, 80, 140]] = 0
    output[20:140, [20, 140]] = 0
    diff = compute_diff(drawing, output, _alignment())

    assert diff.stats["unmatched"] == [
        {
            "direction": "missing",
            "kind": "lines",
            "unit": "px",
            "size_px": 120,
            "box_px": [80, 20, 81, 140],
            "color": "red",
        }
    ]
    assert diff.unmatched.shape == drawing.shape


def test_diff_builds_the_local_comparison_mask_from_dxf_geometry():
    drawing, output = _white(200), _white(200)
    chamfered = np.array(
        [
            (64, 40),
            (136, 40),
            (160, 64),
            (160, 136),
            (136, 160),
            (64, 160),
            (40, 136),
            (40, 64),
        ]
    )
    cv2.polylines(drawing, [chamfered], True, (0, 0, 0), 5)
    cv2.rectangle(output, (40, 40), (160, 160), (0, 0, 0), 1)
    coarse = compute_diff(drawing, output, _alignment())
    assert coarse.stats["unmatched"] == []

    doc = ezdxf.new()
    corners = [(0, 60), (60, 60), (60, 0), (0, 0)]
    for start, end in zip(corners, corners[1:] + corners[:1]):
        doc.modelspace().add_line(start, end)
    # Each drawing pixel is 0.5 mm; CAD +V is opposite the raster's y direction.
    projection_to_uv = np.array([[0.5, 0, -20.25], [0, -0.5, 80.25], [0, 0, 1]])
    detailed = compute_diff(
        drawing,
        output,
        _alignment(),
        projection_modelspace=doc.modelspace(),
        projection_to_uv=projection_to_uv,
    )

    assert len(detailed.stats["unmatched"]) == 2
    assert all(item["kind"] == "lines" for item in detailed.stats["unmatched"])
    for key in ("bounded_chamfer_drawing_px", "material_error", "match_score"):
        assert detailed.stats[key] == coarse.stats[key]


def test_local_comparison_ignores_hidden_dash_phase_near_a_visible_corner():
    drawing, output = _white(200), _white(200)
    for image in (drawing, output):
        cv2.rectangle(image, (40, 40), (160, 160), (0, 0, 0), 1)
    for start, end in [(0, 180), (270, 360)]:
        cv2.ellipse(drawing, (70, 70), (18, 18), 0, start, end, (0, 0, 0), 1)
    for start, end in [(0, 70), (90, 250), (270, 360)]:
        cv2.ellipse(output, (70, 70), (18, 18), 0, start, end, (0, 0, 0), 1)

    doc = ezdxf.new()
    corners = [(40, 40), (160, 40), (160, 160), (40, 160)]
    for start, end in zip(corners, corners[1:] + corners[:1]):
        doc.modelspace().add_line(start, end)
    doc.modelspace().add_circle((70, 70), 18, dxfattribs={"linetype": "HIDDEN"})
    coarse = compute_diff(drawing, output, _alignment())
    detailed = compute_diff(
        drawing,
        output,
        _alignment(),
        projection_modelspace=doc.modelspace(),
        projection_to_uv=np.array([[1.0, 0, -0.5], [0, 1.0, -0.5], [0, 0, 1]]),
    )

    assert coarse.stats["unmatched"] == detailed.stats["unmatched"] == []
    for key in ("bounded_chamfer_drawing_px", "material_error", "match_score"):
        assert detailed.stats[key] == coarse.stats[key]


def test_missing_hole_only_appears_in_input_distances():
    drawing, output = _white(100), _white(100)
    cv2.rectangle(output, (10, 10), (90, 90), (0, 0, 0), 1)
    drawing[:] = output
    cv2.circle(drawing, (50, 50), 10, (0, 0, 0), 1)
    diff = compute_diff(drawing, output, _alignment())

    assert diff.stats["output_to_input"]["max_px"] == 0
    assert diff.stats["input_to_output"]["max_px"] >= 30
    np.testing.assert_array_equal(diff.overlay[10, 50], [0, 0, 128])


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


def test_failed_alignment_returns_no_fake_images():
    drawing, output = _white(32), _white()
    drawing[8:24, 10] = 0
    output[18:34, 20] = 0
    failed = AlignmentResult(
        "directional_chamfer", "similarity", "failed", None, {}, ("no transform",)
    )
    diff = compute_diff(drawing, output, failed)
    assert diff.overlay is None and diff.unmatched is None
    assert diff.stats["comparison_status"] == "failed"
    assert "output_to_input" not in diff.stats
    assert diff.warnings[0] == "no transform"
    assert "Drawing comparison unavailable" in diff.warnings[1]
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
    assert automatic.stats["output_to_input"] == fixed.stats["output_to_input"]
    assert automatic.stats["input_to_output"] == fixed.stats["input_to_output"]
    assert np.all(automatic.overlay[10:55, 22] == [0, 168, 255])
    assert np.all(automatic.overlay[10:55, 26] == [255, 0, 0])
    assert np.all(fixed.overlay[10:55, 22] == [0, 20, 255])
    assert np.all(clipped.overlay[10:55, [22, 26]] == [255, 0, 0])
    assert clipped.stats["output_to_input"] == fixed.stats["output_to_input"]
    assert automatic.warnings == fixed.warnings == ()


def test_automatic_range_ignores_missing_input_lines():
    drawing, output = _white(100), _white(100)
    cv2.rectangle(output, (10, 10), (90, 90), (0, 0, 0), 1)
    drawing[:] = output
    identical = compute_diff(drawing, output, _alignment(), distance_clip_px=None)
    assert identical.stats["red_distance_px"] == 0
    np.testing.assert_array_equal(identical.overlay[10, 50], [0, 0, 128])

    cv2.circle(drawing, (50, 50), 10, (0, 0, 0), 1)
    missing_hole = compute_diff(drawing, output, _alignment(), distance_clip_px=None)
    assert missing_hole.stats["red_distance_px"] == 0
    np.testing.assert_array_equal(missing_hole.overlay[10, 50], [0, 0, 128])


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


def test_thickening_keeps_original_measurements_where_strokes_overlap():
    drawing, output = _white(), _white()
    drawing[10:55, 20] = 0
    output[:] = drawing
    output[10:55, 22] = 0
    diff = compute_diff(drawing, output, _alignment(), distance_clip_px=2)

    # The high-error halo must not turn the adjacent matching line red.
    np.testing.assert_array_equal(diff.overlay[30, 20], [0, 0, 128])
    np.testing.assert_array_equal(diff.overlay[30, 22], [255, 0, 0])
    np.testing.assert_array_equal(diff.overlay[30, 21], [255, 0, 0])
    assert diff.stats["output_to_input"]["count"] == 90
    assert diff.stats["output_to_input"]["max_px"] == 2
    assert diff.stats["input_to_output"]["max_px"] == 0


def test_match_score_adds_line_distance_and_silhouette_offset_in_pixels():
    drawing, output = _white(300), _white(300)
    outline = np.array(
        [
            [40, 40],
            [260, 40],
            [260, 260],
            [40, 260],
            [40, 180],
            [140, 180],
            [140, 100],
            [40, 100],
        ]
    )
    cv2.polylines(drawing, [outline], True, (0, 0, 0), 1)
    cv2.rectangle(output, (40, 40), (260, 260), (0, 0, 0), 1)

    diff = compute_diff(drawing, output, _alignment())
    material = diff.stats["material_error"]

    assert material["status"] == "ok" and material["extra_px2"] > 6000
    assert material["missing_px2"] == 0
    # The outline of the L-shaped input is 1100 px long.
    assert material["input_perimeter_px"] == pytest.approx(1100, rel=0.05)
    offset = (material["missing_px2"] + material["extra_px2"]) / material[
        "input_perimeter_px"
    ]
    line = diff.stats["unbounded_chamfer_drawing_px"]
    assert diff.stats["match_score"] == pytest.approx(line + offset)
    assert line >= diff.stats["bounded_chamfer_drawing_px"]


def test_a_thin_walled_outline_has_no_silhouette():
    """Sheet metal edge-on: the opening that drops annotations erases the part."""
    ink = np.zeros((120, 160), bool)
    cv2.rectangle(ink.view(np.uint8), (20, 20), (140, 24), 1, 1)
    cv2.rectangle(ink.view(np.uint8), (20, 20), (24, 100), 1, 1)
    assert not drawing_area(ink).any()
    solid = np.zeros((120, 160), bool)
    cv2.rectangle(solid.view(np.uint8), (20, 20), (140, 100), 1, 1)
    assert drawing_area(solid).sum() > 0.9 * 121 * 81


def _dimensioned_part(annotation_color, annotation_width=2, outline_width=5):
    """A part with a hole; its dimension lines close a box beside it."""
    image = _white(300)
    for y in (60, 240):  # extension lines continuing the top and bottom edges
        cv2.line(image, (150, y), (270, y), annotation_color, annotation_width)
    cv2.line(image, (250, 50), (250, 250), annotation_color, annotation_width)
    cv2.line(image, (200, 20), (95, 150), annotation_color, annotation_width)  # leader
    cv2.putText(
        image, "R0.13", (175, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, annotation_color, 1
    )
    cv2.rectangle(image, (40, 60), (150, 240), (0, 0, 0), outline_width)
    cv2.circle(image, (95, 150), 20, (0, 0, 0), outline_width)
    return image


def test_light_annotation_neither_encloses_material_nor_joins_a_missing_hole():
    output = _white(300)
    cv2.rectangle(output, (40, 60), (150, 240), (0, 0, 0), 1)

    two_tone = compute_diff(_dimensioned_part((120, 120, 120)), output, _alignment())
    one_tone = compute_diff(_dimensioned_part((0, 0, 0), 5), output, _alignment())
    thin = compute_diff(_dimensioned_part((0, 0, 0)), output, _alignment())

    # In the part's tone and width, the box the dimension lines close reads as
    # missing material; drawn thinner than the part, it does not.
    assert one_tone.stats["material_error"]["missing_px2"] > 15000
    assert two_tone.stats["material_error"]["missing_px2"] < 2500
    assert thin.stats["material_error"]["missing_px2"] < 2500
    assert not any(group["kind"] == "material" for group in thin.stats["unmatched"])
    assert two_tone.stats["match_score"] < one_tone.stats["match_score"] / 3
    # The hole is the only difference, and the leader ending on it is not part of it.
    [hole] = two_tone.stats["unmatched"]
    assert hole["kind"] == "lines" and hole["direction"] == "missing"
    assert hole["box_px"] == pytest.approx([75, 130, 116, 171], abs=3)
    assert one_tone.stats["unmatched"][0]["box_px"][1] < 110  # leader joined in


def _gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


@pytest.mark.parametrize("antialiased", [False, True])
def test_one_tone_drawings_have_no_annotation(antialiased):
    image = _dimensioned_part((0, 0, 0))
    if antialiased:  # a downscaled print: thin lines turn gray, but not as one tone
        image = cv2.resize(image, (100, 100), interpolation=cv2.INTER_AREA)
    gray = _gray(image)

    assert not annotation_mask(gray, foreground_mask(gray)).any()


def test_light_annotation_is_masked_but_a_pale_stretch_of_a_part_line_is_not():
    image = _dimensioned_part((120, 120, 120))
    image[140:160, 40:44] = 125  # blur paling the part's left edge
    gray = _gray(image)
    ink = foreground_mask(gray)

    annotation = annotation_mask(gray, ink)

    assert annotation[150, 250] and annotation[60, 200]  # dimension, extension
    assert annotation[5:17, 175:230].any()  # text
    assert not annotation[145:155, 40:44].any()
    assert not (annotation & (gray < 60)).any()


def _part_lines(width):
    """The part's outline and hole as drawn at width; width 1 is the registered CAD."""
    lines = np.zeros((300, 300), np.uint8)
    cv2.rectangle(lines, (40, 60), (150, 240), 1, width)
    cv2.circle(lines, (95, 150), 20, 1, width)
    return lines.astype(bool)


@pytest.mark.parametrize("outline_width", [5, 9])
def test_thin_lines_are_found_and_wide_outlines_keep_their_edges(outline_width):
    gray = _gray(_dimensioned_part((0, 0, 0), outline_width=outline_width))
    ink = foreground_mask(gray)

    thin = thin_mask(gray, ink, _part_lines(1))

    assert thin[150, 250] and thin[60, 200]  # dimension, extension
    assert not (thin & _part_lines(outline_width)).any()


def test_a_drawing_in_one_width_has_no_thin_lines():
    gray = _gray(_dimensioned_part((0, 0, 0), annotation_width=5))

    assert not thin_mask(gray, foreground_mask(gray), _part_lines(1)).any()


def test_dimensions_beside_the_part_neither_score_nor_cluster():
    part = _white(300)
    cv2.rectangle(part, (40, 60), (150, 240), (0, 0, 0), 5)
    cv2.circle(part, (95, 150), 20, (0, 0, 0), 5)
    dimensioned = part.copy()
    for y in (60, 240):  # extension lines continuing the top and bottom edges
        cv2.line(dimensioned, (150, y), (270, y), (0, 0, 0), 2)
    cv2.line(dimensioned, (250, 50), (250, 250), (0, 0, 0), 2)
    cv2.putText(
        dimensioned, "180", (258, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
    )
    output = _white(300)
    cv2.rectangle(output, (40, 60), (150, 240), (0, 0, 0), 1)
    cv2.circle(output, (95, 150), 20, (0, 0, 0), 1)

    plain = compute_diff(part, output, _alignment())
    diff = compute_diff(dimensioned, output, _alignment())

    assert diff.stats["match_score"] == pytest.approx(
        plain.stats["match_score"], abs=0.05
    )
    assert diff.stats["unmatched"] == []


def test_a_partial_silhouette_leaves_out_only_ink_that_never_joins_it():
    """An outline broken by the crop fills only in part; its other lines stay."""
    ink = np.zeros((300, 200), np.uint8)
    cv2.rectangle(ink, (50, 20), (150, 100), 1, 3)
    for x in (50, 150):  # the outline runs on past the crop's edge
        cv2.line(ink, (x, 100), (x, 299), 1, 3)
    cv2.putText(ink, "NOTE", (2, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1, 1)
    ink = ink.astype(bool)

    beside = beside_part(ink, ink)

    assert beside[260:290, :45].any()  # the note
    assert not beside[:, 45:].any()
