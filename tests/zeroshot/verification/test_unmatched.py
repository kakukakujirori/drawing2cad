import cv2
import numpy as np
import pytest

from zeroshot.pipeline.verification.drawing_diff.unmatched import (
    COLORS,
    INPUT_GRAY,
    draw_unmatched,
    find_unmatched,
    measure_material_error,
    warp_output_to_drawing,
)


def _lines(*segments, width=1, size=200):
    image = np.zeros((size, size), np.uint8)
    for start, end in segments:
        cv2.line(image, start, end, 1, width)
    return image.astype(bool)


OUTLINE = (
    ((40, 40), (160, 40)),
    ((160, 40), (160, 160)),
    ((160, 160), (40, 160)),
    ((40, 160), (40, 40)),
)
INNER = ((60, 100), (140, 100))


def test_a_drawing_line_the_output_lacks_is_one_missing_group():
    groups = find_unmatched(_lines(*OUTLINE, INNER, width=5), _lines(*OUTLINE))

    assert [g.direction for g in groups] == ["missing"]
    x0, y0, x1, y1 = groups[0].box_px
    assert abs(x0 - 60) <= 2 and abs(x1 - 141) <= 2 and y0 <= 100 < y1


def test_an_output_line_the_drawing_lacks_is_extra():
    groups = find_unmatched(_lines(*OUTLINE, width=5), _lines(*OUTLINE, INNER))

    assert [g.direction for g in groups] == ["extra"]


def test_thick_input_line_keeps_its_color_band_and_gray_center_visible():
    drawing = _lines(*OUTLINE, INNER, width=11)
    output = _lines(*OUTLINE)
    groups = find_unmatched(drawing, output)
    image = draw_unmatched(np.uint8(~drawing) * 255, output, groups)

    assert np.all(image[99:102, 90:120] == INPUT_GRAY)
    assert np.all(image[97, 90:120] == COLORS["red"])
    assert np.all(image[103, 90:120] == COLORS["red"])


def test_length_does_not_depend_on_stroke_width():
    thick = find_unmatched(_lines(*OUTLINE, INNER, width=7), _lines(*OUTLINE))
    thin = find_unmatched(_lines(*OUTLINE, width=7), _lines(*OUTLINE, INNER))

    assert abs(thick[0].size_px - thin[0].size_px) <= 6


def test_lines_beyond_the_part_and_short_marks_are_not_listed():
    far = ((40, 195), (160, 195))  # a dimension line below the part
    mark = ((95, 70), (115, 70))  # shorter than any listed group
    drawing = _lines(*OUTLINE, far, mark, width=5)

    assert find_unmatched(drawing, _lines(*OUTLINE)) == []


def test_material_the_drawing_leaves_empty_stands_for_its_own_edges():
    notched = (  # the outline with a 60 x 40 notch cut into its left side
        ((40, 40), (160, 40)),
        ((160, 40), (160, 160)),
        ((160, 160), (40, 160)),
        ((40, 160), (40, 120)),
        ((40, 120), (100, 120)),
        ((100, 120), (100, 80)),
        ((100, 80), (40, 80)),
        ((40, 80), (40, 40)),
    )
    groups = find_unmatched(_lines(*notched, width=3), _lines(*OUTLINE))

    # The notch's lines trace the material's outline, so they are not listed again.
    assert [(g.direction, g.material) for g in groups] == [("extra", True)]
    x0, y0, x1, y1 = groups[0].box_px
    assert [x0, y0, x1, y1] == pytest.approx([40, 80, 100, 120], abs=3)


@pytest.mark.parametrize("offset", [0.0, 0.5, 1.0, 1.5])
def test_thin_lines_survive_shrinking_onto_the_drawing(offset):
    output = _lines(*OUTLINE, size=500)  # one pixel wide
    matrix = np.array([[2.5, 0, offset], [0, 2.5, offset], [0, 0, 1.0]])
    seen = warp_output_to_drawing(output, matrix, (200, 200))

    assert seen[16:64, 16].all() or seen[16:64, 15].all()
    assert not find_unmatched(
        _lines(
            ((16, 16), (64, 16)),
            ((64, 16), (64, 64)),
            ((64, 64), (16, 64)),
            ((16, 64), (16, 16)),
            width=2,
        ),
        seen,
    )


def test_region_mask_excludes_annotation_enclosures_and_does_not_close_gaps():
    drawing = np.zeros((420, 420), np.uint8)
    cv2.rectangle(drawing, (80, 80), (240, 240), 1, 1)
    output = drawing.copy()
    baseline = measure_material_error(drawing.astype(bool), output.astype(bool))
    cv2.rectangle(drawing, (60, 270), (260, 390), 1, 1)
    drawing[390, 160] = 0  # closing this annotation would invent a large filled region
    cv2.rectangle(drawing, (300, 30), (320, 50), 1, 1)  # a closed character

    error = measure_material_error(drawing.astype(bool), output.astype(bool))

    assert error == baseline
    assert error["union_px2"] < 170**2


def test_region_score_counts_missing_material_outside_the_current_output_bbox():
    drawing, output = np.zeros((300, 300), np.uint8), np.zeros((300, 300), np.uint8)
    cv2.rectangle(drawing, (20, 20), (260, 200), 1, 1)
    cv2.rectangle(output, (160, 20), (260, 200), 1, 1)

    error = measure_material_error(drawing.astype(bool), output.astype(bool))

    assert error["missing_px2"] > 20000
    assert error["extra_px2"] < 100  # extraction can round the input's corners
    assert error["ratio"] > 0.5


def test_region_score_counts_all_groups_beyond_the_five_displayed_clusters():
    drawing = np.zeros((500, 500), np.uint8)
    cv2.rectangle(drawing, (20, 20), (100, 100), 1, 1)
    output = drawing.copy()
    for x in (180, 280, 380):
        for y in (200, 300):
            cv2.rectangle(output, (x, y), (x + 50, y + 50), 1, 1)
    drawing, output = drawing.astype(bool), output.astype(bool)

    groups = find_unmatched(drawing, output)
    error = measure_material_error(drawing, output)

    assert len(groups) == 5 and all(group.material for group in groups)
    assert error["extra_px2"] > sum(group.size_px for group in groups)
    assert error["missing_px2"] == 0


def test_open_or_thin_input_has_unavailable_region_error():
    error = measure_material_error(_lines(INNER), _lines(*OUTLINE))

    assert error["status"] == "unavailable"
    assert error["ratio"] is None
    assert "input part area" in error["reason"]


def test_region_score_keeps_a_thin_misalignment_instead_of_reporting_zero():
    drawing, output = np.zeros((350, 450), np.uint8), np.zeros((350, 450), np.uint8)
    cv2.rectangle(drawing, (80, 60), (320, 290), 1, 1)
    cv2.rectangle(output, (88, 60), (328, 290), 1, 1)

    error = measure_material_error(drawing.astype(bool), output.astype(bool))

    assert error["missing_px2"] + error["extra_px2"] == 3696
    assert error["ratio"] > 0.05
