import cv2
import numpy as np
import pytest

from zeroshot.pipeline.verification.drawing_diff.unmatched import (
    find_unmatched,
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
