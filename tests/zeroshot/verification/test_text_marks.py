import numpy as np
import pytest

from zeroshot.pipeline.verification.drawing_diff.align import AlignmentResult
from zeroshot.pipeline.verification.drawing_diff.diff import compute_diff
from zeroshot.pipeline.verification.drawing_diff.ocr import TextLine
from zeroshot.pipeline.verification.drawing_diff.text_marks import (
    text_marks,
    text_regions,
)


def _line(string, confidence=1.0, box=(10, 20, 40, 30)):
    x0, y0, x1, y1 = box
    return TextLine([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], string, confidence)


@pytest.mark.parametrize(
    "string, confidence",
    [
        ("33.3473", 1.0),
        ("R6", 1.0),
        ("40", 0.98),
        ("ø5", 0.73),
        ("026.46mm", 0.96),
        ("ECIFIED:", 0.93),
    ],
)
def test_numbers_and_words_are_text(string, confidence):
    assert text_regions([_line(string, confidence)], (50, 50)).any()


@pytest.mark.parametrize(
    "string, confidence",
    [
        ("O", 0.99),
        ("0", 1.0),
        ("8", 1.0),
        ("00", 1.0),
        ("Q", 1.0),
        ("II", 1.0),
        ("三", 0.76),
        ("—", 0.85),
        ("ROS", 0.59),
        ("7", 0.64),
    ],
)
def test_what_line_work_can_read_as_is_not_text(string, confidence):
    assert not text_regions([_line(string, confidence)], (50, 50)).any()


def test_a_region_covers_its_line_box():
    regions = text_regions([_line("12")], (50, 50))

    assert regions[20:31, 10:41].all()
    assert regions.sum() == 11 * 31


def test_ink_that_the_output_explains_is_not_text():
    ink = np.zeros((50, 50), bool)
    ink[25, 15:20] = True  # a glyph, far from the output
    ink[10:40, 30] = True  # a part line crossing the region
    regions = text_regions([_line("12")], (50, 50))
    output = np.zeros((50, 50), bool)
    output[:, 30] = True

    marks = text_marks(ink, regions, output)

    assert marks[25, 15:20].all()
    assert not marks[:, 30].any()
    assert not marks[:20, :].any()


def _white(height=200, width=300):
    return np.full((height, width, 3), 255, np.uint8)


def _alignment():
    return AlignmentResult(
        "directional_chamfer", "similarity", "ok", np.eye(3).tolist(), {}
    )


def _glyph_inside_a_part():
    output = _white()
    output[40, 40:260] = output[160, 40:260] = 0
    output[40:161, 40] = output[40:161, 260] = 0
    drawing = output.copy()
    # Long enough to be listed as missing when it counts as a part line.
    drawing[100, 120:190] = 0
    regions = np.zeros(drawing.shape[:2], bool)
    regions[92:109, 112:198] = True
    return drawing, output, regions


def test_text_is_not_missing_and_leaves_the_score():
    drawing, output, regions = _glyph_inside_a_part()

    as_lines = compute_diff(drawing, output, _alignment())
    as_text = compute_diff(drawing, output, _alignment(), text_regions=regions)

    assert [item["direction"] for item in as_lines.stats["unmatched"]] == ["missing"]
    assert as_text.stats["unmatched"] == []
    assert as_text.stats["match_score"] < as_lines.stats["match_score"]


def test_text_regions_must_lie_on_the_drawing_grid():
    drawing, output, _ = _glyph_inside_a_part()

    with pytest.raises(ValueError, match="text_regions"):
        compute_diff(
            drawing,
            output,
            _alignment(),
            text_regions=np.zeros((5, 5), bool),
        )
