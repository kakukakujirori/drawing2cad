import cv2
import numpy as np
import pytest

from zeroshot.pipeline.verification.drawing_diff import ocr


def _page_with_text(text="123.45"):
    page = np.full((300, 500, 3), 255, np.uint8)
    cv2.putText(page, text, (130, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 2)
    return page


def _matching(lines, text="123.45"):
    return [line for line in lines if line.string == text]


@pytest.mark.parametrize(
    "turn", [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]
)
def test_a_line_is_found_at_its_place_whichever_way_it_runs(turn):
    page = _page_with_text()
    if turn is not None:
        page = cv2.rotate(page, turn)
    ink = np.argwhere(page[..., 0] < 128)

    lines = _matching(ocr.read_text_lines(page))

    assert lines
    for line in lines:
        corners = np.array(line.corners)
        assert (corners.min(axis=0) <= ink.min(axis=0)[::-1] + 3).all()
        assert (corners.max(axis=0) >= ink.max(axis=0)[::-1] - 3).all()


def test_an_image_is_read_once_for_its_content(monkeypatch):
    page = _page_with_text("77.31")
    first = ocr.read_text_lines(page)

    def never(_image):
        raise AssertionError("read twice")

    monkeypatch.setattr(ocr, "_engine", lambda: never)

    assert ocr.read_text_lines(page.copy()) == first
    with pytest.raises(AssertionError, match="read twice"):
        ocr.read_text_lines(_page_with_text("77.32"))


def test_a_page_without_text_has_no_lines():
    assert ocr.read_text_lines(np.full((100, 100, 3), 255, np.uint8)) == []
