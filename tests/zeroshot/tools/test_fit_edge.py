import math
import re
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.tools.errors import ToolFeedbackError
from zeroshot.pipeline.tools.fit_edge import (
    DASHED_BELOW,
    EXAMPLE_REPORT,
    Circle,
    FitItem,
    Line,
    create_fit_edge_tool,
    find_strokes,
    fit_report,
    load_sheet,
    save_skeleton,
)

SHIFT = 4  # cv2 draws at 1/16 px; pixel centres sit at +0.5 in tool coordinates
# cv2 renders thick anti-aliased strokes up to 0.2 px off their nominal centre.
TOLERANCE_PX = 0.3
# Two-tone drawings print dimensions in about this gray.
LIGHT = 130


def _at(x: float, y: float) -> tuple[int, int]:
    return round((x - 0.5) * 2**SHIFT), round((y - 0.5) * 2**SHIFT)


def _line(img, a, b, thickness=3, tone=0):
    cv2.line(img, _at(*a), _at(*b), tone, thickness, cv2.LINE_AA, SHIFT)


def _dimension(img, a, b, offset):
    """A light dimension `offset` px above a horizontal span, below if negative.

    Its extension lines start on the span and end 5 px past the dimension line.
    """
    (ax, ay), (bx, by) = a, b
    past = offset + math.copysign(5, offset)
    _line(img, (ax, ay), (ax, ay - past), thickness=2, tone=LIGHT)
    _line(img, (bx, by), (bx, by - past), thickness=2, tone=LIGHT)
    _line(img, (ax, ay - offset), (bx, by - offset), thickness=2, tone=LIGHT)


def _arc(img, centre, r, start_deg, end_deg, thickness=3):
    """Angles clockwise from +x in image axes, as cv2 draws them."""
    axes = (round(r * 2**SHIFT), round(r * 2**SHIFT))
    cv2.ellipse(
        img, _at(*centre), axes, 0, start_deg, end_deg, 0, thickness, cv2.LINE_AA, SHIFT
    )


def _sheet(tmp_path: Path, draw) -> Path:
    img = np.full((600, 600), 255, np.uint8)
    draw(img)
    path = tmp_path / "drawing.png"
    Image.fromarray(img).save(path)
    return path


def test_fillet_between_tangent_lines_ignores_leader_and_text(tmp_path):
    def draw(img):
        _line(img, (100, 300), (300, 300))
        _arc(img, (300, 340), 40, 270, 360)
        _line(img, (340, 340), (340, 520))
        _line(img, (250, 430), (390, 270), thickness=1)
        cv2.putText(img, "R40", (380, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)

    finding = find_strokes(
        load_sheet(_sheet(tmp_path, draw)), (240, 260, 400, 420), "circle"
    )

    circle = finding.strokes[0].model
    assert isinstance(circle, Circle)
    assert circle.r == pytest.approx(40, abs=TOLERANCE_PX)
    assert (circle.cx, circle.cy) == pytest.approx((300, 340), abs=TOLERANCE_PX)
    assert math.degrees(finding.strokes[0].end - finding.strokes[0].start) < 110


def test_concentric_circles_with_centre_cross(tmp_path):
    def draw(img):
        _arc(img, (200, 200), 35, 0, 360)
        _arc(img, (200, 200), 20, 0, 360)
        _line(img, (150, 200), (250, 200), thickness=1)
        _line(img, (200, 150), (200, 250), thickness=1)

    finding = find_strokes(
        load_sheet(_sheet(tmp_path, draw)), (150, 150, 250, 250), "circle"
    )

    radii = sorted(s.model.r for s in finding.strokes if isinstance(s.model, Circle))
    assert radii == pytest.approx([20, 35], abs=TOLERANCE_PX)
    assert all(s.end - s.start == pytest.approx(2 * math.pi) for s in finding.strokes)


def test_a_small_circle_crossed_as_thick_as_itself_stays_full(tmp_path):
    def draw(img):
        _arc(img, (300, 300), 12, 0, 360)
        _line(img, (270, 300), (330, 300))
        _line(img, (300, 270), (300, 330))

    finding = find_strokes(
        load_sheet(_sheet(tmp_path, draw)), (252, 252, 348, 348), "circle"
    )

    circles = [s for s in finding.strokes if isinstance(s.model, Circle)]
    assert len(circles) == 1
    assert circles[0].end - circles[0].start == pytest.approx(2 * math.pi)


def test_dashed_circle_is_full_and_broken(tmp_path):
    def draw(img):
        for start in range(0, 360, 15):
            _arc(img, (300, 300), 80, start, start + 9, thickness=2)

    finding = find_strokes(
        load_sheet(_sheet(tmp_path, draw)), (200, 200, 400, 400), "circle"
    )

    stroke = finding.strokes[0]
    assert stroke.model.r == pytest.approx(80, abs=TOLERANCE_PX)
    assert stroke.end - stroke.start == pytest.approx(2 * math.pi)
    assert 0.4 < stroke.coverage < DASHED_BELOW


def test_a_full_box_says_more_strokes_remain(tmp_path):
    def draw(img):
        for y in range(100, 400, 50):
            _line(img, (100, y), (400, y))

    finding = find_strokes(
        load_sheet(_sheet(tmp_path, draw)), (90, 90, 410, 410), "line"
    )

    assert len(finding.strokes) == 4
    assert finding.more


def test_straight_ink_asked_as_circle_gives_lines(tmp_path):
    def draw(img):
        _line(img, (100, 300), (500, 300))

    finding = find_strokes(
        load_sheet(_sheet(tmp_path, draw)), (150, 250, 450, 350), "circle"
    )

    assert finding.reason is not None and "straight" in finding.reason
    line = finding.strokes[0].model
    assert isinstance(line, Line)
    assert line.y == pytest.approx(300, abs=TOLERANCE_PX)


def test_report_says_where_the_box_cuts_a_line(tmp_path):
    path = _sheet(tmp_path, lambda img: _line(img, (100, 300), (400, 300)))
    boxes = [[150, 280, 350, 320], [50, 280, 350, 320], [50, 295, 450, 305]]
    items = [FitItem(box_px=box, kind="line") for box in boxes]

    report = fit_report("/work/drawing.png", path, items, None).splitlines()

    assert "the box cuts both ends" in report[2]
    assert "the box cuts the end at (349.5" in report[4]
    assert "cuts" not in report[6]


def test_a_flat_circle_candidate_gives_lines(tmp_path):
    path = _sheet(tmp_path, lambda img: _line(img, (100, 100), (400, 180)))

    finding = find_strokes(load_sheet(path), (200, 100, 260, 160), "circle")

    assert finding.reason is not None and "straight" in finding.reason
    assert isinstance(finding.strokes[0].model, Line)


def test_a_two_pixel_stroke_centred_on_a_pixel_edge_is_found(tmp_path):
    def draw(img):
        img[100:500, 200:202] = 20
        # Gray levels keep Otsu's threshold off the bar's own value.
        _line(img, (300, 100), (300, 500))

    finding = find_strokes(
        load_sheet(_sheet(tmp_path, draw)), (150, 150, 250, 450), "line"
    )

    line = finding.strokes[0].model
    assert isinstance(line, Line)
    assert line.x == pytest.approx(201, abs=0.1)


def test_empty_box_is_not_measured(tmp_path):
    finding = find_strokes(
        load_sheet(_sheet(tmp_path, lambda img: None)), (10, 10, 90, 90), "line"
    )

    assert finding.strokes == []
    assert finding.reason == "no ink in the box."


def test_light_dimension_lines_are_not_fitted_and_the_report_says_so(tmp_path):
    def draw(img):
        _dimension(img, (100, 300), (400, 300), 60)
        _dimension(img, (100, 400), (400, 400), -60)
        _line(img, (100, 300), (400, 300), thickness=2)
        _line(img, (100, 400), (400, 400), thickness=2)
        _line(img, (100, 300), (100, 400), thickness=2)
        _line(img, (400, 300), (400, 400), thickness=2)

    sheet = load_sheet(_sheet(tmp_path, draw))
    # The part's left edge runs on into its light extension line above.
    beside = find_strokes(sheet, (90, 210, 110, 360), "line")
    alone = find_strokes(sheet, (150, 225, 350, 255), "line")

    [edge] = beside.strokes
    assert isinstance(edge.model, Line) and edge.model.x == pytest.approx(100, abs=1)
    # Not up the extension line, which runs from y 300 to 235.
    assert min(edge.model.point(edge.start)[1], edge.model.point(edge.end)[1]) > 290
    assert beside.reason is None
    assert alone.strokes == []
    assert alone.reason == (
        "strokes here appear to be annotation and are not fitted. "
        "Could they be dimension lines?"
    )


def test_gray_dimensions_preserve_a_thinner_outer_edge_in_fit_and_skeleton(tmp_path):
    def draw(img):
        _dimension(img, (100, 200), (450, 200), 50)
        for left, top, right, bottom, width in [
            (100, 200, 450, 500, 2),
            (130, 230, 420, 470, 7),
        ]:
            for a, b in [
                ((left, top), (right, top)),
                ((right, top), (right, bottom)),
                ((right, bottom), (left, bottom)),
                ((left, bottom), (left, top)),
            ]:
                _line(img, a, b, thickness=width)
        _line(img, (100, 245), (130, 245))  # Connect the two real outlines.

    path = _sheet(tmp_path, draw)
    finding = find_strokes(load_sheet(path), (90, 260, 105, 450), "line")

    [edge] = finding.strokes
    assert isinstance(edge.model, Line)
    assert edge.model.x == pytest.approx(100, abs=TOLERANCE_PX)
    save_skeleton(path, tmp_path / "skeleton.npz")
    with np.load(tmp_path / "skeleton.npz") as points:
        outer = (
            (np.abs(points["x"] - 100) < 0.5)
            & (points["y"] > 260)
            & (points["y"] < 450)
        )
        assert outer.sum() > 180


@pytest.fixture
def workspace(tmp_path: Path) -> SandboxWorkdir:
    def draw(img):
        _line(img, (150, 100), (150, 500))
        _line(img, (213, 100), (213, 500))
        _line(img, (150, 300), (213, 300), thickness=1)
        _arc(img, (400, 300), 40, 0, 360)

    _sheet(tmp_path, draw)
    return SandboxWorkdir(host_bind_dir=tmp_path)


def test_tool_reports_spacing_and_millimetres(workspace):
    fit_edge = create_fit_edge_tool(workspace)

    report = fit_edge.invoke(
        {
            "image_path": "/work/drawing.png",
            "items": [
                {"box_px": [120, 120, 240, 480], "kind": "line"},
                {"box_px": [340, 240, 460, 360], "kind": "circle"},
            ],
            "mm_per_px": 0.5,
        }
    )

    spacing = re.search(r"spacing #\d to #\d: ([\d.]+) px = ([\d.]+) mm", report)
    assert spacing is not None
    assert float(spacing[1]) == pytest.approx(63, abs=0.1)
    assert float(spacing[2]) == pytest.approx(31.5, abs=0.05)
    radius = re.search(r"R [\d.]+ ±[\d.]+ px = R ([\d.]+) ±", report)
    assert radius is not None
    assert float(radius[1]) == pytest.approx(20, abs=0.15)
    assert "full circle" in report


def test_tool_schema_takes_image_items_and_scale(workspace):
    schema = create_fit_edge_tool(workspace).get_input_jsonschema()

    assert set(schema["properties"]) == {"image_path", "items", "mm_per_px"}
    assert set(schema["required"]) == {"image_path", "items"}


def test_tool_refuses_a_file_that_is_not_an_image(workspace, tmp_path):
    (tmp_path / "view.dxf").write_text("0\nSECTION\n")

    with pytest.raises(ToolFeedbackError, match="ezdxf"):
        create_fit_edge_tool(workspace).invoke(
            {
                "image_path": "/work/view.dxf",
                "items": [{"box_px": [0, 0, 10, 10], "kind": "line"}],
            }
        )


def test_the_documented_example_has_the_report_format(tmp_path):
    img = np.full((700, 900), 255, np.uint8)
    _arc(img, (360, 260), 40, 0, 360, thickness=2)
    for start in range(0, 360, 24):
        _arc(img, (360, 260), 24, start, start + 12, thickness=1)
    _line(img, (300, 260), (420, 260), thickness=1)
    _line(img, (360, 200), (360, 320), thickness=1)
    _line(img, (60, 420), (150, 420), thickness=2)
    _arc(img, (150, 450), 30, 270, 360, thickness=2)
    _line(img, (180, 450), (180, 560), thickness=2)
    _line(img, (80, 110), (300, 110), thickness=2)
    _line(img, (80, 129), (300, 129), thickness=2)
    path = tmp_path / "view_drawing.png"
    Image.fromarray(img).save(path)
    boxes = [
        ([300, 200, 420, 320], "circle"),
        ([120, 400, 200, 480], "circle"),
        ([90, 95, 290, 145], "line"),
    ]
    items = [FitItem(box_px=box, kind=kind) for box, kind in boxes]

    report = fit_report("/work/inputs/view_drawing.png", path, items, 0.25)

    def shape(text: str) -> str:
        return re.sub(r"\d+(\.\d+)?", "N", text)

    assert shape(report) == shape(EXAMPLE_REPORT)


def test_skeleton_gives_position_and_direction(tmp_path):
    def draw(img):
        _line(img, (150, 100), (150, 500))
        _line(img, (200, 300), (500, 300))
        _line(img, (300, 560), (500, 460))

    save_skeleton(_sheet(tmp_path, draw), tmp_path / "skeleton.npz")
    points = np.load(tmp_path / "skeleton.npz")
    x, y, angle = points["x"], points["y"], points["angle"]

    assert set(points.files) == {"x", "y", "angle"}
    assert np.all(np.diff(y) >= 0)
    preview = np.asarray(Image.open(tmp_path / "skeleton.png"))
    expected = np.full((600, 600), 255, np.uint8)
    expected[y.astype(int), x.astype(int)] = 0
    np.testing.assert_array_equal(preview, expected)
    for at, expected_angle in (((150, 200), 90), ((350, 300), 0), ((400, 510), 26.57)):
        near = np.hypot(x - at[0], y - at[1]) < 1
        assert near.any()
        turn = np.abs(angle[near] - expected_angle) % 180
        assert np.minimum(turn, 180 - turn).max() < 2


def test_skeleton_leaves_out_light_dimension_lines(tmp_path):
    def draw(img):
        _dimension(img, (100, 300), (400, 300), 60)
        _dimension(img, (100, 400), (400, 400), -60)
        _line(img, (100, 300), (400, 300), thickness=2)
        _line(img, (100, 400), (400, 400), thickness=2)
        cv2.putText(img, "300", (220, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.8, LIGHT, 2)

    save_skeleton(_sheet(tmp_path, draw), tmp_path / "skeleton.npz")
    points = np.load(tmp_path / "skeleton.npz")
    x, y = points["x"], points["y"]

    assert set(points.files) == {"x", "y", "angle"}
    for edge in (300, 400):
        along = np.abs(y - edge) < 1
        assert x[along].min() < 105 and x[along].max() > 395
    assert not ((y < 290) | (y > 410)).any()


@pytest.mark.parametrize("tone", [0, LIGHT])
def test_skeleton_and_fit_edge_exclude_annotation(tmp_path, tone):
    image = np.full((300, 300), 255, np.uint8)
    cv2.rectangle(image, (40, 60), (150, 240), 0, 5)
    for y in (60, 240):
        cv2.line(image, (150, y), (270, y), tone, 1)
    cv2.line(image, (250, 50), (250, 250), tone, 1)
    path = tmp_path / "drawing.png"
    Image.fromarray(image).save(path)
    save_skeleton(path, tmp_path / "skeleton.npz")
    points = np.load(tmp_path / "skeleton.npz")
    assert len(points["x"]) > 400  # pure black/white input must not threshold to empty
    assert not ((points["x"] > 200) & (points["y"] > 80) & (points["y"] < 220)).any()
    sheet = load_sheet(path)
    assert not sheet.ink[150, 250]
    finding = find_strokes(sheet, (245, 80, 255, 220), "line")
    assert finding.strokes == []
    assert finding.reason == (
        "strokes here appear to be annotation and are not fitted. "
        "Could they be dimension lines?"
    )
    edge = find_strokes(sheet, (35, 80, 45, 220), "line").strokes[0].model
    assert isinstance(edge, Line) and edge.x == pytest.approx(40.5, abs=0.3)


def test_loading_an_overwritten_image_refreshes_the_cached_sheet(tmp_path):
    path = _sheet(tmp_path, lambda img: _line(img, (100, 200), (500, 200)))
    before = load_sheet(path)
    assert load_sheet(path) is before

    _sheet(tmp_path, lambda img: _line(img, (100, 400), (500, 400)))
    after = load_sheet(path)

    assert before.ink[200, 300] and not before.ink[400, 300]
    assert after.ink[400, 300] and not after.ink[200, 300]
