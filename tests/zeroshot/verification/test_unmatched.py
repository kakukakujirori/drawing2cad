import cv2
import ezdxf
import numpy as np
import pytest

from zeroshot.pipeline.verification.drawing_diff.unmatched import (
    ANNOTATION_GRAY,
    BOTH_BLUE,
    COLORS,
    INPUT_GRAY,
    TANGENT_TEAL,
    Unmatched,
    describe_unmatched,
    draw_unmatched,
    find_unmatched,
    measure_material_error,
    output_curves,
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


def test_descriptions_define_the_kind_and_unit_for_each_group():
    xy = np.array([[10, 20], [11, 20]])
    descriptions = describe_unmatched(
        [
            Unmatched("missing", xy),
            Unmatched("extra", xy, local=True),
            Unmatched("extra", xy, material=True),
        ]
    )

    assert [(item["kind"], item["unit"]) for item in descriptions] == [
        ("lines", "px"),
        ("lines", "px"),
        ("material", "px²"),
    ]
    assert all("local" not in item for item in descriptions)


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


def test_optional_tangent_cannot_overpaint_a_cluster_number():
    drawing = _lines(*OUTLINE, INNER)
    output = _lines(*OUTLINE)
    groups = find_unmatched(drawing, output)
    gray = np.uint8(~drawing) * 255
    baseline = draw_unmatched(gray, output, groups)

    # Red above the defect belongs to its number, not its highlight band.
    label = np.all(baseline == COLORS["red"], axis=-1)
    label[groups[0].box_px[1] - 10 :] = False
    positions = np.argwhere(label)
    assert len(positions)
    y, x = positions[len(positions) // 2]
    tangent = _lines(((int(x), 10), (int(x), int(y))))

    image = draw_unmatched(gray, output, groups, tangent=tangent)

    assert (label & tangent).any()
    np.testing.assert_array_equal(image[label], baseline[label])
    assert tuple(image[10, x]) == TANGENT_TEAL


def test_length_does_not_depend_on_stroke_width():
    thick = find_unmatched(_lines(*OUTLINE, INNER, width=7), _lines(*OUTLINE))
    thin = find_unmatched(_lines(*OUTLINE, width=7), _lines(*OUTLINE, INNER))

    assert abs(thick[0].size_px - thin[0].size_px) <= 6


def test_lines_beyond_the_part_and_short_marks_are_not_listed():
    far = ((40, 195), (160, 195))  # a dimension line below the part
    mark = ((95, 70), (115, 70))  # shorter than any listed group
    drawing = _lines(*OUTLINE, far, mark, width=5)

    assert find_unmatched(drawing, _lines(*OUTLINE)) == []


FRAME = (
    ((20, 20), (280, 20)),
    ((280, 20), (280, 280)),
    ((280, 280), (20, 280)),
    ((20, 280), (20, 20)),
)


def test_a_line_that_drifts_off_is_one_group_measured_whole():
    """Only its last 30 px lie beyond UNMATCHED_PX, too short to list alone."""
    drawing = _lines(*FRAME, ((40, 150), (260, 150)), size=300)
    output = _lines(*FRAME, ((40, 150), (260, 164)), size=300)

    groups = find_unmatched(drawing, output)

    extra = [g for g in groups if g.direction == "extra"]
    missing = [g for g in groups if g.direction == "missing"]
    assert len(extra) == 1 and len(missing) == 1
    # Beyond WEAK_PX from about x = 120 on.
    assert extra[0].box_px[0] < 130 and extra[0].box_px[2] >= 260


def test_diagonal_and_dashed_lines_count_their_full_length():
    diagonal = ((60, 60), (102, 102))  # 59 px long, 43 skeleton pixels
    dashes = [((x, 200), (x + 5, 200)) for x in range(60, 120, 9)]
    straight = ((60, 240), (112, 240))  # 52 px, the most skeleton pixels
    drawing = _lines(*FRAME, diagonal, *dashes, straight, size=300)

    groups = find_unmatched(drawing, _lines(*FRAME, size=300))

    assert [g.direction for g in groups] == ["missing"] * 3
    assert groups[-1].box_px[1] == 240


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


def test_region_score_counts_all_groups_beyond_the_four_displayed_clusters():
    drawing = np.zeros((500, 500), np.uint8)
    cv2.rectangle(drawing, (20, 20), (100, 100), 1, 1)
    output = drawing.copy()
    for x in (180, 280, 380):
        for y in (200, 300):
            cv2.rectangle(output, (x, y), (x + 50, y + 50), 1, 1)
    drawing, output = drawing.astype(bool), output.astype(bool)

    groups = find_unmatched(drawing, output)
    error = measure_material_error(drawing, output)

    assert len(groups) == 4 and all(group.material for group in groups)
    assert error["extra_px2"] > sum(group.size_px for group in groups)
    assert error["missing_px2"] == 0


def test_open_or_thin_input_has_unavailable_region_error():
    error = measure_material_error(_lines(INNER), _lines(*OUTLINE))

    assert error["status"] == "unavailable"
    assert error["ratio"] is None
    assert "cannot be filled" in error["reason"]


# DXF model units to drawing pixel boundaries: one unit per pixel.
PIXELS = np.array([[1.0, 0, 0.5], [0, 1.0, 0.5], [0, 0, 1.0]])


def _curves(*segments, hidden=()):
    """Output curves of visible and hidden line segments, in drawing pixels."""
    doc = ezdxf.new(setup=True)
    for start, end in segments:
        doc.modelspace().add_line(start, end)
    for start, end in hidden:
        doc.modelspace().add_line(start, end, dxfattribs={"linetype": "HIDDEN"})
    return output_curves(doc.modelspace(), PIXELS)


def test_output_curves_are_the_visible_and_hidden_lines_only():
    doc = ezdxf.new(setup=True)
    doc.layers.add("dashed", linetype="HIDDEN")
    doc.layers.add("TANGENT")
    msp = doc.modelspace()
    msp.add_line((0, 0), (10, 0))
    msp.add_line((0, 5), (10, 5), dxfattribs={"linetype": "HIDDEN"})
    msp.add_line((0, 10), (10, 10), dxfattribs={"layer": "dashed"})
    msp.add_line((0, 15), (10, 15), dxfattribs={"layer": "TANGENT"})
    msp.add_line((0, 20), (10, 20), dxfattribs={"linetype": "CENTER"})

    curves = output_curves(msp, PIXELS)

    assert sorted((xy[0][1], style) for xy, style in curves) == [
        (0, "visible"),
        (5, "hidden"),
        (10, "hidden"),
    ]


@pytest.mark.parametrize(
    "depth, length, hidden, listed",
    [
        (5, 20, False, False),
        (6, 20, False, True),
        (6, 4, False, False),
        (6, 20, True, False),
        (7, 20, True, True),
    ],
)
def test_an_output_stretch_is_listed_once_it_reaches_fine_px(
    depth, length, hidden, listed
):
    """Far too short for a coarse group, yet an output line no input line explains.

    A hidden one must reach further, past half the input's dash gaps.
    """
    copy = ((100, 100 + depth), (100 + length, 100 + depth))
    drawing, output = _lines(*OUTLINE, INNER), _lines(*OUTLINE, INNER, copy)
    curves = (
        _curves(*OUTLINE, INNER, hidden=[copy])
        if hidden
        else _curves(*OUTLINE, INNER, copy)
    )

    groups = find_unmatched(drawing, output, curves)

    assert find_unmatched(drawing, output) == []
    if listed:
        assert len(groups) == 1 and groups[0].local and groups[0].direction == "extra"
        assert groups[0].box_px == (100, 100 + depth, 101 + length, 101 + depth)
    else:
        assert groups == []


@pytest.mark.parametrize("top, listed", [(60, 0), (40, 2)])
def test_a_hole_drawn_through_instead_of_blind_is_extra_but_dash_gaps_are_not(
    top, listed
):
    """The input draws a blind hole's sides dashed, from y = 60 down."""
    drawing = _lines(*OUTLINE, width=3).astype(np.uint8)
    for x in (90, 110):
        for y in range(60, 100, 13):
            cv2.line(drawing, (x, y), (x, min(y + 8, 100)), 1, 1)
    sides = [((x, top), (x, 100)) for x in (90, 110)]
    output = _lines(*OUTLINE, *sides)

    groups = find_unmatched(
        drawing.astype(bool), output, _curves(*OUTLINE, hidden=sides)
    )

    assert len(groups) == listed
    for group in groups:
        assert group.local and group.direction == "extra"
        assert 40 < group.box_px[1] and group.box_px[3] < 60


@pytest.mark.parametrize("finish", ["round", "chamfer", "concave"])
def test_edge_finishes_are_fine_extras_beside_coarse_clutter(finish):
    drawing = np.zeros((200, 200), np.uint8)
    output = _lines(*OUTLINE).astype(np.uint8)
    doc = ezdxf.new()
    msp = doc.modelspace()
    for start, end in OUTLINE:
        msp.add_line(start, end)
    if finish == "concave":
        cv2.rectangle(drawing, (40, 40), (160, 160), 1, 5)
        output[40:65, 40:65] = 0
        cv2.ellipse(output, (40, 40), (25, 25), 0, 0, 90, 1, 1)
        msp.delete_all_entities()
        for start, end in (
            ((65, 40), (160, 40)),
            ((160, 40), (160, 160)),
            ((160, 160), (40, 160)),
            ((40, 160), (40, 65)),
        ):
            msp.add_line(start, end)
        msp.add_arc((40, 40), 25, 0, 90)
    elif finish == "chamfer":
        points = np.array(
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
        cv2.polylines(drawing, [points], True, 1, 5)
    else:
        for start, end in (
            ((75, 40), (125, 40)),
            ((160, 75), (160, 125)),
            ((125, 160), (75, 160)),
            ((40, 125), (40, 75)),
        ):
            cv2.line(drawing, start, end, 1, 5)
        for center, start in [
            ((75, 75), 180),
            ((125, 75), 270),
            ((125, 125), 0),
            ((75, 125), 90),
        ]:
            cv2.ellipse(drawing, center, (35, 35), 0, start, start + 90, 1, 5)

    drawing, output = drawing.astype(bool), output.astype(bool)
    # Small finishes are below the existing whole-view length/area thresholds.
    assert find_unmatched(drawing, output) == []
    groups = find_unmatched(drawing, output, output_curves(msp, PIXELS))
    assert len(groups) == (1 if finish == "concave" else 2)
    assert all(g.local and g.direction == "extra" for g in groups)

    # A long unrelated line must not displace the small corner errors, or be
    # duplicated as several short local groups at its ends.
    cv2.line(output.view(np.uint8), (100, 50), (100, 150), 1, 1)
    msp.add_line((100, 50), (100, 150))
    groups = find_unmatched(drawing, output, output_curves(msp, PIXELS))
    assert sum(g.local for g in groups) == (1 if finish == "concave" else 2)
    assert sum(not g.local for g in groups) == 1

    # Five independent large defects fill four coarse slots before local finishes.
    for x in (65, 80, 120, 135):
        cv2.line(output.view(np.uint8), (x, 80), (x, 135), 1, 1)
        msp.add_line((x, 80), (x, 135))
    groups = find_unmatched(drawing, output, output_curves(msp, PIXELS))
    assert sum(not g.local for g in groups) == 4
    assert sum(g.local for g in groups) == (1 if finish == "concave" else 2)
    assert not any(g.local for g in groups[:4])


def test_a_fine_tail_and_coarse_run_are_displayed_as_one_defect():
    # The tail lies 5 px off, within WEAK_PX: only its fine stretch reaches it.
    drawing = _lines(
        *FRAME,
        ((30, 105), (60, 105)),
        ((60, 105), (80, 120)),
        ((80, 120), (260, 120)),
        size=300,
    )
    line = ((30, 100), (260, 100))
    output = _lines(*FRAME, line, size=300)

    groups = find_unmatched(drawing, output, _curves(*FRAME, line))

    extra = [g for g in groups if g.direction == "extra"]
    assert len(extra) == 1 and not extra[0].local
    assert extra[0].box_px == (30, 100, 261, 101)


@pytest.mark.parametrize("offset", [(0, 0), (2, 2), (3, 0), (0, 3)])
def test_fine_comparison_tolerates_registration_stroke_width_and_annotation(offset):
    """Offsets within matched-line noise stay below FINE_PX even at the corners,
    which the centre line of a thick stroke cuts."""
    drawing = _lines(*OUTLINE, width=7).astype(np.uint8)
    drawing = cv2.warpAffine(
        drawing, np.float32([[1, 0, offset[0]], [0, 1, offset[1]]]), (200, 200)
    )
    cv2.line(drawing, (15, 20), (175, 20), 1, 3)
    cv2.putText(drawing, "100", (80, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1, 1)
    output = _lines(*OUTLINE)

    coarse = find_unmatched(drawing.astype(bool), output)
    groups = find_unmatched(drawing.astype(bool), output, _curves(*OUTLINE))

    assert not any(group.local for group in groups)
    assert describe_unmatched(groups) == describe_unmatched(coarse)


@pytest.mark.parametrize("shape", ["horizontal", "diagonal"])
def test_a_remote_coarse_error_does_not_admit_disconnected_local_fragments(shape):
    if shape == "horizontal":
        size, lo, hi = 500, 20, 480
        common = [((30, 100), (30, 60))]
        actual = [((30, 100), (460, 100))]
        expected = [
            ((30, 106), (350, 106)),
            ((350, 106), (370, 120)),
            ((370, 120), (460, 120)),
            ((30, 106), (30, 100)),
        ]
    else:
        size, lo, hi = 380, 10, 350
        common = [
            line
            for x, y in [(80, 137), (150, 207)]
            for line in [((x, y), (x + 20, y)), ((x, y), (x, y + 20))]
        ]
        actual = [((60, 60), (280, 280))]
        expected = [
            ((60, 68), (200, 208)),
            ((200, 208), (210, 235)),
            ((210, 235), (280, 305)),
        ]
    frame = [
        ((lo, lo), (hi, lo)),
        ((hi, lo), (hi, hi)),
        ((hi, hi), (lo, hi)),
        ((lo, hi), (lo, lo)),
    ]
    drawing = _lines(*frame, *common, *expected, size=size)
    output = _lines(*frame, *common, *actual, size=size)

    coarse = find_unmatched(drawing, output)
    groups = find_unmatched(drawing, output, _curves(*frame, *common, *actual))

    assert len(coarse) == 2
    assert not any(g.local for g in groups)
    # Its fine stretch lengthens the coarse group where the line lies between
    # SUPPORT_PX and WEAK_PX of the input; no disconnected fragment joins it.
    assert [g.direction for g in groups] == [g.direction for g in coarse]
    for group, alone in zip(groups, coarse, strict=True):
        assert np.abs(np.subtract(group.box_px, alone.box_px)).max() <= 2


def test_nearby_cluster_labels_remain_readable_and_inside_the_image(monkeypatch):
    origins = []
    original = cv2.putText

    def record(image, text, origin, font, scale, color, thickness):
        if thickness == 4:
            origins.append((text, origin))
        return original(image, text, origin, font, scale, color, thickness)

    monkeypatch.setattr(cv2, "putText", record)
    groups = [
        Unmatched("extra", np.array([[x, y], [x + 2, y + 5]]))
        for x, y in [(260, 30), (265, 35), (264, 260), (267, 265), (270, 268)]
    ]
    draw_unmatched(
        np.full((300, 300), 255, np.uint8), np.zeros((300, 300), bool), groups
    )
    boxes = []
    for text, (x, y) in origins:
        (width, height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 10
        )
        box = (x, y - height, x + width, y + baseline)
        assert 0 <= box[0] < box[2] <= 300 and 0 <= box[1] < box[3] <= 300
        assert all(
            box[2] <= b[0] or box[0] >= b[2] or box[3] <= b[1] or box[1] >= b[3]
            for b in boxes
        )
        boxes.append(box)
    assert len(boxes) == 5


def test_region_score_keeps_a_thin_misalignment_instead_of_reporting_zero():
    drawing, output = np.zeros((350, 450), np.uint8), np.zeros((350, 450), np.uint8)
    cv2.rectangle(drawing, (80, 60), (320, 290), 1, 1)
    cv2.rectangle(output, (88, 60), (328, 290), 1, 1)

    error = measure_material_error(drawing.astype(bool), output.astype(bool))

    assert error["missing_px2"] + error["extra_px2"] == 3696
    assert error["ratio"] > 0.05


def test_thin_input_lines_keep_their_width_and_still_mark_shared_strokes():
    drawing = _lines(*OUTLINE, INNER)  # one pixel wide, like small print
    output = _lines(*OUTLINE)
    image = draw_unmatched(np.uint8(~drawing) * 255, output, [])

    # Widening would fill small text; the input-only line stays one pixel wide.
    gray = np.all(image == INPUT_GRAY, axis=-1)
    assert gray[100, 70:130].all() and not gray[[99, 101], 70:130].any()
    # The output stroke along an input line is still drawn by both.
    assert np.all(image[40, 70:130] == BOTH_BLUE)


def test_annotation_is_never_missing_and_an_output_along_it_is_not_extra():
    leader = ((60, 100), (190, 100))  # crosses the part into the margin
    note = ((80, 80), (150, 80))  # an output edge an annotation line happens to trace
    drawing = _lines(*OUTLINE, leader, note, width=3)
    annotation = _lines(leader, note, width=3) & ~_lines(*OUTLINE, width=3)
    output = _lines(*OUTLINE, note)

    assert find_unmatched(drawing, output) != []
    assert find_unmatched(drawing, output, annotation=annotation) == []

    image = draw_unmatched(np.uint8(~drawing) * 255, output, [], annotation)
    assert np.all(image[100, 170:190] == ANNOTATION_GRAY)
