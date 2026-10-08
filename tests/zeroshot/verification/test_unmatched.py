from dataclasses import replace

import cv2
import ezdxf
import numpy as np
import pytest

from zeroshot.pipeline.verification.drawing_diff.unmatched import (
    COLORS,
    INPUT_GRAY,
    Unmatched,
    _local_detail,
    describe_unmatched,
    detail_geometry,
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


@pytest.mark.parametrize("length, expected", [(23, False), (24, True)])
def test_local_labels_have_a_minimum_length(length, expected):
    corner = (((70, 100), (82, 100)), ((70, 100), (70, 100 + length - 12)))
    drawing, output = _lines(*OUTLINE), _lines(*OUTLINE, *corner)
    msp = ezdxf.new().modelspace()
    for start, end in (*OUTLINE, *corner):
        msp.add_line(start, end)
    geometry = detail_geometry(
        msp, np.array([[1.0, 0, 0.5], [0, 1.0, 0.5], [0, 0, 1.0]]), drawing.shape
    )
    groups = find_unmatched(drawing, output, geometry)

    assert bool(groups) is expected
    assert all(g.local and g.size_px == length for g in groups)


@pytest.mark.parametrize("finish", ["round", "chamfer", "concave"])
def test_local_edge_finishes_survive_coarse_filtering_and_clutter(finish):
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
    transform = np.array([[1.0, 0, 0.5], [0, 1, 0.5], [0, 0, 1]])
    masks = detail_geometry(msp, transform, drawing.shape)
    groups = find_unmatched(drawing, output, masks)
    assert len(groups) == (1 if finish == "concave" else 2)
    assert all(g.local and g.direction == "extra" for g in groups)

    # A long unrelated line must not displace the small corner errors, or be
    # duplicated as several short local groups at its ends.
    cv2.line(output.view(np.uint8), (100, 50), (100, 150), 1, 1)
    msp.add_line((100, 50), (100, 150))
    masks = detail_geometry(msp, transform, drawing.shape)
    groups = find_unmatched(drawing, output, masks)
    assert sum(g.local for g in groups) == (1 if finish == "concave" else 2)
    assert sum(not g.local for g in groups) == 1

    # Five independent large defects fill four coarse slots before local finishes.
    for x in (65, 80, 120, 135):
        cv2.line(output.view(np.uint8), (x, 80), (x, 135), 1, 1)
        msp.add_line((x, 80), (x, 135))
    groups = find_unmatched(
        drawing,
        output,
        detail_geometry(msp, transform, drawing.shape),
    )
    assert sum(not g.local for g in groups) == 4
    assert sum(g.local for g in groups) == (1 if finish == "concave" else 2)
    assert not any(g.local for g in groups[:4])


@pytest.mark.parametrize("start, end", [(30, 71), (60, 81)])
def test_a_local_tail_and_coarse_run_are_displayed_as_one_defect(start, end):
    outline = (
        ((20, 20), (280, 20)),
        ((280, 20), (280, 280)),
        ((280, 280), (20, 280)),
        ((20, 280), (20, 20)),
    )
    drawing = _lines(
        *outline,
        ((30, 108), (60, 108)),
        ((60, 108), (80, 120)),
        ((80, 120), (260, 120)),
        size=300,
    )
    output = _lines(*outline, ((30, 100), (260, 100)), size=300)
    local = np.zeros_like(output)
    local[100, start:end] = True

    msp = ezdxf.new().modelspace()
    for a, b in (*outline, ((30, 100), (260, 100))):
        msp.add_line(a, b)
    geometry = detail_geometry(
        msp, np.array([[1.0, 0, 0.5], [0, 1, 0.5], [0, 0, 1]]), drawing.shape
    )
    groups = find_unmatched(drawing, output, replace(geometry, mask=local))

    extra = [g for g in groups if g.direction == "extra"]
    assert len(extra) == 1 and not extra[0].local
    assert extra[0].box_px == (start, 100, 261, 101)


@pytest.mark.parametrize("offset", [(0, 0), (2, 2), (6, 6), (6, 0), (0, 6)])
def test_local_comparison_tolerates_registration_stroke_width_and_annotation(offset):
    doc = ezdxf.new()
    for start, end in OUTLINE:
        doc.modelspace().add_line(start, end)
    # Hidden geometry stays outside the local comparison.
    doc.modelspace().add_arc((60, 60), 8, 0, 90, dxfattribs={"linetype": "HIDDEN"})
    drawing = _lines(*OUTLINE, width=7).astype(np.uint8)
    drawing = cv2.warpAffine(
        drawing, np.float32([[1, 0, offset[0]], [0, 1, offset[1]]]), (200, 200)
    )
    cv2.line(drawing, (15, 20), (175, 20), 1, 3)
    cv2.putText(drawing, "100", (80, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1, 1)
    output = _lines(*OUTLINE)
    masks = detail_geometry(
        doc.modelspace(),
        np.array([[1.0, 0, 0.5], [0, 1, 0.5], [0, 0, 1]]),
        drawing.shape,
    )
    coarse = find_unmatched(drawing.astype(bool), output)
    groups = find_unmatched(drawing.astype(bool), output, masks)
    assert not any(group.local for group in groups)
    assert describe_unmatched(groups) == describe_unmatched(coarse)


def test_a_high_curvature_radius_error_is_a_local_defect():
    drawing, output = np.zeros((300, 300), np.uint8), np.zeros((300, 300), np.uint8)
    cv2.circle(drawing, (150, 150), 26, 1, 3)
    cv2.circle(output, (150, 150), 20, 1, 1)
    doc = ezdxf.new()
    doc.modelspace().add_circle((150, 150), 20)
    masks = detail_geometry(
        doc.modelspace(),
        np.array([[1.0, 0, 0.5], [0, 1, 0.5], [0, 0, 1]]),
        drawing.shape,
    )

    groups = find_unmatched(drawing.astype(bool), output.astype(bool), masks)

    assert len(groups) == 1 and groups[0].local
    assert groups[0].direction == "extra"


def test_a_small_radius_error_is_filtered_even_when_its_curvature_qualifies():
    images = []
    msp = ezdxf.new().modelspace()
    for radius in (46, 35):
        image = np.zeros((300, 300), np.uint8)
        for start, end in (
            ((40 + radius, 40), (260, 40)),
            ((260, 40), (260, 260)),
            ((260, 260), (40, 260)),
            ((40, 260), (40, 40 + radius)),
        ):
            cv2.line(image, start, end, 1, 1)
            if radius == 35:
                msp.add_line(start, end)
        cv2.ellipse(
            image, (40 + radius, 40 + radius), (radius, radius), 0, 180, 270, 1, 1
        )
        images.append(image.astype(bool))
    msp.add_arc((75, 75), 35, 180, 270)
    drawing, output = images
    geometry = detail_geometry(
        msp, np.array([[1.0, 0, 0.5], [0, 1.0, 0.5], [0, 0, 1.0]]), drawing.shape
    )

    angles = np.radians(np.linspace(208, 248, 15))
    candidate = Unmatched(
        "extra", 75 + 35 * np.column_stack((np.cos(angles), np.sin(angles)))
    )

    assert _local_detail(candidate, geometry)
    assert find_unmatched(drawing, output, geometry) == []


def test_an_error_on_only_one_side_of_a_corner_is_not_a_local_finish():
    frame = (
        ((10, 10), (240, 10)),
        ((240, 10), (240, 240)),
        ((240, 240), (10, 240)),
        ((10, 240), (10, 10)),
    )
    output_segments = (*frame, ((70, 70), (130, 70)), ((70, 70), (70, 120)))
    drawing = _lines(
        *frame,
        ((70, 70), (70, 120)),
        ((70, 70), (75, 76)),
        ((75, 76), (105, 76)),
        ((105, 76), (110, 70)),
        ((110, 70), (130, 70)),
        size=260,
    )
    output = _lines(*output_segments, size=260)
    msp = ezdxf.new().modelspace()
    for start, end in output_segments:
        msp.add_line(start, end)
    geometry = detail_geometry(
        msp, np.array([[1.0, 0, 0.5], [0, 1.0, 0.5], [0, 0, 1.0]]), drawing.shape
    )

    candidate = Unmatched(
        "extra", np.column_stack((np.arange(78, 106), np.full(28, 100)))
    )
    assert not _local_detail(candidate, geometry)
    assert find_unmatched(drawing, output, geometry) == []


@pytest.mark.parametrize("angle, expected", [(20, False), (40, True), (60, True)])
@pytest.mark.parametrize("reverse", [False, True])
def test_corner_detection_ignores_dxf_entity_direction(angle, expected, reverse):
    msp = ezdxf.new().modelspace()
    ends = [
        np.array([30.0, 0]),
        30 * np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))]),
    ]
    points = []
    for end in ends:
        msp.add_line(end if reverse else (0, 0), (0, 0) if reverse else end)
        points.extend(np.linspace(end / 4, end, 30) + [100, 100])
    geometry = detail_geometry(
        msp, np.array([[1.0, 0, 100.5], [0, 1.0, 100.5], [0, 0, 1.0]]), (200, 200)
    )

    assert _local_detail(Unmatched("extra", np.array(points)), geometry) is expected


def test_a_crossing_line_does_not_make_a_straight_error_a_corner_error():
    msp = ezdxf.new().modelspace()
    msp.add_line((100.2, 80), (100.2, 120))
    msp.add_line((80, 100), (120, 100))
    geometry = detail_geometry(
        msp, np.array([[1.0, 0, 0.5], [0, 1.0, 0.5], [0, 0, 1.0]]), (200, 200)
    )
    group = Unmatched("extra", np.column_stack((np.full(31, 100), np.arange(85, 116))))

    assert not _local_detail(group, geometry)


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
    msp = ezdxf.new().modelspace()
    for start, end in frame + common + actual:
        msp.add_line(start, end)
    geometry = detail_geometry(
        msp, np.array([[1.0, 0, 0.5], [0, 1.0, 0.5], [0, 0, 1.0]]), drawing.shape
    )

    coarse = find_unmatched(drawing, output)
    groups = find_unmatched(drawing, output, geometry)

    assert len(coarse) == 2
    assert not any(g.local for g in groups)
    assert describe_unmatched(groups) == describe_unmatched(coarse)


@pytest.mark.parametrize(
    "linetype,layer_linetype,selected",
    [
        ("CONTINUOUS", "CONTINUOUS", True),
        ("HIDDEN", "CONTINUOUS", False),
        ("BYLAYER", "CONTINUOUS", True),
        ("BYLAYER", "HIDDEN", False),
    ],
)
def test_detail_regions_select_continuous_curved_ellipse_tips(
    linetype, layer_linetype, selected
):
    doc = ezdxf.new(setup=True)
    doc.layers.add("test", linetype=layer_linetype)
    doc.modelspace().add_ellipse(
        (250, 125),
        major_axis=(200, 0),
        ratio=0.1,
        dxfattribs={"linetype": linetype, "layer": "test"},
    )
    geometry = detail_geometry(
        doc.modelspace(),
        np.array([[1.0, 0, 0.5], [0, 1, 0.5], [0, 0, 1]]),
        (250, 500),
    )

    assert bool(geometry.visible.any()) is selected
    assert bool(geometry.mask[125, 450]) is selected
    assert bool(geometry.mask[125, 50]) is selected
    # The ellipse's broad, nearly straight sides are outside the tip ROIs.
    assert not geometry.mask[145, 250]


@pytest.mark.parametrize("scale, selected", [(0.5, True), (1.0, True), (5.0, False)])
def test_curvature_is_measured_in_drawing_pixels(scale, selected):
    doc = ezdxf.new()
    doc.modelspace().add_circle((0, 0), 20)
    geometry = detail_geometry(
        doc.modelspace(),
        np.array([[scale, 0, 150.5], [0, scale, 150.5], [0, 0, 1]]),
        (300, 300),
    )

    assert bool(geometry.mask.any()) is selected


def test_a_straight_spline_only_selects_its_angular_join():
    doc = ezdxf.new()
    doc.modelspace().add_open_spline([(40, 100), (80, 100), (120, 100), (160, 100)])
    doc.modelspace().add_line((160, 100), (160, 40))
    geometry = detail_geometry(
        doc.modelspace(),
        np.array([[1.0, 0, 0.5], [0, 1, 0.5], [0, 0, 1]]),
        (200, 200),
    )

    assert geometry.mask[100, 160]
    assert not geometry.mask[100, 100]


def test_a_shifted_nearly_straight_curve_is_not_a_local_finish():
    drawing, output = np.zeros((300, 400), np.uint8), np.zeros((300, 400), np.uint8)
    for image in (drawing, output):
        cv2.rectangle(image, (30, 25), (370, 270), 1, 1)
    cv2.ellipse(output, (200, -900), (1000, 1000), 0, 85, 95, 1, 1)
    cv2.ellipse(drawing, (200, -894), (1000, 1000), 0, 85, 95, 1, 1)
    doc = ezdxf.new()
    for start, end in (
        ((30, 25), (370, 25)),
        ((370, 25), (370, 270)),
        ((370, 270), (30, 270)),
        ((30, 270), (30, 25)),
    ):
        doc.modelspace().add_line(start, end)
    doc.modelspace().add_arc((200, -900), 1000, 85, 95)
    masks = detail_geometry(
        doc.modelspace(),
        np.array([[1.0, 0, 0.5], [0, 1, 0.5], [0, 0, 1]]),
        drawing.shape,
    )

    assert not masks.mask[100, 200]
    assert find_unmatched(drawing.astype(bool), output.astype(bool), masks) == []


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
