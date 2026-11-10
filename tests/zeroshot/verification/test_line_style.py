"""Style errors must survive positional agreement without flagging dash phase."""

import cv2
import ezdxf
import numpy as np
import pytest

from zeroshot.pipeline.verification.drawing_diff.align import AlignmentResult
from zeroshot.pipeline.verification.drawing_diff.diff import compute_diff
from zeroshot.pipeline.verification.drawing_diff.line_style import (
    find_line_style_mismatches,
    input_hidden_curves,
)
from zeroshot.pipeline.verification.drawing_diff.unmatched import (
    COLORS,
    INPUT_GREEN,
    OUTPUT_MAGENTA,
)


def _rgb(ink):
    return np.repeat(np.where(ink, 0, 255).astype(np.uint8)[..., None], 3, axis=2)


def _circle(style, *, phase=0, radius=40):
    ink = np.zeros((200, 200), np.uint8)
    if style == "visible":
        cv2.circle(ink, (100, 100), radius, 1, 2)
    else:
        for angle in range(phase, 360 + phase, 24):
            cv2.ellipse(ink, (100, 100), (radius, radius), 0, angle, angle + 13, 1, 1)
    return ink.astype(bool)


def _model(style, *, radius=40):
    doc = ezdxf.new(setup=True)
    doc.modelspace().add_circle(
        (100.5, 100.5),
        radius,
        dxfattribs={
            "linetype": "HIDDEN" if style == "hidden" else "Continuous",
        },
    )
    return doc.modelspace()


def test_input_hidden_curve_spans_separate_dashes():
    ink = np.zeros((200, 240), bool)
    for x in range(20, 215, 14):
        ink[100, x : x + 9] = True

    curves = input_hidden_curves(_rgb(ink), ink)

    assert curves
    assert all(style == "hidden" for _, style in curves)
    assert max(np.linalg.norm(xy[-1] - xy[0]) for xy, _ in curves) > 150
    assert all(np.abs(xy[:, 1] - 100).max() <= 1 for xy, _ in curves)


@pytest.mark.parametrize("kind", ["solid", "centre", "short", "empty"])
def test_input_hidden_curves_require_a_long_repeated_dash_pattern(kind):
    ink = np.zeros((200, 240), bool)
    if kind == "solid":
        ink[100, 20:220] = True
    elif kind == "centre":
        for x in range(20, 200, 30):
            ink[100, x : x + 15] = True
            ink[100, x + 21 : x + 24] = True
    elif kind == "short":
        for x in range(20, 60, 14):
            ink[100, x : x + 9] = True

    assert input_hidden_curves(_rgb(ink), ink) == []


def test_local_hidden_evidence_does_not_label_an_entire_solid_continuation():
    ink = np.zeros((200, 400), bool)
    for x in range(20, 160, 14):
        ink[100, x : x + 9] = True
    ink[100, 160:380] = True

    curves = input_hidden_curves(_rgb(ink), ink)

    assert curves
    assert max(xy[:, 0].max() for xy, _ in curves) < 220


def test_input_hidden_missing_reaches_feedback_without_changing_scores(monkeypatch):
    drawing = np.zeros((240, 300), np.uint8)
    cv2.rectangle(drawing, (10, 10), (290, 230), 1)
    output = drawing.copy()
    for x in range(50, 240, 14):
        drawing[120, x : x + 9] = 1
    output[120, 50:101] = 1
    output[120, 150:241] = 1
    alignment = AlignmentResult(
        "directional_chamfer", "similarity", "ok", np.eye(3).tolist(), {}
    )
    result = compute_diff(_rgb(drawing), _rgb(output), alignment)
    monkeypatch.setattr(
        "zeroshot.pipeline.verification.drawing_diff.diff.input_hidden_curves",
        lambda *_: [],
    )
    before = compute_diff(_rgb(drawing), _rgb(output), alignment)

    assert before.stats["unmatched"] == []
    assert len(result.stats["unmatched"]) == 1
    assert result.stats["unmatched"][0]["direction"] == "missing"
    assert result.stats["unmatched"][0]["size_px"] > 30
    assert result.stats["match_score"] == before.stats["match_score"]


@pytest.mark.parametrize("optional", [False, True])
@pytest.mark.parametrize("offset", [0, 8])
def test_hidden_cad_curve_supports_input_across_long_rendered_gaps(optional, offset):
    drawing = np.zeros((300, 300), np.uint8)
    cv2.rectangle(drawing, (20, 20), (280, 280), 1)
    projection = drawing.copy()
    tangent = np.zeros_like(drawing)
    for x in range(50, 241, 14):
        drawing[150, x : min(x + 9, 241)] = 1
    rendered = tangent if optional else projection
    for x in range(50, 241, 50):
        rendered[150 + offset, x : min(x + 20, 241)] = 1

    doc = ezdxf.new(setup=True)
    doc.layers.new("TANGENT")
    model = doc.modelspace()
    for start, end in [
        ((20, 20), (280, 20)),
        ((280, 20), (280, 280)),
        ((280, 280), (20, 280)),
        ((20, 280), (20, 20)),
    ]:
        model.add_line(np.array(start) + 0.5, np.array(end) + 0.5)
    model.add_line(
        (50.5, 150.5 + offset),
        (240.5, 150.5 + offset),
        dxfattribs={"linetype": "HIDDEN", "layer": "TANGENT" if optional else "0"},
    )
    alignment = AlignmentResult(
        "directional_chamfer", "similarity", "ok", np.eye(3).tolist(), {}
    )
    result = compute_diff(
        _rgb(drawing),
        _rgb(projection),
        alignment,
        projection_modelspace=model,
        projection_to_uv=np.eye(3),
        tangent_ink=tangent.astype(bool),
        hide_tangents=True,
    )

    assert result.warnings == ()
    groups = result.stats["unmatched"]
    assert sum(g["direction"] == "missing" for g in groups) == bool(offset)
    if not offset:
        assert groups == []
    if optional:
        assert all(g["direction"] == "missing" for g in groups)


@pytest.mark.parametrize(
    "input_style,output_style", [("visible", "hidden"), ("hidden", "visible")]
)
def test_same_position_different_style_is_one_error(input_style, output_style):
    groups, stats = find_line_style_mismatches(
        _model(output_style),
        np.eye(3),
        _rgb(_circle(input_style)),
        _circle(input_style),
    )
    assert stats["mismatch_count"] == 1
    assert stats["checked_curves"] == 1
    assert stats["mismatch_length_px"] == pytest.approx(2 * np.pi * 40, rel=0.01)
    assert stats["checked_length_px"] == stats["mismatch_length_px"]
    assert stats["unknown_length_px"] == 0
    assert len(groups) == 1
    assert (groups[0].input_style, groups[0].output_style) == (
        input_style,
        output_style,
    )


@pytest.mark.parametrize(
    "style,phase", [("visible", 0), ("hidden", 0), ("hidden", 7), ("hidden", 15)]
)
def test_matching_styles_are_not_errors(style, phase):
    groups, stats = find_line_style_mismatches(
        _model(style),
        np.eye(3),
        _rgb(_circle(style, phase=phase)),
        _circle(style, phase=phase),
    )
    assert groups == []
    assert stats["checked_curves"] == 1
    assert stats["mismatch_length_px"] == 0
    assert stats["checked_length_px"] == pytest.approx(2 * np.pi * 40, rel=0.01)


@pytest.mark.parametrize(
    "case", ["broken", "cropped", "missing", "short", "tangent", "overlap", "nearby"]
)
def test_ambiguous_or_optional_curves_are_not_errors(case):
    model = _model("hidden")
    ink = _circle("visible")
    if case == "broken":
        ink[:90] = False  # a single missing arc is not a dash pattern
    elif case == "cropped":
        ink = ink[:115]
    elif case == "missing":
        ink[:] = False
    elif case == "short":
        model = _model("hidden", radius=2)
        ink = _circle("visible", radius=2)
    elif case == "tangent":
        model.doc.layers.new("TANGENT")
        model[0].dxf.layer = "TANGENT"
    else:
        model.add_circle((100.5, 100.5), 40 if case == "overlap" else 42)
    groups, stats = find_line_style_mismatches(model, np.eye(3), _rgb(ink), ink)
    assert not groups
    assert stats["mismatch_count"] == 0


def test_hidden_circle_is_not_confused_with_close_visible_concentric_circle():
    ink = _circle("hidden") | _circle("visible", radius=36)
    model = _model("hidden")
    model.add_circle((100.5, 100.5), 36)
    groups, stats = find_line_style_mismatches(model, np.eye(3), _rgb(ink), ink)
    assert not groups
    assert stats["checked_curves"] >= 1


@pytest.mark.parametrize("style", ["visible", "hidden"])
@pytest.mark.parametrize("separation", [3.25, 3.5, 4.0])
@pytest.mark.parametrize("flip", [False, True])
@pytest.mark.parametrize("transpose", [False, True])
def test_local_fit_cannot_take_the_style_of_a_different_parallel_curve(
    style, separation, flip, transpose
):
    gray = np.full((200, 200), 255, np.uint8)
    other_style = "visible" if style == "hidden" else "hidden"
    for y, line_style, thickness in (
        (61, other_style, 2),
        (round(60 + separation + 1.5), style, 1),
    ):
        for x in [20] if line_style == "visible" else range(20, 180, 20):
            end = 180 if line_style == "visible" else min(x + 14, 180)
            cv2.line(gray, (x, y), (end, y), 0, thickness, cv2.LINE_AA)
    curves = [
        (np.array([[20.5, 60.5], [180.5, 60.5]]), other_style),
        (np.array([[20.5, 60.5 + separation], [180.5, 60.5 + separation]]), style),
    ]
    if flip:
        gray = np.flipud(gray).copy()
        curves = [(xy * (1, -1) + (0, 200), kind) for xy, kind in curves]
    if transpose:
        gray = gray.T.copy()
        curves = [(xy[:, ::-1], kind) for xy, kind in curves]
    model = ezdxf.new(setup=True).modelspace()
    for xy, kind in curves:
        model.add_line(
            *xy,
            dxfattribs={
                "linetype": "HIDDEN" if kind == "hidden" else "Continuous",
            },
        )

    groups, _ = find_line_style_mismatches(
        model, np.eye(3), np.repeat(gray[..., None], 3, axis=2), gray < 160
    )

    assert not groups


@pytest.mark.parametrize("style", ["visible", "hidden"])
def test_coincident_curves_keep_real_style_errors_after_local_translation(style):
    model = ezdxf.new(setup=True).modelspace()
    for _ in range(2):
        model.add_line(
            (20.5, 100.9),
            (180.5, 100.9),
            dxfattribs={
                "linetype": "HIDDEN" if style == "hidden" else "Continuous",
            },
        )
    ink = np.zeros((200, 200), bool)
    if style == "hidden":
        ink[102, 20:181] = True
    else:
        for x in range(20, 180, 14):
            ink[102, x : min(x + 9, 181)] = True

    groups, stats = find_line_style_mismatches(model, np.eye(3), _rgb(ink), ink)

    assert len(groups) == 1
    assert stats["checked_curves"] == 2
    assert groups[0].input_style != style


def test_duplicate_hidden_curves_report_once_with_nonidentity_transform():
    model = _model("hidden")
    model.add_circle((100.5, 100.5), 40, dxfattribs={"linetype": "HIDDEN"})
    # CAD UV scales and flips into the input grid; a small residual is tolerated.
    transform = np.array([[1.5, 0, -49.5], [0, -1.5, 250.5], [0, 0, 1]])
    groups, stats = find_line_style_mismatches(
        model,
        transform,
        _rgb(_circle("visible", radius=60)),
        _circle("visible", radius=60),
    )
    assert stats["mismatch_count"] == 1
    assert np.allclose(groups[0].box_px, (41, 40, 162, 161), atol=1)
    assert stats["mismatch_length_px"] == pytest.approx(2 * np.pi * 60, rel=0.01)


def test_by_layer_style_and_open_dashed_line():
    model = ezdxf.new(setup=True).modelspace()
    model.doc.layers.new("occluded", dxfattribs={"linetype": "HIDDEN"})
    model.add_line((20.5, 100.5), (180.5, 100.5), dxfattribs={"layer": "occluded"})
    ink = np.zeros((200, 200), bool)
    ink[100, 20:181] = True
    groups, _ = find_line_style_mismatches(model, np.eye(3), _rgb(ink), ink)
    assert groups[0].output_style == "hidden"
    model[0].dxf.linetype = "Continuous"
    for x in range(20, 180, 14):
        ink[100, x : x + 6] = False
    groups, _ = find_line_style_mismatches(model, np.eye(3), _rgb(ink), ink)
    assert groups[0].input_style == "hidden"


@pytest.mark.parametrize(
    "input_style,output_style", [("visible", "hidden"), ("hidden", "visible")]
)
def test_shared_style_error_reaches_both_images_and_keeps_geometry_score(
    input_style, output_style
):
    drawing, output = _circle(input_style), _circle(output_style)
    # Filled outer silhouette makes these legitimate internal part lines.
    drawing = drawing.astype(np.uint8)
    output = output.astype(np.uint8)
    for ink in (drawing, output):
        cv2.rectangle(ink, (10, 10), (190, 190), 1, 2)
    rgb = lambda ink: np.repeat((255 - ink * 255)[..., None], 3, axis=2)
    alignment = AlignmentResult(
        "directional_chamfer", "similarity", "ok", np.eye(3).tolist(), {}
    )
    plain = compute_diff(rgb(drawing), rgb(output), alignment)
    diff = compute_diff(
        rgb(drawing),
        rgb(output),
        alignment,
        projection_modelspace=_model(output_style),
        projection_to_uv=np.eye(3),
    )
    assert diff.stats["comparison_status"] == "ok", diff.warnings
    assert plain.stats["line_style"]["status"] == "unavailable"
    assert diff.stats["match_score"] == plain.stats["match_score"]
    assert diff.stats["line_style"]["mismatch_count"] == 1
    assert diff.stats["line_style"]["displayed_count"] == 1
    assert len(diff.stats["unmatched"]) == 1
    group = diff.stats["unmatched"][0]
    assert (group["direction"], group["input_style"], group["output_style"]) == (
        "line_style",
        input_style,
        output_style,
    )
    circle_ink = _circle(output_style)
    assert (diff.overlay[circle_ink] == [255, 0, 0]).all()
    # The solid circle shows through the other's dash gaps in its own color.
    solid = OUTPUT_MAGENTA if output_style == "visible" else INPUT_GREEN
    assert np.any(np.all(diff.unmatched[_circle("visible")] == solid, axis=1))
    assert np.any(np.all(diff.unmatched == COLORS["dark orange"], axis=2))


def test_nearby_distinct_hidden_edges_can_bridge_gaps_and_are_unknown():
    model = ezdxf.new(setup=True).modelspace()
    for y in (99.5, 102.5):
        model.add_line((20.5, y), (180.5, y), dxfattribs={"linetype": "HIDDEN"})
    ink = np.zeros((200, 200), bool)
    ink[100, 20:181] = True
    groups, stats = find_line_style_mismatches(model, np.eye(3), _rgb(ink), ink)
    assert not groups
    assert stats["unknown_curves"] == 2


def test_repeated_long_short_centre_line_is_not_a_hidden_line():
    model = ezdxf.new().modelspace()
    model.add_line((20.5, 100.5), (180.5, 100.5))
    ink = np.zeros((200, 200), bool)
    for x in range(20, 170, 30):
        ink[100, x : x + 15] = True
        ink[100, x + 21 : x + 24] = True
    groups, stats = find_line_style_mismatches(model, np.eye(3), _rgb(ink), ink)
    assert not groups
    assert stats["unknown_curves"] == 1


def test_style_changing_partway_is_unknown_in_the_whole_curve_prototype():
    ink = _circle("visible")
    ink[:100] = _circle("hidden")[:100]
    groups, stats = find_line_style_mismatches(
        _model("visible"), np.eye(3), _rgb(ink), ink
    )
    assert not groups
    assert stats["unknown_curves"] == 1


def test_open_arc_mismatch_and_small_hole_bypass_the_far_line_minimum_size():
    model = ezdxf.new(setup=True).modelspace()
    model.add_arc((100.5, 100.5), 40, 20, 170, dxfattribs={"linetype": "HIDDEN"})
    groups, stats = find_line_style_mismatches(
        model, np.eye(3), _rgb(_circle("visible")), _circle("visible")
    )
    assert stats["mismatch_count"] == 1
    assert groups[0].input_style == "visible"
    groups, stats = find_line_style_mismatches(
        _model("hidden", radius=5),
        np.eye(3),
        _rgb(_circle("visible", radius=5)),
        _circle("visible", radius=5),
    )
    assert stats["mismatch_count"] == 1
    assert groups[0].size_px < 50


def test_annotation_fragments_near_a_curve_do_not_become_dash_evidence():
    model = ezdxf.new().modelspace()
    model.add_line((20.5, 100.5), (180.5, 100.5))
    ink = np.zeros((200, 200), bool)
    annotation = np.zeros_like(ink)
    for x in range(20, 180, 14):
        ink[100, x : x + 8] = True
        annotation[98, x : x + 8] = True
    groups, _ = find_line_style_mismatches(model, np.eye(3), _rgb(ink), ink)
    assert groups  # periodic surviving text fragments look like dashes without context
    groups, stats = find_line_style_mismatches(
        model, np.eye(3), _rgb(ink), ink, annotation
    )
    assert not groups
    assert stats["unknown_curves"] == 1


def test_overlay_and_unmatched_share_error_after_rotation_scale_and_translation():
    from zeroshot.pipeline.verification.drawing_diff.align import opencv_transform

    drawing = _circle("visible").astype(np.uint8)
    source_output = _circle("hidden").astype(np.uint8)
    for ink in (drawing, source_output):
        cv2.rectangle(ink, (10, 10), (190, 190), 1, 2)
    angle = np.deg2rad(10)
    transform = np.array(
        [
            [1.4 * np.cos(angle), -1.4 * np.sin(angle), 80],
            [1.4 * np.sin(angle), 1.4 * np.cos(angle), 30],
            [0, 0, 1],
        ]
    )
    output = cv2.warpPerspective(
        source_output, opencv_transform(transform), (400, 400), flags=cv2.INTER_NEAREST
    )
    drawing_rgb = np.repeat((255 - drawing * 255)[..., None], 3, axis=2)
    output_rgb = np.repeat((255 - output * 255)[..., None], 3, axis=2)
    alignment = AlignmentResult(
        "directional_chamfer", "similarity", "ok", transform.tolist(), {}
    )
    result = compute_diff(
        drawing_rgb,
        output_rgb,
        alignment,
        projection_modelspace=_model("hidden"),
        projection_to_uv=np.linalg.inv(transform),
    )
    assert result.stats["line_style"]["mismatch_count"] == 1
    group = next(g for g in result.stats["unmatched"] if g["direction"] == "line_style")
    assert np.allclose(group["box_px"], [60, 60, 141, 141], atol=1)
    warped_circle = cv2.warpPerspective(
        _circle("hidden").astype(np.uint8),
        opencv_transform(transform),
        (400, 400),
        flags=cv2.INTER_NEAREST,
    ).astype(bool)
    assert (result.overlay[warped_circle] == [255, 0, 0]).all()


def test_visible_overlap_wins_even_if_visible_edge_is_too_short_to_classify():
    model = ezdxf.new(setup=True).modelspace()
    model.add_line((50.5, 100.5), (75.5, 100.5), dxfattribs={"linetype": "HIDDEN"})
    model.add_line((50.5, 100.5), (73.5, 100.5))
    ink = np.zeros((200, 200), bool)
    ink[100, 50:74] = True
    groups, _ = find_line_style_mismatches(model, np.eye(3), _rgb(ink), ink)
    assert not groups


def test_optional_tangent_can_explain_a_solid_stroke_over_a_hidden_edge():
    model = _model("hidden")
    model.doc.layers.new("TANGENT")
    model.add_circle((100.5, 100.5), 40, dxfattribs={"layer": "TANGENT"})
    groups, stats = find_line_style_mismatches(
        model, np.eye(3), _rgb(_circle("visible")), _circle("visible")
    )
    assert not groups
    assert stats["unknown_curves"] == 1


@pytest.mark.parametrize("circle_first", [False, True])
def test_lengths_do_not_double_count_a_circle_also_present_as_arcs(circle_first):
    model = ezdxf.new(setup=True).modelspace()
    attrs = {"linetype": "HIDDEN"}
    if circle_first:
        model.add_circle((100.5, 100.5), 40, dxfattribs=attrs)
    model.add_arc((100.5, 100.5), 40, 0, 180, dxfattribs=attrs)
    model.add_arc((100.5, 100.5), 40, 180, 360, dxfattribs=attrs)
    if not circle_first:
        model.add_circle((100.5, 100.5), 40, dxfattribs=attrs)
    groups, stats = find_line_style_mismatches(
        model, np.eye(3), _rgb(_circle("visible")), _circle("visible")
    )
    assert len(groups) == stats["mismatch_count"] == 1
    assert stats["mismatch_length_px"] == pytest.approx(2 * np.pi * 40, rel=0.01)
    assert stats["checked_length_px"] == stats["mismatch_length_px"]


def test_unknown_lengths_use_arc_length_and_exclude_overlapping_fragments():
    model = ezdxf.new().modelspace()
    model.add_line((20.5, 20.5), (180.5, 180.5))
    model.add_line((60.5, 60.5), (120.5, 120.5))
    _, stats = find_line_style_mismatches(
        model, np.eye(3), _rgb(np.zeros((200, 200), bool)), np.zeros((200, 200), bool)
    )
    assert stats["unknown_length_px"] == pytest.approx(np.hypot(160, 160), abs=0.1)
    assert stats["checked_length_px"] == 0
    assert stats["mismatch_length_px"] == 0


def test_known_overlap_does_not_also_count_as_unknown_length():
    model = _model("hidden")
    model.add_circle((100.5, 100.5), 40)
    _, stats = find_line_style_mismatches(
        model, np.eye(3), _rgb(_circle("visible")), _circle("visible")
    )
    assert stats["checked_length_px"] == pytest.approx(2 * np.pi * 40, rel=0.01)
    assert stats["unknown_length_px"] == 0


def test_optional_tangent_adds_no_diagnostic_length():
    model = _model("visible")
    model.doc.layers.new("TANGENT")
    model[0].dxf.layer = "TANGENT"
    _, stats = find_line_style_mismatches(
        model, np.eye(3), _rgb(_circle("visible")), _circle("visible")
    )
    assert (
        stats["checked_length_px"]
        == stats["unknown_length_px"]
        == stats["mismatch_length_px"]
        == 0
    )


@pytest.mark.parametrize("gap", [1, 2])
@pytest.mark.parametrize("offset", [0, 0.5, 1.5])
@pytest.mark.parametrize("angle", [0, 30, 45, 90])
def test_narrow_dash_gaps_are_not_classified_as_solid(gap, offset, angle):
    direction = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
    normal = direction[::-1] * (-1, 1)
    start = np.array([20.0, 25.0])
    model = ezdxf.new(setup=True).modelspace()
    model.add_line(
        start + 0.5 + normal * offset,
        start + direction * 160 + 0.5 + normal * offset,
        dxfattribs={"linetype": "HIDDEN"},
    )
    ink = np.zeros((200, 200), np.uint8)
    for x in range(0, 161, 8 + gap):
        cv2.line(
            ink,
            tuple(np.rint(start + direction * x).astype(int)),
            tuple(np.rint(start + direction * min(x + 7, 160)).astype(int)),
            1,
            1,
        )
    groups, stats = find_line_style_mismatches(
        model, np.eye(3), _rgb(ink.astype(bool)), ink.astype(bool)
    )
    assert not groups
    assert stats["mismatch_length_px"] == 0


def test_overlapping_arc_and_circle_have_identical_clusters_in_either_order():
    results = []
    for circle_first in (False, True):
        model = _model("hidden")
        circle = model[0]
        model.add_arc((100.5, 100.5), 40, 0, 270, dxfattribs={"linetype": "HIDDEN"})
        if not circle_first:
            model.unlink_entity(circle)
            model.add_entity(circle)
        groups, stats = find_line_style_mismatches(
            model, np.eye(3), _rgb(_circle("visible")), _circle("visible")
        )
        assert len(groups) == stats["mismatch_count"] == 1
        results.append((set(map(tuple, groups[0].xy)), stats))
    assert results[0] == results[1]


def test_normal_probes_do_not_expand_correspondence_tolerance():
    model = ezdxf.new(setup=True).modelspace()
    model.add_line((20.5, 100.9), (180.5, 100.9), dxfattribs={"linetype": "HIDDEN"})
    ink = np.zeros((200, 200), bool)
    ink[100, 20:120] = True
    ink[101, 120:181] = True  # this tail is over 1 px from the fitted line
    groups, stats = find_line_style_mismatches(model, np.eye(3), _rgb(ink), ink)
    assert not groups
    assert stats["unknown_curves"] == 1


def test_partially_overlapping_mismatch_clusters_do_not_repeat_pixels():
    model = ezdxf.new(setup=True).modelspace()
    for start, end in ((20.5, 130.5), (70.5, 180.5)):
        model.add_line((start, 100.5), (end, 100.5), dxfattribs={"linetype": "HIDDEN"})
    ink = np.zeros((200, 200), bool)
    ink[100, 20:181] = True
    groups, stats = find_line_style_mismatches(model, np.eye(3), _rgb(ink), ink)
    assert len(groups) == 2
    first, second = [set(map(tuple, group.xy)) for group in groups]
    assert not first & second
    assert stats["mismatch_length_px"] == pytest.approx(160, rel=0.01)


@pytest.mark.parametrize("gap_count", [0, 2, 8])
@pytest.mark.parametrize("output_style", ["HIDDEN", "Continuous"])
def test_grayscale_dash_evidence_survives_binarization_in_comparison(
    gap_count, output_style
):
    gray = np.full((200, 200), 255, np.uint8)
    cv2.rectangle(gray, (10, 10), (190, 190), 36, 2)
    gray[100, 20:181] = 36
    if gap_count:
        for x in np.linspace(30, 170, gap_count).astype(int):
            gray[100, x : x + 6] = 80  # Still ink under the global Otsu threshold.
    model = ezdxf.new(setup=True).modelspace()
    model.add_line((20.5, 100.5), (180.5, 100.5), dxfattribs={"linetype": output_style})
    rgb = np.repeat(gray[..., None], 3, axis=2)
    result = compute_diff(
        rgb,
        rgb,
        AlignmentResult(
            "directional_chamfer", "similarity", "ok", np.eye(3).tolist(), {}
        ),
        projection_modelspace=model,
        projection_to_uv=np.eye(3),
    )
    stats = result.stats["line_style"]
    assert stats["mismatch_count"] == int(not gap_count and output_style == "HIDDEN")
    assert stats["unknown_curves"] == int(bool(gap_count))
    assert stats["checked_curves"] == int(not gap_count)


@pytest.mark.parametrize("gaps", [[6], [4, 15]])
def test_short_dash_gaps_are_not_raster_holes(gaps):
    model = ezdxf.new(setup=True).modelspace()
    model.add_line((20.5, 100.5), (45.5, 100.5), dxfattribs={"linetype": "HIDDEN"})
    ink = np.zeros((200, 200), bool)
    ink[100, 20:46] = True
    ink[100, np.array(gaps) + 20] = False
    groups, stats = find_line_style_mismatches(model, np.eye(3), _rgb(ink), ink)
    assert not groups
    assert stats["unknown_curves"] == 1


@pytest.mark.parametrize("output_style", ["HIDDEN", "Continuous"])
def test_hidden_stroke_over_solid_annotation_is_not_solid_evidence(output_style):
    gray = np.full((200, 200), 255, np.uint8)
    gray[100, 5:196] = 36  # A continuous dimension stroke hides the dash gaps.
    for x in range(20, 180, 20):
        gray[101, x : x + 14] = 85
    model = ezdxf.new(setup=True).modelspace()
    model.add_line((20.5, 100.5), (180.5, 100.5), dxfattribs={"linetype": output_style})
    annotation = np.zeros_like(gray, bool)
    annotation[100, :18] = annotation[100, 183:] = True
    groups, stats = find_line_style_mismatches(
        model,
        np.eye(3),
        np.repeat(gray[..., None], 3, axis=2),
        (gray < 160) & ~annotation,
        annotation,
    )
    assert not groups
    assert stats["unknown_curves"] == 1


@pytest.mark.parametrize("angle", [0, 15, 30, 45, 90])
@pytest.mark.parametrize("antialias", [False, True])
def test_raster_width_variation_in_a_solid_stroke_is_not_dash_evidence(
    angle, antialias
):
    start = np.array([30, 30])
    end = np.rint(
        start + 150 * np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
    ).astype(int)
    gray = np.full((220, 220), 255, np.uint8)
    cv2.line(gray, start, end, 36, 2, cv2.LINE_AA if antialias else cv2.LINE_8)
    model = ezdxf.new(setup=True).modelspace()
    model.add_line(start + 0.5, end + 0.5, dxfattribs={"linetype": "HIDDEN"})
    groups, stats = find_line_style_mismatches(
        model, np.eye(3), np.repeat(gray[..., None], 3, axis=2), gray < 160
    )
    assert len(groups) == 1
    assert stats["checked_curves"] == 1


def test_one_raster_hole_and_one_pale_mark_do_not_make_repeated_dashes():
    from zeroshot.pipeline.verification.drawing_diff.line_style import _style

    support = np.ones(100, bool)
    support[30:32] = False
    profiles = np.ones(100)
    profiles[28:32] = 0.6  # Antialiased rim of the tolerated raster hole.
    profiles[70:73] = 0.6
    assert _style(support, 1, True, profiles) == "visible"


@pytest.mark.parametrize("output_style", ["HIDDEN", "Continuous"])
@pytest.mark.parametrize("collinear", [False, True])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("thick", [False, True])
def test_dimension_extension_is_unknown_for_either_cad_style(
    output_style, collinear, transpose, thick
):
    ink = np.zeros((200, 200), bool)
    ink[100, 60:141] = True
    if thick:
        ink[99:102, 60:141] = True
    annotation = np.zeros_like(ink)
    if collinear:
        annotation[100, 20:58] = True
    else:
        annotation[50:99, 57] = True
    start, end = np.array([60.5, 100.5]), np.array([140.5, 100.5])
    if transpose:
        ink, annotation = ink.T, annotation.T
        start, end = start[::-1], end[::-1]
    model = ezdxf.new(setup=True).modelspace()
    model.add_line(start, end, dxfattribs={"linetype": output_style})
    gray = np.where(ink | annotation, 0, 255).astype(np.uint8)
    groups, stats = find_line_style_mismatches(
        model, np.eye(3), np.repeat(gray[..., None], 3, axis=2), ink, annotation
    )
    ambiguous = collinear and not thick
    assert stats["unknown_curves"] == int(ambiguous)
    assert len(groups) == int(not ambiguous and output_style == "HIDDEN")


def test_solid_stroke_beside_dashes_remains_a_mismatch():
    gray = np.full((200, 200), 255, np.uint8)
    cv2.line(gray, (20, 100), (180, 100), 0, 2, cv2.LINE_AA)
    for x in range(20, 180, 20):
        cv2.line(gray, (x, 97), (x + 14, 97), 0, 1, cv2.LINE_AA)
    model = ezdxf.new(setup=True).modelspace()
    model.add_line((20.5, 100.5), (180.5, 100.5), dxfattribs={"linetype": "HIDDEN"})
    groups, stats = find_line_style_mismatches(
        model, np.eye(3), np.repeat(gray[..., None], 3, axis=2), gray < 160
    )
    assert len(groups) == 1
    assert stats["checked_curves"] == 1


def test_continuous_grayscale_support_can_confirm_a_binary_raster_hole():
    from zeroshot.pipeline.verification.drawing_diff.line_style import _style

    support = np.ones(100, bool)
    support[2] = False  # Distance gating near a corner, with dark ink still present.
    assert _style(support, 1, False, np.ones(100)) == "visible"


@pytest.mark.parametrize("gap", [False, True])
@pytest.mark.parametrize("output_style", ["HIDDEN", "Continuous"])
def test_distant_annotation_requires_a_connected_continuation(gap, output_style):
    gray = np.full((200, 240), 255, np.uint8)
    gray[100, 20:221] = 36
    if gap:
        gray[100, 55:63] = 255
    annotation = np.zeros_like(gray, bool)
    annotation[100, 20:40] = True
    model = ezdxf.new(setup=True).modelspace()
    model.add_line(
        (120.5, 100.5), (220.5, 100.5), dxfattribs={"linetype": output_style}
    )
    groups, stats = find_line_style_mismatches(
        model,
        np.eye(3),
        np.repeat(gray[..., None], 3, axis=2),
        (gray < 160) & ~annotation,
        annotation,
    )
    assert stats["unknown_curves"] == int(not gap)
    assert len(groups) == int(gap and output_style == "HIDDEN")
