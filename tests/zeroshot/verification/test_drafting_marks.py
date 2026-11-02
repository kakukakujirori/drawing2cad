import math

import cadquery as cq
import cv2
import ezdxf
import numpy as np
import pytest

from zeroshot.pipeline.stages.interpretation.contracts import View
from zeroshot.pipeline.verification.drawing_diff.drafting_marks import (
    CENTRE_LINES,
    CentreLine,
    centre_lines,
    drafting_marks,
    record_centre_lines,
)
from zeroshot.pipeline.verification.render.constants import ProjectionPaths
from zeroshot.pipeline.verification.render.export_dxf import (
    DEFAULT_PNG_MARGIN_RATIO,
    png_bounds,
    write_view_dxf,
)
from zeroshot.pipeline.verification.render.orthographic import project
from zeroshot.pipeline.verification.run_render import _render_projections

HOLE_X, HOLE_Y, HOLE_R = 4.0, -2.0, 3.0


def _box() -> cq.Workplane:
    return cq.Workplane("XY").box(30, 20, 10).faces(">Z").workplane()


@pytest.fixture(scope="module")
def holed_box():
    return _box().center(HOLE_X, HOLE_Y).hole(2 * HOLE_R).val()


def _spans(lines):
    return sorted(
        (tuple(sorted(np.round([line.start, line.end], 4).tolist())), line.radius)
        for line in lines
    )


def test_a_hole_gets_a_centre_mark_end_on_and_a_centre_line_from_the_side(holed_box):
    arm = 2 * HOLE_R

    assert _spans(centre_lines(holed_box, "+x", "+y")) == _spans(
        [
            CentreLine((HOLE_X - arm, HOLE_Y), (HOLE_X + arm, HOLE_Y), HOLE_R),
            CentreLine((HOLE_X, HOLE_Y - arm), (HOLE_X, HOLE_Y + arm), HOLE_R),
        ]
    )
    # The bore runs through z = -5..5 and the line overruns it by one radius.
    assert _spans(centre_lines(holed_box, "+x", "+z")) == _spans(
        [CentreLine((HOLE_X, -5 - HOLE_R), (HOLE_X, 5 + HOLE_R), HOLE_R)]
    )


def test_a_counterbore_and_its_bore_share_one_line_with_the_bore_radius():
    part = _box().cboreHole(4.0, 8.0, 3.0).val()

    side = centre_lines(part, "+x", "+z")
    end_on = centre_lines(part, "+x", "+y")

    # Bore z = -5..2 overrun by 2, counterbore z = 2..5 overrun by 4.
    assert _spans(side) == _spans([CentreLine((0.0, -7.0), (0.0, 9.0), 2.0)])
    assert len(end_on) == 2
    assert all(line.radius == pytest.approx(2.0) for line in end_on)


def test_a_cone_is_marked_with_its_largest_radius():
    cone = cq.Solid.makeCone(0.0, 5.0, 10.0)

    (line,) = centre_lines(cone, "+x", "+z")

    assert line.radius == pytest.approx(5.0)
    assert sorted([line.start[1], line.end[1]]) == pytest.approx([-5.0, 15.0])


def test_fillets_turn_too_little_to_mark_an_axis():
    rounded = cq.Workplane("XY").box(30, 20, 10).edges("|Z").fillet(3).val()

    assert centre_lines(rounded, "+x", "+y") == []


def test_recorded_centre_lines_are_neither_entities_nor_extent(holed_box, tmp_path):
    path = tmp_path / "top.dxf"
    write_view_dxf(path, *project(holed_box, "+x", "+y")[:2], "top")
    before = ezdxf.readfile(path).modelspace()
    entities = sorted(entity.dxftype() for entity in before)
    bounds = png_bounds(before, margin_ratio=DEFAULT_PNG_MARGIN_RATIO)

    record_centre_lines(path, holed_box, "+x", "+y")

    after = ezdxf.readfile(path)
    assert sorted(entity.dxftype() for entity in after.modelspace()) == entities
    assert png_bounds(
        after.modelspace(), margin_ratio=DEFAULT_PNG_MARGIN_RATIO
    ) == pytest.approx(bounds)
    assert len(after.rootdict.get(CENTRE_LINES).tags) == 6


def test_rendered_projections_carry_their_centre_lines(holed_box, tmp_path):
    step = tmp_path / "part.step"
    cq.exporters.export(holed_box, str(step))
    paths = ProjectionPaths(top=tmp_path / "top.dxf", front=tmp_path / "front.dxf")

    _render_projections(step, paths, {View.TOP: ("+x", "+y"), View.FRONT: ("+x", "+z")})

    for path in (paths.top, paths.front):
        assert ezdxf.readfile(path).rootdict.get(CENTRE_LINES) is not None


SIZE = 220
CENTRE, RADIUS = (100, 100), 20
SQUARE = (
    ((40, 40), (160, 40)),
    ((160, 40), (160, 160)),
    ((160, 160), (40, 160)),
    ((40, 160), (40, 40)),
)
# Model U and V to input pixel boundaries: as given, and as in a comparison,
# where V runs up while pixel rows run down, at another scale.
MAPPINGS = {
    "identity": np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]),
    "flipped": np.array([[2.0, 0.0, 10.5], [0.0, -2.0, 210.5], [0.0, 0.0, 1.0]]),
}


def _model(mapping, pixel):
    x, y, w = np.linalg.solve(mapping, [pixel[0] + 0.5, pixel[1] + 0.5, 1.0])
    return x / w, y / w


def _scale(mapping):
    return math.sqrt(abs(np.linalg.det(mapping[:2, :2])))


def _drawing(mapping, *, circle=False, arc=None, lines=()):
    """A DXF of the build's circle or arc and of recorded lines, given in pixels."""
    doc = ezdxf.new()
    radius = RADIUS / _scale(mapping)
    if circle:
        doc.modelspace().add_circle(_model(mapping, CENTRE), radius)
    if arc is not None:
        start, end = arc
        if np.linalg.det(mapping[:2, :2]) < 0:
            start, end = -end, -start
        doc.modelspace().add_arc(
            _model(mapping, CENTRE), radius, start % 360, end % 360
        )
    tags = []
    for start, end, line_radius in lines:
        tags += [
            (10, (*_model(mapping, start), 0.0)),
            (11, (*_model(mapping, end), 0.0)),
            (40, line_radius / _scale(mapping)),
        ]
    if tags:
        doc.rootdict.add_xrecord(CENTRE_LINES).reset(tags)
    return doc.modelspace()


def _ink(*segments, circle=False, arc=None):
    image = np.zeros((SIZE, SIZE), np.uint8)
    for start, end in segments:
        cv2.line(image, start, end, 1, 2)
    if circle:
        cv2.circle(image, CENTRE, RADIUS, 1, 2)
    if arc is not None:
        cv2.ellipse(image, CENTRE, (RADIUS, RADIUS), 0, *arc, 1, 2)
    return image.astype(bool)


def _ray(degrees, start, end):
    """Points at start and end radii from the centre along one direction."""
    d = np.array([math.cos(math.radians(degrees)), math.sin(math.radians(degrees))])
    return tuple(np.rint(CENTRE + start * d).astype(int)), tuple(
        np.rint(CENTRE + end * d).astype(int)
    )


def _outside_square(ink):
    beyond = np.ones_like(ink)
    beyond[38:163, 38:163] = False
    return ink & beyond


def _marks(modelspace, mapping, ink, output):
    return drafting_marks(modelspace, mapping, ink, _outside_square(ink), output)


def _at(marks, point):
    return marks[point[1], point[0]]


@pytest.mark.parametrize("mapping", MAPPINGS.values(), ids=MAPPINGS)
def test_centre_marks_and_a_leader_to_their_hole_are_marks_but_part_lines_are_not(
    mapping,
):
    cross = [((70, 100), (130, 100)), ((100, 70), (100, 130))]
    leader = _ray(-30, RADIUS, 92)
    shoulder = (leader[1], (leader[1][0] + 18, leader[1][1]))
    missing_edge = ((50, 140), (150, 140))
    stub = _ray(45, RADIUS, 55)  # radial, but it ends inside the part
    ink = _ink(*SQUARE, *cross, leader, shoulder, missing_edge, stub, circle=True)
    lines = [(*arm, RADIUS) for arm in cross]

    marks = _marks(
        _drawing(mapping, circle=True, lines=lines),
        mapping,
        ink,
        _ink(*SQUARE, circle=True),
    )

    assert _at(marks, (88, 100)) and _at(marks, (112, 100)) and _at(marks, (100, 88))
    assert _at(marks, _ray(-30, RADIUS, 40)[1])
    assert not _at(marks, (100, 140))
    assert not _at(marks, _ray(45, RADIUS, 45)[1])


@pytest.mark.parametrize("mapping", MAPPINGS.values(), ids=MAPPINGS)
def test_a_sectors_radial_edge_is_not_a_leader_to_its_arc(mapping):
    edge = _ray(0, RADIUS, 75)  # from the arc's end, out of the part
    leader = _ray(60, RADIUS, 92)
    shoulder = (leader[1], (leader[1][0] + 18, leader[1][1]))
    ink = _ink(*SQUARE, edge, leader, shoulder, arc=(0, 90))

    marks = _marks(
        _drawing(mapping, arc=(0, 90)), mapping, ink, _ink(*SQUARE, arc=(0, 90))
    )

    assert _at(marks, _ray(60, RADIUS, 40)[1])
    assert not _at(marks, (140, 100))


def test_a_leader_drawn_from_the_centre_is_a_mark():
    """The run holds the centre itself, so its direction must come from the run."""
    mapping = MAPPINGS["identity"]
    radius = 30
    leader = (CENTRE, _ray(-30, 0, 92)[1])
    shoulder = (leader[1], (leader[1][0] + 18, leader[1][1]))
    ink = _ink(*SQUARE, leader, shoulder)
    cv2.circle(ink.view(np.uint8), CENTRE, radius, 1, 2)
    output = _ink(*SQUARE)
    cv2.circle(output.view(np.uint8), CENTRE, radius, 1, 2)
    doc = ezdxf.new()
    doc.modelspace().add_circle(_model(mapping, CENTRE), radius)

    marks = _marks(doc.modelspace(), mapping, ink, output)

    assert _at(marks, _ray(-30, 0, 15)[1]) and _at(marks, _ray(-30, 0, 50)[1])


def test_lines_that_cross_outrun_or_flank_a_centre_line_are_not_marks():
    mapping = MAPPINGS["identity"]
    outrunning = ((50, 72), (150, 72))
    shallow = ((70, 92), (130, 108))
    crossing = ((100, 112), (100, 155))
    centre_line = ((65, 130), (135, 130))
    flank = ((65, 58), (135, 58))  # a small bore's side, beside its axis
    lines = [
        ((60, 70), (120, 70), 40),
        ((60, 100), (140, 100), 40),
        ((60, 130), (140, 130), 40),
        ((60, 54), (140, 54), 4),
    ]
    ink = _ink(*SQUARE, outrunning, shallow, crossing, centre_line, flank)

    marks = drafting_marks(
        _drawing(mapping, lines=lines),
        mapping,
        ink,
        np.zeros_like(ink),
        _ink(*SQUARE),
    )

    assert _at(marks, (80, 130))
    assert not _at(marks, (100, 72))
    assert not _at(marks, (100, 100))
    assert not _at(marks, (100, 122)) and not _at(marks, (100, 145))
    assert not _at(marks, (100, 58))


def test_without_circles_or_recorded_axes_nothing_is_a_mark():
    mapping = MAPPINGS["identity"]
    ink = _ink(*SQUARE, ((50, 140), (150, 140)), _ray(-30, 0, 92))

    marks = _marks(_drawing(mapping), mapping, ink, _ink(*SQUARE))

    assert not marks.any()
