"""STEP loading, and which edges each orthographic view draws where.

The probe box has three distinct edge lengths and sits off-centre on all axes,
so both the axis mapping and its sign are observable.
"""

import math

import cadquery as cq
import pytest

from zeroshot.pipeline.stages.interpretation.contracts import (
    TOWARD_VIEWER,
    View,
    cross_axis,
)
from zeroshot.pipeline.verification.render import orthographic
from zeroshot.pipeline.verification.render.orthographic import (
    STANDARD_VIEW_FRAMES,
    load_shape,
    project,
)

BOX_X, BOX_Y, BOX_Z = 30.0, 20.0, 10.0
BOX_CENTER = (4.0, -3.0, 6.0)


def _box() -> cq.Workplane:
    return cq.Workplane("XY").box(BOX_X, BOX_Y, BOX_Z)


@pytest.fixture(scope="module")
def box():
    return _box().translate(BOX_CENTER).findSolid()


def _extents(edges: list[cq.Edge]) -> tuple[float, float, float, float]:
    bounds = cq.Compound.makeCompound(edges).BoundingBox()
    return bounds.xmin, bounds.ymin, bounds.xmax, bounds.ymax


def test_load_shape_reads_every_solid_in_the_file(tmp_path):
    path = tmp_path / "two.step"
    two = [cq.Solid.makeBox(1, 1, 1), cq.Solid.makeBox(1, 1, 1, cq.Vector(5, 0, 0))]
    cq.exporters.export(cq.Compound.makeCompound(two), str(path))

    assert len(load_shape(path).Solids()) == 2


def test_load_shape_rejects_a_non_step_file(tmp_path):
    junk = tmp_path / "not.step"
    junk.write_text("this is not a STEP file")
    with pytest.raises(ValueError, match="could not be loaded"):
        load_shape(junk)


def test_load_shape_rejects_a_missing_file(tmp_path):
    with pytest.raises(ValueError, match="could not be loaded"):
        load_shape(tmp_path / "absent.step")


def _model_range(size, center):
    return center - size / 2, center + size / 2


X_LO, X_HI = _model_range(BOX_X, BOX_CENTER[0])
Y_LO, Y_HI = _model_range(BOX_Y, BOX_CENTER[1])
Z_LO, Z_HI = _model_range(BOX_Z, BOX_CENTER[2])


@pytest.mark.parametrize(
    ("frame", "expected"),
    [
        (STANDARD_VIEW_FRAMES[View.FRONT], (X_LO, Z_LO, X_HI, Z_HI)),
        (STANDARD_VIEW_FRAMES[View.BACK], (-X_HI, Z_LO, -X_LO, Z_HI)),
        (STANDARD_VIEW_FRAMES[View.TOP], (X_LO, Y_LO, X_HI, Y_HI)),
        (STANDARD_VIEW_FRAMES[View.BOTTOM], (X_LO, -Y_HI, X_HI, -Y_LO)),
        (STANDARD_VIEW_FRAMES[View.RIGHT], (Y_LO, Z_LO, Y_HI, Z_HI)),
        (STANDARD_VIEW_FRAMES[View.LEFT], (-Y_HI, Z_LO, -Y_LO, Z_HI)),
        # A right view beside the top view shares its +Y, so +U is -Z.
        (("-z", "+y"), (-Z_HI, Y_LO, -Z_LO, Y_HI)),
    ],
    ids=["front", "back", "top", "bottom", "right", "left", "right_turned"],
)
def test_a_view_lands_where_its_frame_says(box, frame, expected):
    """Model coordinates, not a sheet: a view may sit left of the origin."""
    visible, hidden = project(box, *frame)

    assert _extents(visible + hidden) == pytest.approx(expected, abs=1e-3)


def test_each_standard_view_faces_the_viewer_its_role_names():
    assert set(STANDARD_VIEW_FRAMES) == set(TOWARD_VIEWER)
    for view, axes in STANDARD_VIEW_FRAMES.items():
        assert cross_axis(*axes) == TOWARD_VIEWER[view], view


def test_a_frame_needs_two_axes_that_span_a_plane(box):
    with pytest.raises(ValueError, match="span no plane"):
        project(box, "+x", "-x")


def test_the_eye_is_on_the_side_the_view_faces():
    """FRONT looks from -Y, so a hole opening on the -Y face is visible."""
    part = _box().faces("<Y").workplane().hole(6.0, BOX_Y / 2).findSolid()

    visible, hidden = project(part, *STANDARD_VIEW_FRAMES[View.FRONT])

    assert "CIRCLE" in {edge.geomType() for edge in visible}
    assert "CIRCLE" not in {edge.geomType() for edge in hidden}


def test_an_edge_behind_a_visible_one_is_not_drawn_hidden(box):
    _, hidden = project(box, *STANDARD_VIEW_FRAMES[View.FRONT])

    assert hidden == []


def test_a_bore_inside_the_part_is_drawn_hidden():
    part = _box().faces("<Y").workplane().hole(4.0).findSolid()

    _, hidden = project(part, *STANDARD_VIEW_FRAMES[View.TOP])

    assert hidden
    assert {edge.geomType() for edge in hidden} == {"LINE"}


# The rim fillet outgrows the corner radius, so each corner is a patch that the
# STEP reader marks as meeting its neighbours sharply.
PLATE_TOP, RIM = 11.5, 5.79


@pytest.fixture(scope="module")
def plate(tmp_path_factory):
    solid = (
        cq.Workplane()
        .rect(100, 38)
        .extrude(PLATE_TOP)
        .edges("|Z")
        .fillet(5.4)
        .edges(">Z")
        .fillet(RIM)
    )
    path = tmp_path_factory.mktemp("plate") / "plate.step"
    cq.exporters.export(solid, str(path))
    return load_shape(path)


def _on_plate_front_outline(point: cq.Vector) -> bool:
    """On the base, the top, an end, or a rim-fillet arc."""
    u, v = abs(point.x), point.y
    arc = math.hypot(u - (50 - RIM), v - (PLATE_TOP - RIM)) - RIM
    return min(abs(v), abs(v - PLATE_TOP), abs(u - 50), abs(arc)) < 0.05


def test_a_seam_between_tangent_faces_is_not_drawn(plate):
    visible, _ = project(plate, *STANDARD_VIEW_FRAMES[View.FRONT])

    points = [edge.positionAt(t) for edge in visible for t in (0.1, 0.5, 0.9)]
    assert all(_on_plate_front_outline(p) for p in points)


def test_a_seam_along_the_outline_is_still_drawn(plate):
    """From above, the rim fillet's foot is the outline over the base's edge."""
    _, hidden = project(plate, *STANDARD_VIEW_FRAMES[View.TOP])

    assert hidden == []


def test_a_straight_spline_is_drawn_as_a_line():
    spline = cq.Edge.makeSpline(
        [cq.Vector(0, 0, 0), cq.Vector(3, 1, 0), cq.Vector(9, 3, 0)]
    )

    line = orthographic._simplest(spline, 1e-6)

    assert line.geomType() == "LINE"
    assert line.startPoint() == spline.startPoint()
    assert line.endPoint() == spline.endPoint()


@pytest.mark.parametrize(
    "through",
    [((0, 0, 0), (1, 1, 0), (2, 0, 0)), ((2, 0, 0), (1, 1, 0), (0, 0, 0))],
    ids=["counter_clockwise", "clockwise"],
)
def test_a_circular_spline_is_drawn_as_the_arc_it_runs_along(through):
    spline = cq.Edge.makeThreePointArc(*through).toSplines()

    arc = orthographic._simplest(spline, 1e-2)

    assert arc.geomType() == "CIRCLE"
    for t in (0.0, 0.5, 1.0):
        assert (arc.positionAt(t) - spline.positionAt(t)).Length < 1e-2


def test_a_closed_circular_spline_is_drawn_as_a_circle():
    turns = [2 * math.pi * i / 16 for i in range(16)]
    on_circle = [cq.Vector(3 + 2 * math.cos(t), 4 + 2 * math.sin(t), 0) for t in turns]
    spline = cq.Edge.makeSpline(on_circle, periodic=True)

    circle = orthographic._simplest(spline, 1e-2)

    assert circle.geomType() == "CIRCLE"
    assert circle.IsClosed()
    assert circle.radius() == pytest.approx(2.0, abs=1e-2)


def test_a_curved_spline_stays_a_spline():
    points = [
        cq.Vector(0, 0, 0),
        cq.Vector(1, 2, 0),
        cq.Vector(3, -1, 0),
        cq.Vector(4, 0, 0),
    ]
    spline = cq.Edge.makeSpline(points)

    assert orthographic._simplest(spline, 1e-3).geomType() == "BSPLINE"


def test_an_edge_that_leaves_the_parts_footprint_is_not_drawn():
    """HLR can trace an outline far off the part; that outline is dropped."""
    footprint = ((0.0, 10.0), (0.0, 5.0))
    inside = cq.Edge.makeLine(cq.Vector(1, 1, 0), cq.Vector(9, 4, 0))
    runaway = cq.Edge.makeLine(cq.Vector(1, 1, 0), cq.Vector(90, 4, 0))

    assert orthographic._drawable(inside, footprint, 1e-3)
    assert not orthographic._drawable(runaway, footprint, 1e-3)


def test_a_spline_stays_when_the_circle_that_fits_it_runs_elsewhere(monkeypatch):
    """HLR returns out-and-back slivers, and a huge circle can fit one within tol."""
    sliver = cq.Edge.makeSpline(
        [cq.Vector(0, 0, 0), cq.Vector(1, 1, 0), cq.Vector(2, 0, 0)]
    )
    monkeypatch.setattr(
        orthographic, "_as_line_or_circle", lambda _edge, _tol: cq.Edge.makeCircle(500)
    )

    assert orthographic._simplest(sliver, 0.3) is sliver
