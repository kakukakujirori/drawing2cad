"""STEP loading and the third-angle frame convention.

The frames decide which model axis becomes which screen axis in every view, and
with what sign; a silent change there would still produce a plausible-looking
drawing.  The probe box has three distinct edge lengths *and* sits off-centre in
Z, so both the axis mapping and its sign are observable -- a Z-symmetric box
would make top's screen +Y == -Z indistinguishable from +Z.
"""

import cadquery as cq
import pytest

from zeroshot.pipeline.verification.render.project import (
    bbox_diagonal,
    load_shape,
    project_views,
)

BOX_X, BOX_Y, BOX_Z = 30.0, 20.0, 10.0
BOX_Z_CENTER = 6.0  # model z spans [1, 11]: no sign ambiguity


@pytest.fixture(scope="module")
def box_step(tmp_path_factory):
    path = tmp_path_factory.mktemp("step") / "box.step"
    solid = cq.Workplane("XY").box(BOX_X, BOX_Y, BOX_Z).translate((0, 0, BOX_Z_CENTER))
    cq.exporters.export(solid, str(path))
    return path


@pytest.fixture(scope="module")
def box_views(box_step):
    return project_views(load_shape(box_step))


def _extents(projection):
    x0, y0, x1, y1 = projection.bbox(include_hidden=True)
    return x1 - x0, y1 - y0


def test_load_shape_rejects_a_non_step_file(tmp_path):
    junk = tmp_path / "not.step"
    junk.write_text("this is not a STEP file")
    with pytest.raises(RuntimeError, match="STEP read failed"):
        load_shape(junk)


def test_load_shape_rejects_a_missing_file(tmp_path):
    with pytest.raises(RuntimeError, match="STEP read failed"):
        load_shape(tmp_path / "absent.step")


def test_bbox_diagonal_matches_the_box(box_step):
    expected = (BOX_X**2 + BOX_Y**2 + BOX_Z**2) ** 0.5
    assert bbox_diagonal(load_shape(box_step)) == pytest.approx(expected, rel=1e-6)


def test_all_three_views_are_projected(box_views):
    for name in ("front", "top", "right"):
        projection = getattr(box_views, name)
        assert projection.visible.count() > 0, f"{name} has no visible edges"


def _model_range(size, center=0.0):
    return center - size / 2, center + size / 2


X_LO, X_HI = _model_range(BOX_X)
Y_LO, Y_HI = _model_range(BOX_Y)
Z_LO, Z_HI = _model_range(BOX_Z, BOX_Z_CENTER)


def test_front_maps_to_screen_x_y(box_views):
    """front: (x, y, z) -> screen (x, y)."""
    assert box_views.front.bbox() == pytest.approx((X_LO, Y_LO, X_HI, Y_HI))


def test_top_maps_to_screen_x_minus_z(box_views):
    """top: (x, y, z) -> screen (x, -z).

    Screen +Y is -Z, not +Z: moving up the top view means moving away from the
    front view's viewer, and +Z faces that viewer.
    """
    assert box_views.top.bbox() == pytest.approx((X_LO, -Z_HI, X_HI, -Z_LO))


def test_right_maps_to_screen_minus_z_y(box_views):
    """right: (x, y, z) -> screen (-z, y)."""
    assert box_views.right.bbox() == pytest.approx((-Z_HI, Y_LO, -Z_LO, Y_HI))


def test_eye_direction_is_toward_the_viewer_not_along_the_gaze(tmp_path):
    """FRONT's first vector is +Z, and the eye really does sit on the +Z side.

    A blind hole opening on the +Z face projects as a *visible* circle in the
    front view; if the frame meant the gaze direction instead, the eye would be
    at -Z and the same circle would come back hidden.
    """
    path = tmp_path / "blind_hole.step"
    solid = (
        cq.Workplane("XY")
        .box(BOX_X, BOX_Y, BOX_Z)
        .faces(">Z")
        .workplane()
        .hole(6.0, BOX_Z / 2)
    )
    cq.exporters.export(solid, str(path))
    front = project_views(load_shape(path)).front
    assert len(front.visible.circles) == 1
    assert len(front.hidden.circles) == 0


def test_top_and_right_agree_on_depth(box_views):
    """Both show the model's Z extent; arrange() relies on that to pick a scale."""
    _, top_depth = _extents(box_views.top)
    right_depth, _ = _extents(box_views.right)
    assert top_depth == pytest.approx(right_depth, rel=1e-6)


def test_projection_is_reproducible(box_step):
    """Same input, same primitives -- the golden comparisons depend on this."""
    first = project_views(load_shape(box_step))
    second = project_views(load_shape(box_step))
    for name in ("front", "top", "right"):
        a = getattr(first, name)
        b = getattr(second, name)
        assert a.bbox() == b.bbox()
        assert a.visible.count() == b.visible.count()
        assert a.hidden.count() == b.hidden.count()
