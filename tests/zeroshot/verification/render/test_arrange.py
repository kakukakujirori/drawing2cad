"""Placement invariants for the third-angle L-arrangement.

These run without OCC: ``arrange`` is pure geometry over the projected
primitives, so the views are built by hand.
"""

import math

import pytest

from zeroshot.pipeline.verification.render._hlr import (
    Circle,
    ProjectedEdges,
    Segment,
    ViewProjection,
)
from zeroshot.pipeline.verification.render.arrange import (
    MIN_VIEW_EXTENT_MM,
    SCALE_LADDER,
    VIEW_GAP_H_MM,
    VIEW_GAP_V_MM,
    DegenerateDrawingError,
    arrange,
    select_scale,
)
from zeroshot.pipeline.verification.render.constants import (
    ENVELOPE_H_MM,
    ENVELOPE_W_MM,
    SHEET_CENTER_MM,
    SHEET_H_MM,
    SHEET_W_MM,
)
from zeroshot.pipeline.verification.render.project import ViewProjections


def _box(width: float, height: float, x0: float = 0.0, y0: float = 0.0):
    """A rectangular outline as a visible-only projection."""
    x1, y1 = x0 + width, y0 + height
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    edges = ProjectedEdges(
        segments=[
            Segment(corners[i], corners[(i + 1) % 4]) for i in range(len(corners))
        ]
    )
    return ViewProjection(visible=edges, hidden=ProjectedEdges())


def _part(front_w=2.0, front_h=1.0, depth=0.5, offset=0.0):
    """A consistent third-angle set: top and right both show ``depth``."""
    return ViewProjections(
        front=_box(front_w, front_h, offset, offset),
        top=_box(front_w, depth, offset, offset),
        right=_box(depth, front_h, offset, offset),
    )


def _all(layout):
    return (layout.front, layout.top, layout.right)


def test_top_shares_the_front_x_extent():
    views = arrange(_part())
    assert views.top.bbox[0] == pytest.approx(views.front.bbox[0])
    assert views.top.bbox[2] == pytest.approx(views.front.bbox[2])


def test_right_shares_the_front_y_extent():
    views = arrange(_part())
    assert views.right.bbox[1] == pytest.approx(views.front.bbox[1])
    assert views.right.bbox[3] == pytest.approx(views.front.bbox[3])


def test_top_is_above_and_right_is_right_of_front():
    views = arrange(_part())
    assert views.top.bbox[1] > views.front.bbox[3]
    assert views.right.bbox[0] > views.front.bbox[2]


def test_gaps_are_the_configured_ones():
    views = arrange(_part())
    assert views.top.bbox[1] - views.front.bbox[3] == pytest.approx(VIEW_GAP_V_MM)
    assert views.right.bbox[0] - views.front.bbox[2] == pytest.approx(VIEW_GAP_H_MM)


def test_cluster_is_centred_on_the_sheet():
    views = _all(arrange(_part()))
    x0 = min(v.bbox[0] for v in views)
    x1 = max(v.bbox[2] for v in views)
    y0 = min(v.bbox[1] for v in views)
    y1 = max(v.bbox[3] for v in views)
    assert (x0 + x1) / 2 == pytest.approx(SHEET_CENTER_MM[0])
    assert (y0 + y1) / 2 == pytest.approx(SHEET_CENTER_MM[1])


def test_placement_does_not_depend_on_the_projection_origin():
    """Views are centred by their own bbox, so a shifted model lands identically."""
    centred = arrange(_part(offset=0.0))
    shifted = arrange(_part(offset=17.5))
    for a, b in zip(_all(centred), _all(shifted)):
        assert b.bbox == pytest.approx(a.bbox)


def test_all_views_stay_on_the_sheet():
    for front_w, front_h, depth in [(2, 1, 0.5), (0.01, 0.02, 0.01), (500, 400, 300)]:
        for view in _all(arrange(_part(front_w, front_h, depth))):
            assert view.bbox[0] >= 0.0
            assert view.bbox[1] >= 0.0
            assert view.bbox[2] <= SHEET_W_MM
            assert view.bbox[3] <= SHEET_H_MM


def test_every_view_is_scaled_by_the_same_factor():
    """The three views must share one drawing scale, or the drawing lies about
    the part's proportions.  Each view's factor is recovered from its own bbox."""
    front_w, front_h, depth = 2.0, 1.0, 0.5
    views = arrange(_part(front_w, front_h, depth))
    factors = [
        (views.front.bbox[2] - views.front.bbox[0]) / front_w,
        (views.front.bbox[3] - views.front.bbox[1]) / front_h,
        (views.top.bbox[2] - views.top.bbox[0]) / front_w,
        (views.top.bbox[3] - views.top.bbox[1]) / depth,
        (views.right.bbox[2] - views.right.bbox[0]) / depth,
        (views.right.bbox[3] - views.right.bbox[1]) / front_h,
    ]
    assert factors == pytest.approx([factors[0]] * len(factors))


def test_circles_scale_with_the_drawing():
    """A circle's radius must follow the same factor as the outline it sits in,
    otherwise holes come out the wrong size relative to the part."""
    front_w = 2.0
    projections = _part(front_w=front_w, front_h=1.0)
    projections.front.visible.circles.append(Circle((1.0, 0.5), 0.25))
    front = arrange(projections).front
    factor = (front.bbox[2] - front.bbox[0]) / front_w
    (circle,) = front.visible.circles
    assert circle.radius == pytest.approx(0.25 * factor)


def test_hidden_edges_are_placed_with_the_visible_ones():
    projections = _part()
    projections.front.hidden.segments.append(Segment((0.0, 0.0), (2.0, 1.0)))
    front = arrange(projections).front
    (hidden,) = front.hidden.segments
    assert front.bbox[0] <= hidden.p0[0] <= front.bbox[2]
    assert front.bbox[1] <= hidden.p1[1] <= front.bbox[3]


def test_select_scale_takes_the_largest_ladder_value_that_fits():
    scale = select_scale(front_w=2.0, front_h=1.0, depth=0.5)
    assert scale in SCALE_LADDER
    assert scale * (2.0 + 0.5) + VIEW_GAP_H_MM <= ENVELOPE_W_MM
    assert scale * (1.0 + 0.5) + VIEW_GAP_V_MM <= ENVELOPE_H_MM
    larger = SCALE_LADDER[SCALE_LADDER.index(scale) - 1]
    assert (
        larger * (2.0 + 0.5) + VIEW_GAP_H_MM > ENVELOPE_W_MM
        or larger * (1.0 + 0.5) + VIEW_GAP_V_MM > ENVELOPE_H_MM
    )


def test_select_scale_falls_back_to_the_smallest_ladder_value():
    assert select_scale(front_w=1e9, front_h=1e9, depth=1e9) == SCALE_LADDER[-1]


def test_empty_projection_is_rejected():
    empty = ViewProjection(visible=ProjectedEdges(), hidden=ProjectedEdges())
    projections = _part()
    object.__setattr__(projections, "right", empty)
    with pytest.raises(DegenerateDrawingError, match="near-zero extent"):
        arrange(projections)


def test_knife_edge_view_is_rejected():
    """A plate too thin to survive scaling collapses to a line, not a drawing.

    The front view is comfortably on-sheet here, so this isolates the extent
    guard from the off-sheet one.
    """
    projections = _part(front_w=2.0, front_h=1.0, depth=1e-9)
    with pytest.raises(DegenerateDrawingError, match="near-zero extent"):
        arrange(projections)


def test_runaway_projection_is_rejected():
    """Nothing on the ladder fits a model this large, so the cluster runs off
    the sheet rather than being silently written out."""
    projections = _part(front_w=1e6, front_h=1e6, depth=1e6)
    with pytest.raises(DegenerateDrawingError, match="runs off the sheet"):
        arrange(projections)


def test_non_finite_geometry_is_rejected():
    projections = _part()
    projections.front.visible.segments.append(Segment((0.0, 0.0), (math.inf, math.nan)))
    with pytest.raises(DegenerateDrawingError):
        arrange(projections)


def test_min_view_extent_is_the_rejection_threshold():
    """Just above the threshold is accepted, so the guard is not over-eager."""
    for view in _all(arrange(_part(front_w=1.0, front_h=1.0, depth=1.0))):
        assert view.bbox[2] - view.bbox[0] >= MIN_VIEW_EXTENT_MM
        assert view.bbox[3] - view.bbox[1] >= MIN_VIEW_EXTENT_MM
