"""What one view's DXF holds, and what it must not hold.

The projections are built by hand, so these run without OCC.
"""

import math

import ezdxf
import pytest

from zeroshot.pipeline.verification.render._hlr import (
    Arc,
    Circle,
    Ellipse,
    Polyline,
    ProjectedEdges,
    Segment,
    ViewProjection,
)
from zeroshot.pipeline.verification.render.export_dxf import export_view


def _view(
    visible: ProjectedEdges | None = None,
    hidden: ProjectedEdges | None = None,
) -> ViewProjection:
    return ViewProjection(
        visible=visible or ProjectedEdges(),
        hidden=hidden or ProjectedEdges(),
    )


def _written(tmp_path, projection: ViewProjection, layer: str = "front"):
    path = tmp_path / f"{layer}.dxf"
    export_view(path, projection, layer)
    return ezdxf.readfile(path)


def test_a_view_is_one_layer_and_its_edges_keep_their_linetypes(tmp_path):
    doc = _written(
        tmp_path,
        _view(
            visible=ProjectedEdges(segments=[Segment((0.0, 0.0), (1.0, 0.0))]),
            hidden=ProjectedEdges(segments=[Segment((2.0, 0.0), (3.0, 0.0))]),
        ),
    )

    assert "front" in doc.layers
    assert {entity.dxf.layer for entity in doc.modelspace()} == {"front"}
    assert [entity.dxf.linetype for entity in doc.modelspace()] == [
        "Continuous",
        "HIDDEN",
    ]


def test_a_view_carries_nothing_but_its_own_linework(tmp_path):
    """No sheet frame, and no centre marks: the reader throws those away."""
    doc = _written(
        tmp_path,
        _view(visible=ProjectedEdges(circles=[Circle((6.0, 6.0), 3.0)])),
    )

    assert [entity.dxftype() for entity in doc.modelspace()] == ["CIRCLE"]


def test_every_primitive_kind_survives_being_written(tmp_path):
    visible = ProjectedEdges(
        segments=[Segment((0.0, 0.0), (1.0, 1.0))],
        arcs=[
            Arc((2.0, 2.0), 1.0, math.pi / 6, math.pi / 2),
            Arc((4.0, 2.0), 1.0, math.pi / 6, math.pi / 2, ccw=False),
        ],
        circles=[Circle((6.0, 6.0), 3.0), Circle((6.1, 6.1), 5.0)],
        ellipses=[Ellipse((8.0, 8.0), 2.0, 1.0, math.pi / 4)],
        polylines=[
            Polyline([(0.0, 0.0), (1.0, 0.2), (2.0, 0.8), (3.0, 1.0)]),
            Polyline([(0.0, 2.0), (1.0, 3.0)]),
        ],
    )
    hidden = ProjectedEdges(
        segments=[Segment((1.0, 0.0), (2.0, 1.0))],
        circles=[Circle((20.0, 20.0), 4.0)],
    )

    entities = list(_written(tmp_path, _view(visible, hidden)).modelspace())

    assert [entity.dxftype() for entity in entities] == [
        "LINE",
        "ARC",
        "ARC",
        "CIRCLE",
        "CIRCLE",
        "ELLIPSE",
        "SPLINE",
        "LWPOLYLINE",
        "LINE",
        "CIRCLE",
    ]
    assert all(entity.dxf.linetype == "Continuous" for entity in entities[:8])
    assert all(entity.dxf.linetype == "HIDDEN" for entity in entities[8:])


def test_the_extents_header_reports_the_view_and_not_a_sheet(tmp_path):
    """A reader that trusts the header must be told the view's own size."""
    projection = _view(
        visible=ProjectedEdges(segments=[Segment((0.0, 0.0), (40.0, 25.0))]),
        hidden=ProjectedEdges(circles=[Circle((10.0, 10.0), 2.0)]),
    )

    doc = _written(tmp_path, projection)

    assert tuple(doc.header["$EXTMIN"])[:2] == pytest.approx((0.0, 0.0))
    assert tuple(doc.header["$EXTMAX"])[:2] == pytest.approx((40.0, 25.0))
