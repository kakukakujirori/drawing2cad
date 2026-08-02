"""DXF-format boundary and centre-mark contracts."""

import math

import ezdxf
import pytest
from ezdxf.bbox import extents

from zeroshot.pipeline.verification.render._hlr import (
    Arc,
    Circle,
    Ellipse,
    Polyline,
    ProjectedEdges,
    Segment,
    ViewProjection,
)
from zeroshot.pipeline.verification.render.arrange import Layout, PlacedView, arrange
from zeroshot.pipeline.verification.render.export_dxf import (
    CENTERMARK_CROSS_MM,
    CENTERMARK_EXT_MM,
    CENTERMARK_GAP_MM,
    export_dxf,
)
from zeroshot.pipeline.verification.render.project import ViewProjections

VIEW_LAYERS = ("front", "top", "right")


def _view(
    visible: ProjectedEdges | None = None,
    hidden: ProjectedEdges | None = None,
) -> PlacedView:
    return PlacedView(
        visible=visible or ProjectedEdges(),
        hidden=hidden or ProjectedEdges(),
        bbox=(0.0, 0.0, 10.0, 10.0),
    )


def _empty_layout(*, front: PlacedView | None = None) -> Layout:
    return Layout(front=front or _view(), top=_view(), right=_view())


def _rect(width: float, height: float, circles: tuple = ()) -> ViewProjection:
    """A rectangular outline, optionally with holes, as a visible-only projection."""
    corners = [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)]
    return ViewProjection(
        visible=ProjectedEdges(
            segments=[
                Segment(corners[i], corners[(i + 1) % 4]) for i in range(len(corners))
            ],
            circles=[Circle(center, radius) for center, radius in circles],
        ),
        hidden=ProjectedEdges(),
    )


def _arranged(circles: tuple = ()) -> Layout:
    """A really placed drawing, so that a view written to the wrong layer shows.

    The three views have different extents and only ``front`` carries holes, so
    no pair of layers can be swapped without failing the assertions below.  The
    projections are built by hand, so this stays OCC-free like ``arrange``.
    """
    return arrange(
        ViewProjections(
            front=_rect(2.0, 1.0, circles),
            top=_rect(2.0, 0.5),
            right=_rect(0.5, 1.0),
        )
    )


def _layer_bbox(doc, layer: str) -> tuple[float, float, float, float]:
    """Read one layer's extent back out of a written DXF.

    This is how the planned GT aligner recovers each view, so the tests below
    exercise the same path rather than reading ``PlacedView.bbox`` directly.
    """
    box = extents(entity for entity in doc.modelspace() if entity.dxf.layer == layer)
    assert box.has_data
    return (box.extmin.x, box.extmin.y, box.extmax.x, box.extmax.y)


def test_view_fields_become_nonempty_dxf_layers_with_edge_linetypes(tmp_path):
    def placed(y: float) -> PlacedView:
        return _view(
            visible=ProjectedEdges(segments=[Segment((0.0, y), (1.0, y))]),
            hidden=ProjectedEdges(segments=[Segment((2.0, y), (3.0, y))]),
        )

    path = tmp_path / "views.dxf"
    export_dxf(path, Layout(front=placed(1.0), top=placed(2.0), right=placed(3.0)))
    doc = ezdxf.readfile(path)

    assert all(layer in doc.layers for layer in ("front", "top", "right"))
    for layer in ("front", "top", "right"):
        entities = [entity for entity in doc.modelspace() if entity.dxf.layer == layer]
        assert [entity.dxf.linetype for entity in entities] == [
            "Continuous",
            "HIDDEN",
        ]


def test_each_view_layer_reports_the_extent_it_was_placed_at(tmp_path):
    """Per-layer bboxes must be recoverable from the file, and must be the ones
    ``arrange`` chose.  The planned GT aligner has nothing else to work from: it
    reads these three extents and moves each layer onto its GT counterpart."""
    layout = _arranged()
    path = tmp_path / "extents.dxf"
    export_dxf(path, layout)
    doc = ezdxf.readfile(path)

    for layer in VIEW_LAYERS:
        assert _layer_bbox(doc, layer) == pytest.approx(getattr(layout, layer).bbox)


def test_layer_bboxes_hold_the_third_angle_arrangement(tmp_path):
    """Read back from the DXF, the layers must still be in the L-arrangement --
    which fails if a view's geometry is written to another view's layer."""
    path = tmp_path / "arrangement.dxf"
    export_dxf(path, _arranged())
    doc = ezdxf.readfile(path)
    front = _layer_bbox(doc, "front")
    top = _layer_bbox(doc, "top")
    right = _layer_bbox(doc, "right")

    assert top[0] + top[2] == pytest.approx(front[0] + front[2])  # shared x centre
    assert right[1] + right[3] == pytest.approx(front[1] + front[3])  # shared y centre
    assert top[1] > front[3]  # top sits above front
    assert right[0] > front[2]  # right sits beside front


def test_geometry_stays_on_view_layers_and_marks_stay_on_layer_10(tmp_path):
    """The layer of an entity is what identifies its view, so no geometry may
    land anywhere else and layer "10" must hold annotations only."""
    path = tmp_path / "layers.dxf"
    export_dxf(path, _arranged(circles=(((0.5, 0.5), 0.2), ((1.5, 0.5), 0.2))))
    doc = ezdxf.readfile(path)

    used = {entity.dxf.layer for entity in doc.modelspace()}
    assert used == {*VIEW_LAYERS, "10"}
    for entity in doc.modelspace():
        is_mark = entity.dxftype() == "INSERT"
        assert entity.dxf.layer == "10" if is_mark else entity.dxf.layer in VIEW_LAYERS


def test_all_primitives_and_center_marks_are_exported(tmp_path):
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
    path = tmp_path / "primitives.dxf"
    export_dxf(path, _empty_layout(front=_view(visible, hidden)))
    doc = ezdxf.readfile(path)
    front = [entity for entity in doc.modelspace() if entity.dxf.layer == "front"]

    assert [entity.dxftype() for entity in front] == [
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
    assert all(entity.dxf.linetype == "Continuous" for entity in front[:8])
    assert all(entity.dxf.linetype == "HIDDEN" for entity in front[8:])

    inserts = [
        entity
        for entity in doc.modelspace()
        if entity.dxftype() == "INSERT" and entity.dxf.layer == "10"
    ]
    assert (
        len(inserts) == 1
    )  # two visible concentric circles collapse; hidden is ignored
    insert = inserts[0]
    assert tuple(insert.dxf.insert)[:2] == pytest.approx((6.0, 6.0))
    lines = [
        entity
        for entity in doc.blocks.get(insert.dxf.name)
        if entity.dxftype() == "LINE"
    ]
    assert len(lines) == 8
    endpoints = [
        coordinate for line in lines for coordinate in (*line.dxf.start, *line.dxf.end)
    ]
    assert CENTERMARK_CROSS_MM in endpoints
    assert CENTERMARK_CROSS_MM + CENTERMARK_GAP_MM in endpoints
    assert 5.0 + CENTERMARK_EXT_MM in endpoints


def test_short_center_mark_keeps_eight_connected_segments(tmp_path):
    radius = 2.0
    path = tmp_path / "short_mark.dxf"
    front = _view(visible=ProjectedEdges(circles=[Circle((6.0, 6.0), radius)]))
    export_dxf(path, _empty_layout(front=front))
    doc = ezdxf.readfile(path)
    insert = next(entity for entity in doc.modelspace() if entity.dxftype() == "INSERT")
    lines = [
        entity
        for entity in doc.blocks.get(insert.dxf.name)
        if entity.dxftype() == "LINE"
    ]

    assert len(lines) == 8
    positive_y_inner, positive_y_outer = lines[0], lines[4]
    assert tuple(positive_y_inner.dxf.start)[:2] == pytest.approx((0.0, 0.0))
    assert tuple(positive_y_inner.dxf.end)[:2] == pytest.approx(
        (0.0, CENTERMARK_CROSS_MM)
    )
    assert tuple(positive_y_outer.dxf.start)[:2] == pytest.approx(
        (0.0, CENTERMARK_CROSS_MM)
    )
    assert tuple(positive_y_outer.dxf.end)[:2] == pytest.approx(
        (0.0, radius + CENTERMARK_EXT_MM)
    )
