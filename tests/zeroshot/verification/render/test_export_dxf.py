"""What one view's DXF holds, and what it must not hold."""

import math
from itertools import pairwise

import cadquery as cq
import ezdxf
import numpy as np
import pytest
from ezdxf.path import make_path
from PIL import Image, ImageChops
from scipy.spatial import cKDTree

from zeroshot.pipeline.stages.interpretation.contracts import View
from zeroshot.pipeline.verification.render import orthographic
from zeroshot.pipeline.verification.render.export_dxf import (
    DegenerateDrawingError,
    export_to_png,
    write_view_dxf,
)


def test_png_keeps_all_four_border_lines_inside_white_margins(tmp_path):
    path = tmp_path / "rectangle.dxf"
    doc = ezdxf.new()
    doc.modelspace().add_lwpolyline(
        [(-30, 10), (10, 10), (10, 30), (-30, 30)], close=True
    )
    doc.saveas(path)

    with Image.open(export_to_png(path)) as image:
        ink = ImageChops.invert(image.convert("L")).point(
            lambda p: 255 if p > 127 else 0
        )
        left, top, right, bottom = ink.getbbox()
        assert 5 < left < right < image.width - 5
        assert 5 < top < bottom < image.height - 5
        # Each edge is present along its length, not just at its corners.
        assert ink.getpixel(((left + right) // 2, top))
        assert ink.getpixel(((left + right) // 2, bottom - 1))
        assert ink.getpixel((left, (top + bottom) // 2))
        assert ink.getpixel((right - 1, (top + bottom) // 2))


def _line(start, end) -> cq.Edge:
    return cq.Edge.makeLine(cq.Vector(*start), cq.Vector(*end))


def _dense(points: np.ndarray, step: float) -> np.ndarray:
    """A polyline's vertices, filled in so no two are farther apart than step."""
    runs = [points[:1]]
    for start, end in pairwise(points):
        count = max(1, math.ceil(np.linalg.norm(end - start) / step))
        runs.append(start + (end - start) * np.linspace(0, 1, count + 1)[1:, None])
    return np.vstack(runs)


def _dxf_ink(path, step: float) -> np.ndarray:
    return np.vstack(
        [
            _dense(np.array([(p.x, p.y) for p in make_path(e).flattening(step)]), step)
            for e in ezdxf.readfile(path).modelspace()
        ]
    )


def _edge_ink(edges: list[cq.Edge], step: float) -> np.ndarray:
    return np.vstack(
        [
            np.array(
                [
                    (p.x, p.y)
                    for p in edge.positions(
                        np.linspace(0, 1, max(2, math.ceil(edge.Length() / step) + 1))
                    )
                ]
            )
            for edge in edges
        ]
    )


def _farthest(points: np.ndarray, targets: np.ndarray) -> float:
    """How far the point farthest from every target lies from its nearest one."""
    return float(cKDTree(targets).query(points)[0].max())


def test_a_written_view_is_one_layer_with_its_hidden_edges_dashed(tmp_path):
    path = tmp_path / "front.dxf"

    write_view_dxf(
        path, [_line((0, 0, 0), (10, 5, 0))], [_line((2, 1, 0), (8, 1, 0))], "front"
    )

    doc = ezdxf.readfile(path)
    assert "front" in doc.layers
    assert {entity.dxf.layer for entity in doc.modelspace()} == {"front"}
    assert [entity.dxf.linetype for entity in doc.modelspace()] == [
        "Continuous",
        "HIDDEN",
    ]


def test_every_edge_kind_is_drawn_where_the_edge_runs(tmp_path):
    """A DXF arc always turns counter-clockwise; one that turns the other way keeps its span."""
    edges = [
        _line((0, 0, 0), (10, 0, 0)),
        cq.Edge.makeCircle(3, cq.Vector(20, 0, 0), cq.Vector(0, 0, 1), 0, 90),
        cq.Edge.makeCircle(3, cq.Vector(30, 0, 0), cq.Vector(0, 0, -1), 0, 90),
        cq.Edge.makeCircle(2, cq.Vector(40, 0, 0)),
        cq.Edge.makeEllipse(
            4, 2, cq.Vector(50, 0, 0), cq.Vector(0, 0, 1), cq.Vector(1, 0, 0), 10, 80
        ),
        cq.Edge.makeEllipse(
            4, 2, cq.Vector(60, 0, 0), cq.Vector(0, 0, -1), cq.Vector(1, 0, 0), 10, 80
        ),
        cq.Edge.makeSpline(
            [
                cq.Vector(70, 0, 0),
                cq.Vector(72, 3, 0),
                cq.Vector(75, -1, 0),
                cq.Vector(78, 0, 0),
            ]
        ),
    ]
    path = tmp_path / "front.dxf"

    write_view_dxf(path, edges, [], "front")

    kinds = [entity.dxftype() for entity in ezdxf.readfile(path).modelspace()]
    assert kinds == ["LINE", "ARC", "ARC", "CIRCLE", "ELLIPSE", "ELLIPSE", "SPLINE"]
    ink, drawn = _dxf_ink(path, 0.01), _edge_ink(edges, 0.01)
    assert _farthest(ink, drawn) < 0.02
    assert _farthest(drawn, ink) < 0.02


# Its rim fillet once came out as arcs spanning the rest of their circles.
ROUNDED_PLATE = (
    cq.Workplane()
    .rect(100, 38)
    .extrude(11.5)
    .edges("|Z")
    .fillet(5.4)
    .edges(">Z")
    .fillet(5.79)
)


@pytest.mark.parametrize("view", [View.FRONT, View.TOP, View.RIGHT])
def test_a_filleted_rim_is_drawn_along_its_projected_edges(tmp_path, view):
    plate = ROUNDED_PLATE.findSolid()
    tol = orthographic.TOL_FRAC * plate.BoundingBox().DiagonalLength
    visible, hidden = orthographic.project(
        plate, *orthographic.STANDARD_VIEW_FRAMES[view]
    )
    path = tmp_path / f"{view.value}.dxf"

    write_view_dxf(path, visible, hidden, view.value)

    ink = _dxf_ink(path, tol / 4)
    assert _farthest(ink, _edge_ink(visible + hidden, tol / 4)) < tol
    assert _farthest(_edge_ink(visible, tol / 4), ink) < tol


def test_the_extents_header_reports_the_written_view(tmp_path):
    """A reader that trusts the header must be told the view's own size."""
    path = tmp_path / "front.dxf"

    write_view_dxf(
        path,
        [_line((-5, 2, 0), (40, 25, 0))],
        [cq.Edge.makeCircle(2, cq.Vector(10, 10, 0))],
        "front",
    )

    header = ezdxf.readfile(path).header
    assert tuple(header["$EXTMIN"])[:2] == pytest.approx((-5.0, 2.0))
    assert tuple(header["$EXTMAX"])[:2] == pytest.approx((40.0, 25.0))


@pytest.mark.parametrize(
    "visible",
    [[], [_line((0, 0, 0), (10, 0, 0))]],
    ids=["empty", "knife_edge"],
)
def test_a_view_with_no_area_is_refused_by_name(tmp_path, visible):
    path = tmp_path / "right.dxf"

    with pytest.raises(DegenerateDrawingError, match="right"):
        write_view_dxf(path, visible, [], "right")
    assert not path.exists()
