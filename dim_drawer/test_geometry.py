"""Runnable regression check: python -m dim_drawer.test_geometry."""

import random
from pathlib import Path
from tempfile import TemporaryDirectory

import ezdxf
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer

from dim_drawer.dimensions import (
    DimensionWriter,
    Occupancy,
    create_text_styles,
    sample_style,
)
from dim_drawer.extract import extract_doc
from dim_drawer.placement import _significant_coords, dimension_view
from dim_drawer.scale_step import _extents, _read_step, drawing_extents, scale_step
from dim_drawer.views import split_views


def test_geometry():
    # A short edge in the same bucket must not move the longest edge's anchor.
    segments = [(13.3, 0, 13.3, 1), (13.3558341369, 0, 13.3558341369, 20)]
    assert _significant_coords(segments, "x", 0.4, 1, 1) == [13.3558341369]
    assert _significant_coords([(0, 7.123, 20, 7.123)], "y", 0.4, 1, 1) == [7.123]

    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (40, 0), (40, 20), (0, 20)], close=True)
    # A 30-degree circular arc must remain eligible for a radius dimension.
    msp.add_arc((20, 10), 4, 0, 30)
    view = split_views(extract_doc(doc))[0]
    create_text_styles(doc)
    writer = DimensionWriter(doc, msp, sample_style(random.Random(0)))
    dimension_view(writer, view, [], (-100, -100, 200, 200), Occupancy())
    radii = [e.get_measurement() for e in msp.query("DIMENSION") if e.dimtype == 4]
    assert len(radii) == 1 and abs(radii[0] - 4) < 1e-8

    with TemporaryDirectory() as scratch:
        root = Path(scratch)
        doc = ezdxf.new(setup=True, units=ezdxf.units.MM)
        msp = doc.modelspace()
        for x, y, w, h in [(0, 0, 40, 20), (0, 60, 40, 10), (100, 0, 10, 20)]:
            msp.add_lwpolyline(
                [(x, y), (x + w, y), (x + w, y + h), (x, y + h)], close=True
            )
        mark = doc.blocks.new("SW_CENTERMARKSYMBOL_0")
        mark.add_line((-50, 0), (50, 0))
        msp.add_blockref(mark.name, (20, 10))
        doc.layers.add("marks", linetype="CENTER")
        msp.add_line((-80, 5), (80, 5), dxfattribs={"layer": "marks"})
        assert len(extract_doc(doc)["line"]) == 12
        assert len(msp) == 5  # Read-only extraction keeps both marks in the DXF.
        dxf = root / "part.dxf"
        doc.saveas(dxf)
        assert drawing_extents(dxf) == (40, 20, 40, 10, 10, 20)

        source = root / "part.step"
        writer = STEPControl_Writer()
        assert (
            writer.Transfer(BRepPrimAPI_MakeBox(4, 2, 1).Shape(), STEPControl_AsIs) == 1
        )
        assert writer.Write(str(source)) == 1
        original = source.read_bytes()
        output = root / "scaled" / source.name
        factor, error = scale_step(source, dxf, output)
        assert abs(factor - 10) < 1e-8 and error < 1e-6
        assert all(
            abs(a - b) < 1e-6
            for a, b in zip(
                _extents(_read_step(output)), drawing_extents(dxf), strict=True
            )
        )
        assert source.read_bytes() == original

        # Wrong proportions must fail without replacing an existing output.
        saved = output.read_bytes()
        msp.add_line((110, 0), (115, 0))
        doc.saveas(dxf)
        try:
            scale_step(source, dxf, output)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted inconsistent view scales")
        assert output.read_bytes() == saved


if __name__ == "__main__":
    test_geometry()
    print("geometry regression check passed")
