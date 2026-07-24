from __future__ import annotations

import queue
import sys
from pathlib import Path
from types import ModuleType

import ezdxf

from src.data.render.config import Render3dPaths, TechdrawPaths
from src.data.render.hlr import ProjectedEdges, Segment
from src.data.render.layout import PlacedView
from src.data.render.render_dataset import _outputs_exist, _worker
from src.data.render.writers.dxf_writer import write_dxf


def _view(name: str, y0: float) -> PlacedView:
    """A minimal placed view carrying one visible and one hidden segment."""
    visible = ProjectedEdges()
    visible.segments = [Segment((20.0, y0), (60.0, y0))]
    hidden = ProjectedEdges()
    hidden.segments = [Segment((20.0, y0 + 5.0), (60.0, y0 + 5.0))]
    return PlacedView(name, visible, hidden, (20.0, y0, 60.0, y0 + 5.0))


def test_worker_writes_geometry_on_view_layers(monkeypatch, tmp_path) -> None:
    """`_worker` runs techdraw and the DXF carries geometry on per-view layers.

    The manifest is abolished, so the worker no longer stores any techdraw
    metadata in ``extra`` -- it only reports success and leaves output files
    behind. The DXF the writer produces must place each view's edges on the
    layer named for that view ("front"/"top"/"right"), which is the exact
    contract the DXF parser reads (entity.dxf.layer -> view direction).
    """
    views = [_view("front", 20.0), _view("top", 120.0), _view("right", 20.0)]

    def fake_generate(step, paths):
        write_dxf(paths.dxf, views, [])
        return None  # new contract: no manifest info returned

    techdraw_module = ModuleType("src.data.render.techdraw")
    techdraw_module.generate_techdraw = fake_generate
    monkeypatch.setitem(sys.modules, "src.data.render.techdraw", techdraw_module)

    dxf_path = tmp_path / "part.dxf"
    paths = TechdrawPaths(tmp_path / "part.svg", dxf_path, tmp_path / "part.pdf")
    result_queue: queue.Queue = queue.Queue()

    _worker(Path("part.step"), paths, None, result_queue)

    record = result_queue.get_nowait()
    assert record["ok"] is True
    assert record["techdraw_ok"] is True
    # No manifest metadata is carried anymore.
    assert record["extra"] == {}

    doc = ezdxf.readfile(dxf_path)
    layers = {e.dxf.layer for e in doc.modelspace()}
    assert {"front", "top", "right"} <= layers
    # Geometry never lands on the old catch-all layer "0".
    assert "0" not in layers
    # The view layers exist in the layer table, not only on the entities.
    for name in ("front", "top", "right"):
        assert name in doc.layers


def test_worker_reports_failure_via_partresult(monkeypatch, tmp_path) -> None:
    """A techdraw exception becomes ok=False in the queued PartResult.

    Previously this outcome was inferred from the manifest; now the worker's
    PartResult is the only signal (main() logs it to render_errors.jsonl).
    """

    def boom(step, paths):
        raise RuntimeError("projection failed")

    techdraw_module = ModuleType("src.data.render.techdraw")
    techdraw_module.generate_techdraw = boom
    monkeypatch.setitem(sys.modules, "src.data.render.techdraw", techdraw_module)

    paths = TechdrawPaths(tmp_path / "p.svg", tmp_path / "p.dxf", tmp_path / "p.pdf")
    result_queue: queue.Queue = queue.Queue()

    _worker(Path("p.step"), paths, None, result_queue)

    record = result_queue.get_nowait()
    assert record["ok"] is False
    assert record["techdraw_ok"] is False
    assert "projection failed" in record["error"]


def test_resume_is_driven_by_outputs_exist(tmp_path) -> None:
    """Resume skips a part only when every requested output file is present.

    This replaces the old manifest-based ``_load_done`` gate: skipping is now a
    pure function of which output files exist on disk.
    """
    td = TechdrawPaths(tmp_path / "s.svg", tmp_path / "s.dxf", tmp_path / "s.pdf")
    r3 = Render3dPaths(tmp_path / "a.png", tmp_path / "b.png", tmp_path / "c.png")

    # Nothing written yet: not resumable.
    assert _outputs_exist(td, None) is False
    assert _outputs_exist(td, r3) is False

    for p in (td.svg, td.dxf, td.pdf):
        p.write_text("x")
    # All techdraw outputs present -> skip when only techdraw is requested.
    assert _outputs_exist(td, None) is True
    # A partial techdraw set is not enough.
    td.pdf.unlink()
    assert _outputs_exist(td, None) is False
    td.pdf.write_text("x")

    # render_3d still missing -> not skipped when both are requested.
    assert _outputs_exist(td, r3) is False
    for p in (r3.hlg, r3.shaded, r3.hlg_translucent):
        p.write_text("x")
    assert _outputs_exist(td, r3) is True
