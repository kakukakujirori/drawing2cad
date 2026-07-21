from __future__ import annotations

import queue
import sys
from pathlib import Path
from types import ModuleType

from src.data.render.config import TechdrawPaths
from src.data.render.render_dataset import _load_done, _worker


def test_worker_records_techdraw_info_in_manifest_extra(monkeypatch) -> None:
    info = {
        "scale": 1.0,
        "bbox_format": "xyxy",
        "bbox_coordinate_system": {
            "unit": "mm",
            "origin": "sheet_bottom_left",
            "x_axis": "right",
            "y_axis": "up",
        },
        "cluster_bbox": [10.0, 20.0, 100.0, 120.0],
        "views": {
            "front": {"bbox": [10.0, 20.0, 50.0, 60.0]},
            "top": {"bbox": [10.0, 85.0, 50.0, 120.0]},
            "right": {"bbox": [85.0, 20.0, 100.0, 60.0]},
        },
    }
    techdraw_module = ModuleType("src.data.render.techdraw")
    techdraw_module.generate_techdraw = lambda step, paths: info
    monkeypatch.setitem(sys.modules, "src.data.render.techdraw", techdraw_module)

    paths = TechdrawPaths(Path("part.svg"), Path("part.dxf"), Path("part.pdf"))
    result_queue: queue.Queue = queue.Queue()

    _worker(Path("part.step"), paths, None, result_queue)

    record = result_queue.get_nowait()
    assert record["ok"] is True
    assert record["techdraw_ok"] is True
    assert record["extra"] == {"techdraw": info}


def test_load_done_requires_current_techdraw_metadata(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        '{"name":"old","ok":true,"extra":{}}\n'
        '{"name":"new","ok":true,"extra":{"techdraw":{"bbox_format":"xyxy"}}}\n',
        encoding="utf-8",
    )

    assert _load_done(manifest) == {"old", "new"}
    assert _load_done(manifest, require_techdraw_info=True) == {"new"}
