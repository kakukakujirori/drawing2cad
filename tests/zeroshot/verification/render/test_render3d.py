from pathlib import Path

import pytest
from PIL import Image

from zeroshot.pipeline.verification.render import _render3d
from zeroshot.pipeline.verification.render.constants import Render3dPaths


def _paths(base: Path) -> Render3dPaths:
    return Render3dPaths(
        hlg_perspective=base / "hlg.png",
        transparent_shaded_edges_perspective=base / "shaded.png",
        hlg_translucent_faces_perspective=base / "translucent.png",
    )


def test_generate_render3d_explains_each_unsuccessful_style(tmp_path, monkeypatch):
    paths = _paths(tmp_path)

    monkeypatch.setattr(_render3d, "_load_shape", lambda _path: object())
    monkeypatch.setattr(_render3d, "_build_camera", lambda _shape, _cfg: object())

    def fail_bbox(_shape):
        raise RuntimeError("tessellation failed")

    def render_hlg(_shape, _camera, _cfg, out_path, **_kwargs):
        out_path.write_bytes(b"HLG")
        return object(), object(), object()

    def write_png(*_args, **kwargs):
        Path(kwargs["write_to"]).write_bytes(b"PNG")

    monkeypatch.setattr(_render3d, "_bbox", fail_bbox)
    monkeypatch.setattr(_render3d, "_render_hlg", render_hlg)
    monkeypatch.setattr(_render3d, "_write_hlg_svg", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(_render3d.cairosvg, "svg2png", write_png)

    result = _render3d.generate_render3d(tmp_path / "input.step", paths)

    assert result["rendered"] == [
        "hlg_perspective",
        "hlg_translucent_faces_perspective",
    ]
    assert result["errors"] == {
        "transparent_shaded_edges_perspective": (
            "tessellation: RuntimeError: tessellation failed"
        )
    }


def test_composite_translucent_removes_partial_line_png(tmp_path, monkeypatch):
    out_path = tmp_path / "translucent.png"

    monkeypatch.setattr(_render3d, "_write_hlg_svg", lambda *_args, **_kwargs: "")

    def fail_png(*_args, **kwargs):
        Path(kwargs["write_to"]).write_bytes(b"partial")
        raise RuntimeError("conversion failed")

    monkeypatch.setattr(_render3d.cairosvg, "svg2png", fail_png)

    with pytest.raises(RuntimeError, match="conversion failed"):
        _render3d._composite_translucent(
            Image.new("RGB", (10, 10)),
            object(),
            object(),
            object(),
            _render3d.Render3dConfig(canvas_w=10, canvas_h=10),
            out_path,
        )

    assert not out_path.with_suffix(".lines.tmp.png").exists()
