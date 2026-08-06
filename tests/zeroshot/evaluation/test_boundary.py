"""The ground truth must stay out of the inference pipeline.

``zeroshot/evaluation`` is the only package allowed to read a target STEP, and
``zeroshot/run_pipeline.py`` is the only seam that calls it -- after the run is
closed. These tests keep that a property of the tree rather than a convention.
"""

from pathlib import Path

import pytest

from zeroshot.pipeline.messages import InputManifest

PIPELINE_DIR = Path(__file__).resolve().parents[3] / "zeroshot" / "pipeline"


def _pipeline_sources() -> list[Path]:
    sources = sorted(
        path for path in PIPELINE_DIR.rglob("*.py") if "__pycache__" not in path.parts
    )
    assert sources, f"no sources found under {PIPELINE_DIR}"
    return sources


@pytest.mark.parametrize("source", _pipeline_sources(), ids=lambda p: p.name)
def test_pipeline_never_imports_evaluation(source: Path) -> None:
    assert "zeroshot.evaluation" not in source.read_text(encoding="utf-8")


@pytest.mark.parametrize("source", _pipeline_sources(), ids=lambda p: p.name)
def test_pipeline_never_names_a_target(source: Path) -> None:
    text = source.read_text(encoding="utf-8")
    for forbidden in ("target_step", "target_dir"):
        assert forbidden not in text, f"{source.name} names {forbidden!r}"


def test_input_manifest_rejects_a_target(tmp_path: Path) -> None:
    """A config key added next to the drawing cannot reach the sandbox."""

    dxf_path = tmp_path / "input.dxf"
    dxf_path.write_bytes(b"DXF")
    with pytest.raises(TypeError):
        InputManifest(  # type: ignore[call-arg]
            sample_id="sample-1",
            dxf_path=dxf_path,
            render3d_paths={},
            target_step_path=tmp_path / "target.step",
        )
