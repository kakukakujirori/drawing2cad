"""Solids and run directories for the scoring tests.

Every fixture is generated with CadQuery. ``data/`` and ``outputs/`` are not
tracked by git, so the suite must not reach for either.
"""

import shutil
from pathlib import Path

import pytest

from zeroshot.pipeline.runner import PipelineRunner


@pytest.fixture(scope="session")
def solids(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    import cadquery as cq

    directory = tmp_path_factory.mktemp("solids")
    shapes = {
        "box": cq.Workplane("XY").box(30, 20, 10),
        "fillet": cq.Workplane("XY").box(30, 20, 10).edges("|Z").fillet(3),
        # A curved pair ten times apart in size. Scale invariance has to be
        # checked on a curve: a box is planar, so no chord error can change how
        # it triangulates.
        "sphere": cq.Workplane("XY").sphere(15),
        "sphere10": cq.Workplane("XY").sphere(150),
        # 20 units long, but its B-spline control hull reaches 22.06, which is
        # what an untriangulated `BRepBndLib.Add` reports.
        "loft": cq.Workplane("XY").circle(10).workplane(offset=20).rect(20, 6).loft(),
    }

    paths: dict[str, Path] = {}
    for name, shape in shapes.items():
        path = directory / f"{name}.step"
        cq.exporters.export(shape, str(path), exportType="STEP")
        paths[name] = path
    return paths


@pytest.fixture
def junk_step(tmp_path: Path) -> Path:
    path = tmp_path / "junk.step"
    path.write_text("not a STEP file", encoding="utf-8")
    return path


@pytest.fixture
def make_run_dir(tmp_path: Path):
    """Build a sample artifact directory from ``{attempt_id: solid or None}``."""

    def build(attempts: dict[str, Path | None]) -> Path:
        run_dir = tmp_path / "run"
        attempts_dir = run_dir / PipelineRunner.WORKSPACE_DIRNAME / "attempts"
        attempts_dir.mkdir(parents=True)
        for name, step in attempts.items():
            attempt = attempts_dir / name
            attempt.mkdir()
            if step is not None:
                shutil.copyfile(step, attempt / "output.step")
        return run_dir

    return build
