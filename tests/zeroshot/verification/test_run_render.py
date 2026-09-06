"""Process supervision and artifact reporting for trusted view rendering."""

from dataclasses import fields
from pathlib import Path

import cadquery as cq
import pytest
from PIL import Image

from zeroshot.pipeline.verification import run_render
from zeroshot.pipeline.verification.render.constants import (
    ProjectionPaths,
    Render3dPaths,
)
from zeroshot.pipeline.verification.render.project import DegenerateDrawingError
from zeroshot.pipeline.verification.run_render import RenderStatus, StepRenderer


def _paths(base: Path) -> Render3dPaths:
    return Render3dPaths(
        hlg_perspective=base / "custom-hlg.png",
        transparent_shaded_edges_perspective=base / "custom-shaded.png",
        hlg_translucent_faces_perspective=base / "custom-translucent.png",
    )


def _projection_paths(base: Path) -> ProjectionPaths:
    return ProjectionPaths(
        front=base / "custom-front.dxf",
        top=base / "custom-top.dxf",
        right=base / "custom-right.dxf",
    )


def _write_projections(
    _step_path: Path,
    paths: ProjectionPaths,
) -> tuple[ProjectionPaths, dict[str, str]]:
    for path in paths.as_mapping().values():
        path.write_text("DXF", encoding="utf-8")
    return paths, {}


_EVERY_VIEW = ("front", "top", "right")


def _present_paths(paths: ProjectionPaths | Render3dPaths) -> set[Path]:
    return {
        path
        for path_field in fields(paths)
        if (path := getattr(paths, path_field.name)) is not None
    }


def test_partial_render_reports_only_successful_paths(tmp_path, monkeypatch):
    projection_paths = _projection_paths(tmp_path)
    paths = _paths(tmp_path)

    def render3d(_step_path, output_paths):
        output_paths.hlg_perspective.write_bytes(b"HLG")
        output_paths.hlg_translucent_faces_perspective.write_bytes(b"TRANSLUCENT")
        return {
            "rendered": (
                "hlg_perspective",
                "hlg_translucent_faces_perspective",
            ),
            "errors": {
                "transparent_shaded_edges_perspective": (
                    "RuntimeError: shaded pass failed"
                )
            },
        }

    monkeypatch.setattr(run_render, "_render_projections", _write_projections)
    monkeypatch.setattr(run_render, "generate_render3d", render3d)

    report = run_render._render_once(
        tmp_path / "input.step",
        projection_paths,
        paths,
    )

    assert report.status is RenderStatus.PARTIAL
    assert report.projection_paths == projection_paths
    assert report.render3d_paths == Render3dPaths(
        hlg_perspective=paths.hlg_perspective,
        hlg_translucent_faces_perspective=(paths.hlg_translucent_faces_perspective),
    )
    assert report.projection_errors == {}
    assert report.render3d_errors == {
        "transparent_shaded_edges_perspective": "RuntimeError: shaded pass failed"
    }
    assert not paths.transparent_shaded_edges_perspective.exists()


def test_degenerate_projection_keeps_the_reason(tmp_path, monkeypatch):
    def reject_projection(_step_path, _paths):
        raise DegenerateDrawingError("right view has near-zero extent")

    def reject_render3d(_step_path, _paths):
        raise RuntimeError("perspective rendering failed")

    monkeypatch.setattr(run_render, "_render_projections", reject_projection)
    monkeypatch.setattr(run_render, "_render_3d", reject_render3d)

    report = run_render._render_once(
        tmp_path / "input.step",
        _projection_paths(tmp_path),
        _paths(tmp_path),
    )

    assert report.status is RenderStatus.FAILED
    assert report.projection_paths == ProjectionPaths()
    assert report.render3d_paths == Render3dPaths()
    projection_error = "DegenerateDrawingError: right view has near-zero extent"
    render3d_error = "RuntimeError: perspective rendering failed"
    assert report.projection_errors == dict.fromkeys(_EVERY_VIEW, projection_error)
    assert report.render3d_errors == {
        "hlg_perspective": render3d_error,
        "transparent_shaded_edges_perspective": render3d_error,
        "hlg_translucent_faces_perspective": render3d_error,
    }


class _ClosedConnection:
    def poll(self, timeout=None):
        del timeout
        return False

    def close(self):
        pass


class _HangingProcess:
    exitcode = None

    def __init__(self):
        self.alive = True
        self.terminated = False
        self.killed = False

    def start(self):
        pass

    def join(self, timeout=None):
        del timeout

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True
        self.alive = False


class _HangingContext:
    def __init__(self):
        self.process = _HangingProcess()

    def Pipe(self, duplex=False):
        assert duplex is False
        return _ClosedConnection(), _ClosedConnection()

    def Process(self, **kwargs):
        assert kwargs["daemon"] is True
        return self.process


def test_timeout_is_distinct_and_kills_a_stuck_process(tmp_path, monkeypatch):
    context = _HangingContext()
    monkeypatch.setattr(run_render.mp, "get_context", lambda method: context)
    projection_paths = _projection_paths(tmp_path)
    paths = _paths(tmp_path)

    report = StepRenderer(timeout_s=0.1).render(
        tmp_path / "input.step",
        projection_paths,
        paths,
    )

    assert report.status is RenderStatus.TIMEOUT
    assert report.projection_paths == ProjectionPaths()
    assert report.render3d_paths == Render3dPaths()
    message = "render timed out after 0.1s"
    assert report.projection_errors == dict.fromkeys(_EVERY_VIEW, message)
    assert report.render3d_errors == {
        "hlg_perspective": message,
        "transparent_shaded_edges_perspective": message,
        "hlg_translucent_faces_perspective": message,
    }
    assert context.process.terminated is True
    assert context.process.killed is True


def test_real_box_renders_to_caller_assigned_paths(tmp_path):
    step_path = tmp_path / "box.step"
    cq.exporters.export(
        cq.Workplane("XY").box(10.0, 20.0, 30.0),
        str(step_path),
        exportType="STEP",
    )
    output_dir = tmp_path / "assigned"
    output_dir.mkdir()
    projection_paths = _projection_paths(output_dir)
    paths = _paths(output_dir)

    report = StepRenderer(timeout_s=60.0).render(
        step_path,
        projection_paths,
        paths,
    )

    assert report.status is RenderStatus.OK
    assert report.projection_paths == projection_paths
    assert report.render3d_paths == paths
    assert report.projection_errors == {}
    assert report.render3d_errors == {}
    # Each view is written twice: the DXF a reader measures, and a picture of
    # it beside itself.
    assert set(output_dir.iterdir()) == {
        *_present_paths(report.projection_paths),
        *(path.with_suffix(".png") for path in _present_paths(report.projection_paths)),
        *_present_paths(report.render3d_paths),
    }
    for path in _present_paths(report.render3d_paths):
        with Image.open(path) as image:
            assert image.size == (1400, 1000)
            assert image.mode == "RGB"


def test_every_error_key_names_a_field_of_its_paths_dto(tmp_path, monkeypatch):
    """The two error maps live in different namespaces; a key that matches no
    field would describe an artifact no consumer can pair with a path."""
    projection_fields = {f.name for f in fields(ProjectionPaths)}
    render3d_fields = {f.name for f in fields(Render3dPaths)}

    def reject(_step_path, _paths):
        raise RuntimeError("boom")

    monkeypatch.setattr(run_render, "_render_projections", reject)
    monkeypatch.setattr(run_render, "_render_3d", reject)
    report = run_render._render_once(
        tmp_path / "input.step", _projection_paths(tmp_path), _paths(tmp_path)
    )

    assert set(report.projection_errors) <= projection_fields
    assert set(report.render3d_errors) <= render3d_fields
    assert report.projection_errors and report.render3d_errors


def test_batch_renders_every_request_in_order_within_the_worker_limit(tmp_path):
    """Two more requests than workers, so the last ones only start once slots free."""
    boxes = [cq.Workplane("XY").box(10.0 + index, 20.0, 30.0) for index in range(4)]
    requests = []
    for index, box in enumerate(boxes):
        directory = tmp_path / f"job{index}"
        directory.mkdir()
        step_path = directory / "shape.step"
        cq.exporters.export(box, str(step_path), exportType="STEP")
        requests.append(
            run_render.RenderRequest(
                step_path,
                _projection_paths(directory),
                _paths(directory),
            )
        )

    reports = StepRenderer(timeout_s=120.0, max_workers=2).render_many(requests)

    assert len(reports) == len(requests)
    for request, report in zip(requests, reports, strict=True):
        assert report.status is RenderStatus.OK
        assert report.projection_paths == request.projection_paths
        assert report.render3d_paths == request.render3d_paths
        for path in _present_paths(report.render3d_paths):
            assert path.is_file()


def test_an_empty_batch_starts_no_process(monkeypatch):
    def fail(_method):
        raise AssertionError("no context is needed for an empty batch")

    monkeypatch.setattr(run_render.mp, "get_context", fail)

    assert StepRenderer(timeout_s=1.0).render_many([]) == []


class _CountingProcess:
    exitcode = 0

    def __init__(self):
        self.alive = True
        self.terminated = False

    def start(self):
        pass

    def join(self, timeout=None):
        del timeout

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.terminated = True
        self.alive = False

    def kill(self):
        self.alive = False


class _FailingContext:
    def __init__(self, fail_at: int):
        self.fail_at = fail_at
        self.processes: list[_CountingProcess] = []

    def Pipe(self, duplex=False):
        assert duplex is False
        return _ClosedConnection(), _ClosedConnection()

    def Process(self, **kwargs):
        assert kwargs["daemon"] is True
        if len(self.processes) == self.fail_at:
            raise OSError("cannot start a render process")
        self.processes.append(_CountingProcess())
        return self.processes[-1]


def test_a_batch_that_cannot_start_ends_the_renders_it_already_started(
    tmp_path, monkeypatch
):
    """The third start fails, and the two processes already running are ended."""
    context = _FailingContext(fail_at=2)
    monkeypatch.setattr(run_render.mp, "get_context", lambda method: context)
    requests = [
        run_render.RenderRequest(
            tmp_path / f"shape{index}.step",
            _projection_paths(tmp_path),
            _paths(tmp_path),
        )
        for index in range(3)
    ]

    with pytest.raises(OSError, match="cannot start a render process"):
        StepRenderer(timeout_s=60.0, max_workers=4).render_many(requests)

    assert [process.terminated for process in context.processes] == [True, True]
