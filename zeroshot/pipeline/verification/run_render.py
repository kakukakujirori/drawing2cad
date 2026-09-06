"""Render verified STEPs to explicitly assigned DXF and PNG paths.

Each orthographic view is drawn to a DXF of its own, at 1:1 and read from its
own corner: nothing is composed onto a sheet.

OCC HLR and VTK can hang in native code, so every render runs in a fresh
``spawn`` process.  The caller owns artifact naming and directory layout; this
module only runs the render stages and supervises those processes.

A build reports one drawing per planned operation, so the renders come in
batches of twenty-odd; they are independent, and ``render_many`` runs them
several at a time.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from enum import Enum
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

from zeroshot.pipeline.messages.contracts.drawings import View
from zeroshot.pipeline.verification.render._render3d import generate_render3d
from zeroshot.pipeline.verification.render.constants import (
    ProjectionPaths,
    Render3dPaths,
)
from zeroshot.pipeline.verification.render.export_dxf import export_to_png, export_view
from zeroshot.pipeline.verification.render.project import (
    load_shape,
    project_views,
    to_own_corner,
)


class RenderStatus(Enum):
    OK = "OK"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"


@dataclass(frozen=True)
class RenderReport:
    """What each requested output became: a path on success, a reason on failure."""

    status: RenderStatus
    projection_paths: ProjectionPaths
    render3d_paths: Render3dPaths
    projection_errors: Mapping[str, str] = field(default_factory=dict)
    render3d_errors: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RenderRequest:
    """One STEP and the paths its views are to be written to."""

    step_path: Path
    projection_paths: ProjectionPaths
    render3d_paths: Render3dPaths


def _render_projections(
    step_path: Path,
    projection_paths: ProjectionPaths,
) -> tuple[ProjectionPaths, dict[str, str]]:
    """Draw each requested orthographic view into a DXF of its own."""
    errors: dict[str, str] = {}
    requested = projection_paths.as_mapping()
    if requested:
        wanted = [View(name) for name in requested]
        projections = project_views(load_shape(step_path), wanted)
        for view in wanted:
            path = requested[view.value]
            export_view(path, to_own_corner(projections[view], view.value), view.value)
            export_to_png(path)

    for view, path in requested.items():
        if not path.is_file():
            projection_paths = replace(projection_paths, **{view: None})
            errors[view] = f"projection renderer did not create {view}.dxf"
    return projection_paths, errors


def _render_3d(
    step_path: Path,
    render3d_paths: Render3dPaths,
) -> tuple[Render3dPaths, dict[str, str]]:
    """Render all perspective styles and interpret their component results."""
    errors: dict[str, str] = {}

    # NOTE: generate_render3d requires all paths for now
    assert all(
        getattr(render3d_paths, path_field.name) is not None
        for path_field in fields(render3d_paths)
    ), "generate_render3d currently requires all output paths"

    results = generate_render3d(step_path, render3d_paths)
    successful_styles = set(results.get("rendered", ()))
    for path_field in fields(render3d_paths):
        style = path_field.name
        path = getattr(render3d_paths, style)
        if path is None:
            continue
        if style not in successful_styles or not path.is_file():
            path.unlink(missing_ok=True)
            render3d_paths = replace(render3d_paths, **{style: None})
            errors[style] = str(
                results.get("errors", {}).get(style) or f"{style} was not generated"
            )
    return render3d_paths, errors


def _error_text(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def _get_render_num(paths: ProjectionPaths | Render3dPaths) -> int:
    return sum(
        getattr(paths, path_field.name) is not None for path_field in fields(paths)
    )


def _render_once(
    input_step_path: Path,
    output_projection_paths: ProjectionPaths,
    output_render3d_paths: Render3dPaths,
) -> RenderReport:
    """Run each component once and retain paths for successful outputs only."""
    requested_render_num = _get_render_num(output_projection_paths) + _get_render_num(
        output_render3d_paths
    )

    # projections
    try:
        output_projection_paths, projection_errors = _render_projections(
            input_step_path, output_projection_paths
        )
    except Exception as error:  # noqa: BLE001
        for path_field in fields(output_projection_paths):
            path = getattr(output_projection_paths, path_field.name)
            if path is not None:
                path.unlink(missing_ok=True)  # delete a possibly broken DXF file
        projection_errors = {
            path_field.name: _error_text(error)
            for path_field in fields(ProjectionPaths)
            if getattr(output_projection_paths, path_field.name) is not None
        }
        output_projection_paths = ProjectionPaths()

    # render3d
    try:
        output_render3d_paths, render3d_errors = _render_3d(
            input_step_path, output_render3d_paths
        )
    except Exception as error:  # noqa: BLE001
        for path_field in fields(output_render3d_paths):
            path = getattr(output_render3d_paths, path_field.name)
            if path is not None:
                path.unlink(missing_ok=True)  # delete a possibly broken PNG file
        render3d_errors = {
            path_field.name: _error_text(error)
            for path_field in fields(Render3dPaths)
            if getattr(output_render3d_paths, path_field.name) is not None
        }
        output_render3d_paths = Render3dPaths()

    # summary
    completed_render_num = _get_render_num(output_projection_paths) + _get_render_num(
        output_render3d_paths
    )

    if requested_render_num == completed_render_num:
        status = RenderStatus.OK
    elif completed_render_num > 0:
        status = RenderStatus.PARTIAL
    else:
        status = RenderStatus.FAILED

    return RenderReport(
        status=status,
        projection_paths=output_projection_paths,
        render3d_paths=output_render3d_paths,
        projection_errors=projection_errors,
        render3d_errors=render3d_errors,
    )


def _worker(
    input_step_path: Path,
    output_projection_paths: ProjectionPaths,
    output_render3d_paths: Render3dPaths,
    connection: Connection,
) -> None:
    try:
        connection.send(
            _render_once(
                input_step_path,
                output_projection_paths,
                output_render3d_paths,
            )
        )
    finally:
        connection.close()


@dataclass
class _Attempt:
    """A render in flight, and when it stops being worth waiting for."""

    index: int
    request: RenderRequest
    process: Any
    receiver: Connection
    deadline: float


# Short next to a render that takes seconds, so a worker that finished is
# collected -- and its slot handed to the next request -- without a delay
# worth measuring.
_POLL_INTERVAL_S = 0.05


class StepRenderer:
    def __init__(self, timeout_s: float = 120.0, max_workers: int = 8) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        self.timeout_s = timeout_s
        self.max_workers = max_workers

    def render(
        self,
        input_step_path: Path,
        output_projection_paths: ProjectionPaths,
        output_render3d_paths: Render3dPaths,
    ) -> RenderReport:
        """Render one STEP while supervising native-code hangs."""
        (report,) = self.render_many(
            [
                RenderRequest(
                    input_step_path,
                    output_projection_paths,
                    output_render3d_paths,
                )
            ]
        )
        return report

    def render_many(self, requests: Sequence[RenderRequest]) -> list[RenderReport]:
        """Render every request, at most `max_workers` at a time, in request order.

        One process per request rather than a pool of reusable workers: a
        render that hangs in native code is ended by killing the process it
        holds, and a pool that lost a worker that way could not carry on.

        Each request owns its own timeout, counted from the moment it starts
        rather than from the batch, so a request that waited for a slot is
        given the same time to run as the one that started first.
        """
        # A build that failed early leaves nothing to draw, which is the
        # common case; it costs no process machinery at all.
        if not requests:
            return []

        context = mp.get_context("spawn")
        reports: dict[int, RenderReport] = {}
        queued = deque(range(len(requests)))
        running: list[_Attempt] = []

        try:
            while queued or running:
                while queued and len(running) < self.max_workers:
                    index = queued.popleft()
                    running.append(self._start(context, index, requests[index]))

                waiting: list[_Attempt] = []
                for attempt in running:
                    if attempt.receiver.poll():
                        reports[attempt.index] = self._collect(attempt)
                    elif time.monotonic() >= attempt.deadline:
                        reports[attempt.index] = self._give_up(attempt)
                    else:
                        waiting.append(attempt)
                collected = len(running) - len(waiting)
                running = waiting
                if not collected and running:
                    time.sleep(_POLL_INTERVAL_S)
        finally:
            # A batch holds several renders at once, each of them a process of
            # its own: leaving on any other path than the loop's own end has to
            # end them, or they are left running and holding their memory.
            for abandoned in running:
                abandoned.receiver.close()
                _end(abandoned.process, terminate=True)

        # Indexed rather than appended, so a request whose report went missing
        # raises here instead of shifting every later report onto the wrong one.
        return [reports[index] for index in range(len(requests))]

    def _start(
        self,
        context: Any,
        index: int,
        request: RenderRequest,
    ) -> _Attempt:
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(
            target=_worker,
            args=(
                request.step_path,
                request.projection_paths,
                request.render3d_paths,
                sender,
            ),
            daemon=True,
        )
        process.start()
        sender.close()
        return _Attempt(
            index=index,
            request=request,
            process=process,
            receiver=receiver,
            deadline=time.monotonic() + self.timeout_s,
        )

    def _collect(self, attempt: _Attempt) -> RenderReport:
        """Read what a worker sent, and reap it."""
        try:
            report = attempt.receiver.recv()
        except EOFError:
            report = None
        finally:
            attempt.receiver.close()
        _end(attempt.process)

        if isinstance(report, RenderReport):
            return report
        return _failure_report(
            attempt.request,
            RenderStatus.FAILED,
            "render process exited without a report "
            f"(exitcode={attempt.process.exitcode})",
        )

    def _give_up(self, attempt: _Attempt) -> RenderReport:
        """End a render that ran out of time and say so."""
        _end(attempt.process, terminate=True)
        attempt.receiver.close()
        return _failure_report(
            attempt.request,
            RenderStatus.TIMEOUT,
            f"render timed out after {self.timeout_s:g}s",
        )


def _end(process: Any, terminate: bool = False) -> None:
    if terminate:
        process.terminate()
        process.join(5.0)
    else:
        process.join(5.0)
    if process.is_alive():
        process.kill()
        process.join()


def _failure_report(
    request: RenderRequest,
    status: RenderStatus,
    message: str,
) -> RenderReport:
    """Discard whatever a failed render left behind, and name it as the reason."""
    for paths in (request.projection_paths, request.render3d_paths):
        for path in paths.as_mapping().values():
            path.unlink(missing_ok=True)
    return RenderReport(
        status=status,
        projection_paths=ProjectionPaths(),
        render3d_paths=Render3dPaths(),
        projection_errors=dict.fromkeys(request.projection_paths.as_mapping(), message),
        render3d_errors=dict.fromkeys(request.render3d_paths.as_mapping(), message),
    )
