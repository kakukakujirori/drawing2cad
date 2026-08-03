"""Render a verified STEP to explicitly assigned DXF and PNG paths.

OCC HLR and VTK can hang in native code, so rendering runs in a fresh
``spawn`` process.  The caller owns artifact naming and directory layout; this
module only runs the render stages and supervises that process.
"""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from enum import Enum
from multiprocessing.connection import Connection
from pathlib import Path

from zeroshot.pipeline.verification.render._render3d import generate_render3d
from zeroshot.pipeline.verification.render.arrange import arrange
from zeroshot.pipeline.verification.render.constants import (
    Render3dPaths,
    TechdrawPaths,
)
from zeroshot.pipeline.verification.render.export_dxf import export_dxf
from zeroshot.pipeline.verification.render.project import load_shape, project_views


class RenderStatus(Enum):
    OK = "OK"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"


@dataclass(frozen=True)
class RenderReport:
    """What each requested output became: a path on success, a reason on failure."""

    status: RenderStatus
    techdraw_paths: TechdrawPaths
    render3d_paths: Render3dPaths
    techdraw_errors: Mapping[str, str] = field(default_factory=dict)
    render3d_errors: Mapping[str, str] = field(default_factory=dict)


def _render_techdraw(
    step_path: Path,
    techdraw_paths: TechdrawPaths,
) -> tuple[TechdrawPaths, dict[str, str]]:
    """Render the three orthographic views to DXF."""
    errors: dict[str, str] = {}

    # TODO: currently only DXF is supported
    if techdraw_paths.svg is not None or techdraw_paths.pdf is not None:
        raise NotImplementedError(
            "techdraw_paths.svg and techdraw_paths.pdf must be None"
        )
    if techdraw_paths.dxf is not None:
        export_dxf(techdraw_paths.dxf, arrange(project_views(load_shape(step_path))))

    if techdraw_paths.dxf and not techdraw_paths.dxf.is_file():
        techdraw_paths = replace(techdraw_paths, dxf=None)
        errors["dxf"] = "techdraw renderer did not create a DXF file"
    return techdraw_paths, errors


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


def _get_render_num(paths: TechdrawPaths | Render3dPaths) -> int:
    return sum(
        getattr(paths, path_field.name) is not None for path_field in fields(paths)
    )


def _render_once(
    input_step_path: Path,
    output_techdraw_paths: TechdrawPaths,
    output_render3d_paths: Render3dPaths,
) -> RenderReport:
    """Run each component once and retain paths for successful outputs only."""
    requested_render_num = _get_render_num(output_techdraw_paths) + _get_render_num(
        output_render3d_paths
    )

    # techdraw
    try:
        output_techdraw_paths, techdraw_errors = _render_techdraw(
            input_step_path, output_techdraw_paths
        )
    except Exception as error:  # noqa: BLE001
        for path_field in fields(output_techdraw_paths):
            path = getattr(output_techdraw_paths, path_field.name)
            if path is not None:
                path.unlink(missing_ok=True)  # delete a possibly broken DXF file
        techdraw_errors = {
            path_field.name: _error_text(error)
            for path_field in fields(TechdrawPaths)
            if getattr(output_techdraw_paths, path_field.name) is not None
        }
        output_techdraw_paths = TechdrawPaths()

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
    completed_render_num = _get_render_num(output_techdraw_paths) + _get_render_num(
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
        techdraw_paths=output_techdraw_paths,
        render3d_paths=output_render3d_paths,
        techdraw_errors=techdraw_errors,
        render3d_errors=render3d_errors,
    )


def _worker(
    input_step_path: Path,
    output_techdraw_paths: TechdrawPaths,
    output_render3d_paths: Render3dPaths,
    connection: Connection,
) -> None:
    try:
        connection.send(
            _render_once(
                input_step_path,
                output_techdraw_paths,
                output_render3d_paths,
            )
        )
    finally:
        connection.close()


class StepRenderer:
    def __init__(self, timeout_s: float = 120.0) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.timeout_s = timeout_s

    def render(
        self,
        input_step_path: Path,
        output_techdraw_paths: TechdrawPaths,
        output_render3d_paths: Render3dPaths,
    ) -> RenderReport:
        """Render one STEP while supervising native-code hangs."""
        context = mp.get_context("spawn")
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(
            target=_worker,
            args=(
                input_step_path,
                output_techdraw_paths,
                output_render3d_paths,
                sender,
            ),
            daemon=True,
        )
        process.start()
        sender.close()

        process.join(self.timeout_s)
        timed_out = process.is_alive()
        if timed_out:
            process.terminate()
            process.join(5.0)
            if process.is_alive():
                process.kill()
                process.join()

        try:
            report = receiver.recv() if not timed_out and receiver.poll() else None
        except EOFError:
            report = None
        finally:
            receiver.close()

        if isinstance(report, RenderReport):
            return report

        # --- below is error handling --- #

        for path in (
            *(
                getattr(output_techdraw_paths, path_field.name)
                for path_field in fields(output_techdraw_paths)
            ),
            *(
                getattr(output_render3d_paths, path_field.name)
                for path_field in fields(output_render3d_paths)
            ),
        ):
            if path is not None:
                path.unlink(missing_ok=True)

        if timed_out:
            message = f"render timed out after {self.timeout_s:g}s"
            status = RenderStatus.TIMEOUT
        else:
            message = (
                f"render process exited without a report (exitcode={process.exitcode})"
            )
            status = RenderStatus.FAILED
        return RenderReport(
            status=status,
            techdraw_paths=TechdrawPaths(),
            render3d_paths=Render3dPaths(),
            techdraw_errors=dict.fromkeys(output_techdraw_paths.as_mapping(), message),
            render3d_errors=dict.fromkeys(output_render3d_paths.as_mapping(), message),
        )
