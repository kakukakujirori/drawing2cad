import shutil
from collections.abc import Callable
from inspect import cleandoc
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal

from langchain_core.tools import BaseTool, tool
from pydantic import Field

from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.stages.interpretation.contracts import View
from zeroshot.pipeline.tools.errors import ToolFeedbackError
from zeroshot.pipeline.verification.render.constants import (
    ProjectionPaths,
    Render3dPaths,
)
from zeroshot.pipeline.verification.render.orthographic import (
    STANDARD_VIEW_FRAMES,
    ViewFrames,
)
from zeroshot.pipeline.verification.run_render import StepRenderer

RENDERS_DIRNAME = PurePosixPath("renders")

ViewName = Literal["front", "back", "top", "bottom", "left", "right"]


def create_render_step_tool(
    workdir: SandboxWorkdir,
    renderer: StepRenderer,
    drawing_frames: Callable[[], ViewFrames],
) -> BaseTool:
    """`drawing_frames` gives the axes of the views the drawing shows, when called."""
    root = workdir.host_bind_dir / RENDERS_DIRNAME
    if root.is_symlink():
        raise ValueError(f"{RENDERS_DIRNAME} must not be a symlink")
    root.mkdir(exist_ok=True)
    # The host writes here, so the model must not be able to redirect it.
    if RENDERS_DIRNAME not in workdir.read_only_subdirs:
        workdir.read_only_subdirs.append(RENDERS_DIRNAME)

    description = cleandoc(
        f"""Draw orthographic views of a STEP file, as verification draws them.

        Use it to look at a trial shape: build it with run_shell, export it to STEP,
        and draw the views that decide your question. Each call writes a DXF and a
        PNG per view to a new directory under
        {workdir.sandbox_bind_dir / RENDERS_DIRNAME}. Open a PNG with load_image.

        A view the drawing shows is drawn in the drawing's axes, and any other view
        in standard axes. Each view reports the model axes that point right (u_axis)
        and up (v_axis); a DXF's x and y are model coordinates along them.

        This only draws. It builds no program and compares nothing with the drawing.
        A failed render says nothing about whether the shape is right.
        """
    )

    @tool("render_step", description=description)
    def render_step(
        step_path: Annotated[str, "A STEP file in the workspace."],
        views: Annotated[list[ViewName], Field(min_length=1), "The views to draw."],
    ) -> dict[str, Any]:
        source = _workspace_file(workdir, step_path)
        frames = {**STANDARD_VIEW_FRAMES, **drawing_frames()}
        chosen = {View(view): frames[View(view)] for view in views}
        directory = _new_directory(root)
        step = directory / "shape.step"
        shutil.copyfile(source, step)

        report = renderer.render(
            step, ProjectionPaths.flat(directory, chosen), Render3dPaths(), chosen
        )

        def sandbox(path: Path) -> str:
            return str(workdir.host_to_sandbox_path(path))

        drawn = report.projection_paths.as_mapping()
        results: dict[str, dict[str, str]] = {}
        for view, (u_axis, v_axis) in chosen.items():
            result: dict[str, str] = {"u_axis": u_axis, "v_axis": v_axis}
            if dxf := drawn.get(view.value):
                result |= {"png": sandbox(dxf.with_suffix(".png")), "dxf": sandbox(dxf)}
            else:
                result["error"] = report.projection_errors[view.value]
            results[view.value] = result
        return {"status": report.status.value, "views": results}

    return render_step


def _workspace_file(workdir: SandboxWorkdir, sandbox_path: str) -> Path:
    try:
        path = workdir.sandbox_to_host_path(sandbox_path).resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        raise ToolFeedbackError(f"Cannot read STEP file: {sandbox_path}") from None
    if not path.is_relative_to(workdir.host_bind_dir.resolve()) or not path.is_file():
        raise ToolFeedbackError(f"Cannot read STEP file: {sandbox_path}")
    return path


def _new_directory(root: Path) -> Path:
    # Probing rather than listing, so two calls at once never take one number.
    number = 0
    while True:
        try:
            (directory := root / f"{number:03d}").mkdir()
            return directory
        except FileExistsError:
            number += 1
