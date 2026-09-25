"""What one render_step call draws, where it writes, and what it refuses."""

from pathlib import Path, PurePosixPath

import cadquery as cq
import ezdxf
import pytest

from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.stages.interpretation.contracts import View
from zeroshot.pipeline.tools.errors import ToolFeedbackError
from zeroshot.pipeline.tools.render_step import create_render_step_tool
from zeroshot.pipeline.verification.render.orthographic import ViewFrames
from zeroshot.pipeline.verification.run_render import StepRenderer

# Three distinct lengths, so a view's extent says which axes it was drawn in.
BOX_X, BOX_Y, BOX_Z = 30.0, 20.0, 10.0


@pytest.fixture
def workdir(tmp_path):
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    box = cq.Workplane().box(BOX_X, BOX_Y, BOX_Z)
    cq.exporters.export(box, str(tmp_path / "tmp" / "trial.step"))
    return workdir


def _render(render_step, views, step_path="/tmp/trial.step"):
    return render_step.invoke({"step_path": step_path, "views": views})


def _files_outside_renders(root: Path) -> dict[Path, bytes]:
    return {
        path: path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).parts[0] != "renders"
    }


def _extent(workdir: SandboxWorkdir, dxf: str) -> tuple[float, float]:
    header = ezdxf.readfile(workdir.sandbox_to_host_path(dxf)).header
    (x0, y0, _), (x1, y1, _) = header["$EXTMIN"], header["$EXTMAX"]
    return x1 - x0, y1 - y0


def test_each_call_draws_the_views_it_names_into_a_new_directory(workdir):
    host = workdir.host_bind_dir
    (host / "model.py").write_text("result = None\n")
    untouched = _files_outside_renders(host)
    render_step = create_render_step_tool(workdir, StepRenderer(), dict)

    first = _render(render_step, ["front", "top"])
    second = _render(render_step, ["front"])

    assert first["status"] == "OK"
    assert first["views"]["top"]["png"] == "/work/renders/000/top.png"
    assert first["views"]["top"]["dxf"] == "/work/renders/000/top.dxf"
    assert second["views"]["front"]["png"] == "/work/renders/001/front.png"
    assert {path.name for path in (host / "renders" / "000").iterdir()} == {
        "shape.step",
        "front.dxf",
        "front.png",
        "top.dxf",
        "top.png",
    }
    assert _files_outside_renders(host) == untouched


def test_a_view_the_drawing_shows_is_drawn_in_its_current_axes(workdir):
    # Beside the top view, a right view shares the top view's +Y.
    drawing: ViewFrames = {View.RIGHT: ("-z", "+y")}
    render_step = create_render_step_tool(workdir, StepRenderer(), lambda: drawing)

    turned = _render(render_step, ["right", "left"])["views"]
    drawing = {}  # a later round's drawing shows no right view
    standard = _render(render_step, ["right"])["views"]

    assert (turned["right"]["u_axis"], turned["right"]["v_axis"]) == ("-z", "+y")
    assert _extent(workdir, turned["right"]["dxf"]) == pytest.approx(
        (BOX_Z, BOX_Y), abs=1e-3
    )
    assert (turned["left"]["u_axis"], turned["left"]["v_axis"]) == ("-y", "+z")
    assert (standard["right"]["u_axis"], standard["right"]["v_axis"]) == ("+y", "+z")
    assert _extent(workdir, standard["right"]["dxf"]) == pytest.approx(
        (BOX_Y, BOX_Z), abs=1e-3
    )


@pytest.mark.parametrize(
    "step_path",
    [
        "/etc/hostname",
        "/work/../trial.step",
        "/work/absent.step",
        "/work/tmp",
        "/work/link.step",
    ],
    ids=["outside", "escaping", "missing", "directory", "symlink_outside"],
)
def test_a_path_that_names_no_workspace_file_is_refused(
    workdir, tmp_path_factory, step_path
):
    outside = tmp_path_factory.mktemp("outside") / "part.step"
    outside.write_text("ISO-10303-21;")
    (workdir.host_bind_dir / "link.step").symlink_to(outside)
    render_step = create_render_step_tool(workdir, StepRenderer(), dict)

    with pytest.raises(ToolFeedbackError, match="Cannot read STEP file"):
        _render(render_step, ["front"], step_path)
    assert not any((workdir.host_bind_dir / "renders").iterdir())


def test_a_render_that_fails_is_reported_for_each_view(workdir):
    (workdir.host_bind_dir / "junk.step").write_text("not a STEP file")
    render_step = create_render_step_tool(workdir, StepRenderer(), dict)

    result = _render(render_step, ["front", "top"], "/work/junk.step")

    assert result["status"] == "FAILED"
    assert set(result["views"]) == {"front", "top"}
    for view in result["views"].values():
        assert "png" not in view
        assert "could not be loaded" in view["error"]


def test_a_render_that_times_out_leaves_the_next_call_working(workdir):
    renderer = StepRenderer(timeout_s=0.01)
    render_step = create_render_step_tool(workdir, renderer, dict)

    timed_out = _render(render_step, ["front"])
    renderer.timeout_s = 60.0
    retried = _render(render_step, ["front"])

    assert timed_out["status"] == "TIMEOUT"
    assert timed_out["views"]["front"]["error"] == "render timed out after 0.01s"
    assert retried["status"] == "OK"


def test_the_model_cannot_write_where_renders_go(workdir):
    create_render_step_tool(workdir, StepRenderer(), dict)

    assert PurePosixPath("renders") in workdir.read_only_subdirs


def test_a_renders_directory_that_leads_elsewhere_is_refused(
    tmp_path, tmp_path_factory
):
    (tmp_path / "renders").symlink_to(tmp_path_factory.mktemp("elsewhere"))

    with pytest.raises(ValueError, match="symlink"):
        create_render_step_tool(
            SandboxWorkdir(host_bind_dir=tmp_path), StepRenderer(), dict
        )
