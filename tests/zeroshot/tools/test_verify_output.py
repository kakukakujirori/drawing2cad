import json
from dataclasses import replace
from pathlib import Path, PurePosixPath

import pytest
from langchain_core.tools import BaseTool

from zeroshot.pipeline.messages import MessageBuilder
from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.tools.verify_output import (
    VerifyOutputResult,
    create_verify_output_tool,
)
from zeroshot.pipeline.verification.render.constants import (
    Render3dPaths,
    TechdrawPaths,
)
from zeroshot.pipeline.verification.run_cadquery import (
    CadQueryExecutionReport,
    ExecutionStatus,
)
from zeroshot.pipeline.verification.run_render import RenderReport, RenderStatus

RENDER3D_STYLES = (
    "hlg_perspective",
    "transparent_shaded_edges_perspective",
    "hlg_translucent_faces_perspective",
)

VALID_SOURCE = """\
import cadquery as cq

result = cq.Workplane("XY").box(10, 20, 30)
"""


class StubCadQueryExecutor:
    def __init__(self, report: CadQueryExecutionReport) -> None:
        self.report = report
        self.calls: list[tuple[Path, Path | None]] = []

    def execute(
        self,
        model_path: Path,
        output_step_path: Path | None = None,
    ) -> CadQueryExecutionReport:
        self.calls.append((model_path, output_step_path))
        # A verified run leaves the STEP behind, which is what gets rendered.
        if (
            output_step_path is not None
            and self.report.status is ExecutionStatus.VERIFIED
        ):
            output_step_path.write_text("ISO-10303-21;\nEND-ISO-10303-21;\n")
        return self.report


def _execution_report(
    *,
    source: str | None = VALID_SOURCE,
    status: ExecutionStatus = ExecutionStatus.VERIFIED,
    returncode: int | None = 0,
    stdout: str = "construction log",
    stderr: str = "",
    executor_error: str | None = None,
) -> CadQueryExecutionReport:
    return CadQueryExecutionReport(
        source=source,
        status=status,
        executor_error=executor_error,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _create_tool(
    executor: StubCadQueryExecutor,
    workdir: SandboxWorkdir,
    *,
    renderer: object | None = None,  # defaults to a StubRenderer
    message_builder: MessageBuilder | None = None,
    source_filename: str = "model.py",
    output_dirname: PurePosixPath = PurePosixPath("attempts"),
    serialize_output: bool = True,
) -> BaseTool:
    return create_verify_output_tool(
        executor,  # type: ignore[arg-type]
        workdir,
        renderer=renderer or StubRenderer(),  # type: ignore[arg-type]
        message_builder=message_builder,
        source_filename=source_filename,
        output_dirname=output_dirname,
        serialize_output=serialize_output,
    )


class StubRenderer:
    """A ``StepRenderer`` that writes placeholder artifacts instead of rendering.

    ``skip_styles`` drops perspective styles the way a partial render does, so
    the manifest can be checked for reporting only what actually exists.
    """

    def __init__(self, *, skip_styles: tuple[str, ...] = ()) -> None:
        self.skip_styles = skip_styles
        self.calls: list[tuple[Path, TechdrawPaths, Render3dPaths]] = []

    def render(
        self,
        input_step_path: Path,
        output_techdraw_paths: TechdrawPaths,
        output_render3d_paths: Render3dPaths,
    ) -> RenderReport:
        self.calls.append(
            (input_step_path, output_techdraw_paths, output_render3d_paths)
        )
        assert output_techdraw_paths.dxf is not None
        output_techdraw_paths.dxf.write_text("0\nEOF\n", encoding="utf-8")

        errors: dict[str, str] = {}
        for style in self.skip_styles:
            output_render3d_paths = replace(output_render3d_paths, **{style: None})
            errors[style] = f"RuntimeError: {style} failed"
        for style, path in output_render3d_paths.as_mapping().items():
            assert style not in self.skip_styles
            path.write_bytes(b"PNG")

        return RenderReport(
            status=RenderStatus.OK if not errors else RenderStatus.PARTIAL,
            techdraw_paths=output_techdraw_paths,
            render3d_paths=output_render3d_paths,
            render3d_errors=errors,
        )


def _message_builder(feedback_render3d: str = "path") -> MessageBuilder:
    return MessageBuilder(
        access_render3d="none",
        access_render3d_styles=(),
        feedback_render3d=feedback_render3d,  # type: ignore[arg-type]
        feedback_render3d_styles=RENDER3D_STYLES,
    )


def _text(result: object) -> str:
    """Concatenate the text the model would read out of the tool result."""
    assert isinstance(result, list)
    return "\n".join(b["text"] for b in result if b["type"] == "text")


def _report_json(result: object) -> dict:
    """The verification report the tool result leads with."""
    text = _text(result)
    start = text.index("{")
    return json.loads(text[start : text.rindex("}") + 1])


def test_tool_schema_and_factory_prepare_pipeline_managed_output(
    tmp_path: Path,
) -> None:
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)

    verify_output = _create_tool(
        executor,
        workdir,
        source_filename="candidate.py",
    )

    assert verify_output.name == "verify_output"
    assert verify_output.get_input_jsonschema()["properties"] == {}
    assert "/work/candidate.py" in verify_output.description
    assert (tmp_path / "attempts").is_dir()
    assert workdir.read_only_subdirs == [PurePosixPath("attempts")]


def test_tool_delegates_paths_and_returns_json_safe_mapping(tmp_path: Path) -> None:
    executor = StubCadQueryExecutor(
        _execution_report(
            returncode=0,
            stdout="construction log",
            stderr="construction warning",
        )
    )
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")
    verify_output = _create_tool(executor, workdir)

    result = verify_output.invoke({})

    assert executor.calls == [
        (
            tmp_path / "model.py",
            tmp_path / "attempts" / "000" / "output.step",
        )
    ]
    assert _report_json(result) == {
        "verification_id": "000",
        "status": "VERIFIED",
        "returncode": 0,
        "stdout": "construction log",
        "stderr": "construction warning",
        "executor_error": None,
    }
    report = _report_json(result)
    assert isinstance(report["returncode"], int)
    assert "source" not in report
    assert (tmp_path / "attempts" / "000" / "model.py").read_text(
        encoding="utf-8"
    ) == VALID_SOURCE


def test_tool_preserves_failed_attempt_and_execution_report(tmp_path: Path) -> None:
    executor = StubCadQueryExecutor(
        _execution_report(
            status=ExecutionStatus.FAILED,
            returncode=1,
            stdout="partial output",
            stderr="execution failed",
            executor_error="output.step was not generated",
        )
    )
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")
    verify_output = _create_tool(executor, workdir)

    result = verify_output.invoke({})

    assert _report_json(result) == {
        "verification_id": "000",
        "status": "FAILED",
        "returncode": 1,
        "stdout": "partial output",
        "stderr": "execution failed",
        "executor_error": "output.step was not generated",
    }
    attempt_dir = tmp_path / "attempts" / "000"
    assert (attempt_dir / "model.py").read_text(encoding="utf-8") == VALID_SOURCE
    assert not (attempt_dir / "output.step").exists()


def test_tool_assigns_incrementing_verification_ids(tmp_path: Path) -> None:
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")
    verify_output = _create_tool(executor, workdir)

    first = verify_output.invoke({})
    second = verify_output.invoke({})

    assert _report_json(first)["verification_id"] == "000"
    assert _report_json(second)["verification_id"] == "001"
    assert (tmp_path / "attempts" / "000").is_dir()
    assert (tmp_path / "attempts" / "001").is_dir()


def test_tool_rejects_missing_source_without_issuing_id(tmp_path: Path) -> None:
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    verify_output = _create_tool(executor, workdir)

    result = verify_output.invoke({})

    assert _report_json(result) == {
        "verification_id": None,
        "status": "REJECTED",
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "executor_error": "model.py was not found",
    }
    assert executor.calls == []
    assert list((tmp_path / "attempts").iterdir()) == []


def test_tool_rejects_source_symlink_without_issuing_id(tmp_path: Path) -> None:
    real_source = tmp_path / "real-model.py"
    real_source.write_text(VALID_SOURCE, encoding="utf-8")
    (tmp_path / "model.py").symlink_to(real_source)
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    verify_output = _create_tool(executor, workdir)

    result = verify_output.invoke({})

    assert _report_json(result)["verification_id"] is None
    assert _report_json(result)["status"] == "REJECTED"
    assert _report_json(result)["executor_error"] == "model.py must not be a symlink"
    assert executor.calls == []
    assert list((tmp_path / "attempts").iterdir()) == []


def test_tool_preserves_executor_rejection_without_source_snapshot(
    tmp_path: Path,
) -> None:
    executor = StubCadQueryExecutor(
        _execution_report(
            source=None,
            status=ExecutionStatus.REJECTED,
            returncode=None,
            stdout="",
            stderr="",
            executor_error="model.py must be valid UTF-8",
        )
    )
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    (tmp_path / "model.py").write_bytes(b"\xff")
    verify_output = _create_tool(executor, workdir)

    result = verify_output.invoke({})

    assert _report_json(result)["verification_id"] == "000"
    assert _report_json(result)["status"] == "REJECTED"
    assert _report_json(result)["executor_error"] == "model.py must be valid UTF-8"
    assert not (tmp_path / "attempts" / "000" / "model.py").exists()
    assert len(executor.calls) == 1


def test_unserialized_tool_result_preserves_source(tmp_path: Path) -> None:
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")
    verify_output = _create_tool(executor, workdir, serialize_output=False)

    result = verify_output.invoke({})

    assert result == VerifyOutputResult(
        verification_id="000",
        status="VERIFIED",
        source=VALID_SOURCE,
        returncode=0,
        stdout="construction log",
    )


@pytest.mark.parametrize(
    "source_filename",
    [
        "",
        ".",
        "..",
        "./model.py",
        "../model.py",
        "nested/model.py",
        "/work/model.py",
    ],
)
def test_tool_rejects_source_filename_outside_workdir_root(
    tmp_path: Path,
    source_filename: str,
) -> None:
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)

    with pytest.raises(
        ValueError,
        match="source_filename must be a filename in the workdir root",
    ):
        _create_tool(
            executor,
            workdir,
            source_filename=source_filename,
        )


@pytest.mark.parametrize(
    "output_dirname",
    [
        PurePosixPath(""),
        PurePosixPath("."),
        PurePosixPath(".."),
        PurePosixPath("../attempts"),
        PurePosixPath("nested/attempts"),
        PurePosixPath("/work/attempts"),
    ],
)
def test_tool_rejects_output_dirname_outside_workdir_root(
    tmp_path: Path,
    output_dirname: PurePosixPath,
) -> None:
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)

    with pytest.raises(ValueError, match="output_dirname must be a directory basename"):
        _create_tool(executor, workdir, output_dirname=output_dirname)


def test_tool_rejects_symlink_output_directory(tmp_path: Path) -> None:
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    (tmp_path / "attempts").symlink_to(outside_dir, target_is_directory=True)
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)

    with pytest.raises(ValueError, match="output directory must not be a symlink"):
        _create_tool(executor, workdir)


def test_verified_output_is_rendered_and_offered_to_the_model(tmp_path: Path) -> None:
    """A verified STEP must yield a drawing plus one render per style, and the
    tool result must point the model at all of them in sandbox coordinates."""
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")
    renderer = StubRenderer()
    verify_output = _create_tool(
        executor, workdir, renderer=renderer, message_builder=_message_builder()
    )

    text = _text(verify_output.invoke({}))

    verification_dir = tmp_path / "attempts" / "000"
    (rendered_step, _, _) = renderer.calls[0]
    assert rendered_step == verification_dir / "output.step"
    sandbox_dir = f"{workdir.sandbox_bind_dir}/attempts/000"
    assert f"{sandbox_dir}/techdraw/dxf/output.dxf" in text
    for style in RENDER3D_STYLES:
        assert f"{sandbox_dir}/render_3d/{style}/output.png" in text


def test_rendered_artifacts_stay_inside_the_verification_directory(
    tmp_path: Path,
) -> None:
    """Nothing may be written where another attempt, or the agent, could see it."""
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")
    verify_output = _create_tool(
        executor, workdir, renderer=StubRenderer(), message_builder=_message_builder()
    )

    verify_output.invoke({})

    verification_dir = tmp_path / "attempts" / "000"
    written = {p for p in tmp_path.rglob("*") if p.is_file()}
    assert written == {
        tmp_path / "model.py",
        verification_dir / "model.py",
        verification_dir / "output.step",
        verification_dir / "techdraw" / "dxf" / "output.dxf",
        *(
            verification_dir / "render_3d" / style / "output.png"
            for style in RENDER3D_STYLES
        ),
    }


def test_failed_verification_renders_nothing_and_reports_only_the_error(
    tmp_path: Path,
) -> None:
    """Without a valid STEP there is nothing to draw, so the renderer never runs."""
    executor = StubCadQueryExecutor(
        _execution_report(status=ExecutionStatus.FAILED, returncode=1)
    )
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")
    renderer = StubRenderer()
    verify_output = _create_tool(
        executor, workdir, renderer=renderer, message_builder=_message_builder()
    )

    result = verify_output.invoke({})

    assert renderer.calls == []
    assert _report_json(result)["status"] == "FAILED"
    assert "output.dxf" not in _text(result)


def test_partial_render_offers_only_existing_styles_and_explains_the_rest(
    tmp_path: Path,
) -> None:
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")
    renderer = StubRenderer(skip_styles=("transparent_shaded_edges_perspective",))
    verify_output = _create_tool(
        executor, workdir, renderer=renderer, message_builder=_message_builder()
    )

    result = verify_output.invoke({})
    text = _text(result)

    assert "transparent_shaded_edges_perspective/output.png" not in text
    assert "hlg_perspective/output.png" in text
    # The reason belongs where the render would have been, not in the report.
    assert "render_errors" not in _report_json(result)
    assert (
        "- transparent_shaded_edges_perspective: unavailable "
        "(RuntimeError: transparent_shaded_edges_perspective failed)"
    ) in text


def test_tool_result_carries_paths_but_never_the_drawing_itself(
    tmp_path: Path,
) -> None:
    """The model is handed a path to open deliberately, not the DXF body."""
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")
    verify_output = _create_tool(
        executor, workdir, renderer=StubRenderer(), message_builder=_message_builder()
    )

    text = _text(verify_output.invoke({}))

    dxf_body = (
        tmp_path / "attempts" / "000" / "techdraw" / "dxf" / "output.dxf"
    ).read_text(encoding="utf-8")
    assert "output.dxf" in text
    assert dxf_body not in text


def test_images_are_embedded_only_when_the_builder_asks_for_them(
    tmp_path: Path,
) -> None:
    """``feedback_render3d`` is MessageBuilder's decision, and the tool obeys it."""
    executor = StubCadQueryExecutor(_execution_report())
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")

    def block_types(mode: str) -> list[str]:
        workdir = SandboxWorkdir(host_bind_dir=tmp_path)
        result = _create_tool(
            executor,
            workdir,
            renderer=StubRenderer(),
            message_builder=_message_builder(mode),
        ).invoke({})
        assert isinstance(result, list)
        return [block["type"] for block in result]

    assert "image" not in block_types("path")
    assert "image" in block_types("image")
    assert "image" not in block_types("none")


def test_without_a_message_builder_the_model_sees_only_the_report(
    tmp_path: Path,
) -> None:
    """Rendering for later evaluation must not leak artifacts into the context."""
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")
    verify_output = _create_tool(executor, workdir, renderer=StubRenderer())

    result = verify_output.invoke({})

    assert _report_json(result)["status"] == "VERIFIED"
    assert "output.dxf" not in _text(result)
    assert (tmp_path / "attempts" / "000" / "techdraw" / "dxf" / "output.dxf").is_file()


def test_unserialized_result_is_the_report_object_for_graph_state(
    tmp_path: Path,
) -> None:
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")
    verify_output = _create_tool(
        executor,
        workdir,
        renderer=StubRenderer(),
        message_builder=_message_builder(),
        serialize_output=False,
    )

    result = verify_output.invoke({})

    assert isinstance(result, VerifyOutputResult)
    assert result.status == "VERIFIED"
