import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path, PurePosixPath

import pytest

from zeroshot.pipeline.messages import ArtifactPresenter
from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.tools.verify_output import create_verify_output_tool
from zeroshot.pipeline.verification.render.constants import (
    Render3dPaths,
    TechdrawPaths,
)
from zeroshot.pipeline.verification.run_cadquery import (
    CadQueryExecutionReport,
    ExecutionStatus,
    IntermediateReturn,
)
from zeroshot.pipeline.verification.run_render import (
    RenderReport,
    RenderRequest,
    RenderStatus,
)
from zeroshot.pipeline.verification.shape_census import ShapeCensus
from zeroshot.pipeline.verification.verify_output import (
    OutputVerifier,
    VerifyOutputResult,
    _census_table,
)

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
    def __init__(
        self,
        report: CadQueryExecutionReport,
        return_names: tuple[str, ...] = (),
    ) -> None:
        self.report = report
        self.return_names = return_names
        self.calls: list[tuple[Path, Path | None]] = []
        self.intermediate_returns_dirs: list[Path | None] = []

    def execute(
        self,
        model_path: Path,
        output_step_path: Path | None = None,
        intermediate_returns_dir: Path | None = None,
    ) -> CadQueryExecutionReport:
        self.calls.append((model_path, output_step_path))
        self.intermediate_returns_dirs.append(intermediate_returns_dir)
        # A verified run leaves the STEP behind, which is what gets rendered.
        if (
            output_step_path is not None
            and self.report.status is ExecutionStatus.VERIFIED
        ):
            output_step_path.write_text("ISO-10303-21;\nEND-ISO-10303-21;\n")
        # Kept whenever the program ran, as the real executor does: a
        # `result` that fails to verify still leaves every ret_xxx behind.
        if intermediate_returns_dir is None or not self.return_names:
            return self.report
        return replace(
            self.report,
            intermediate_returns=tuple(
                self._keep(intermediate_returns_dir, index, name)
                for index, name in enumerate(self.return_names)
            ),
        )

    @staticmethod
    def _keep(returns_dir: Path, index: int, name: str) -> IntermediateReturn:
        step_path = returns_dir / name / "output.step"
        step_path.parent.mkdir(parents=True, exist_ok=True)
        step_path.write_text("ISO-10303-21;\nEND-ISO-10303-21;\n")
        return IntermediateReturn(
            name,
            step_path=step_path,
            census=ShapeCensus(
                1, 100.0 * (index + 1), Counter({"Plane": 6}), Counter({"Line": 12})
            ),
        )


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


def _create_verifier(
    executor: StubCadQueryExecutor,
    workdir: SandboxWorkdir,
    *,
    renderer: object | None = None,  # defaults to a StubRenderer
    artifact_presenter: ArtifactPresenter | None = None,
    source_filename: str = "model.py",
    output_dirname: PurePosixPath = PurePosixPath("attempts"),
) -> OutputVerifier:
    return OutputVerifier(
        executor,  # type: ignore[arg-type]
        workdir,
        renderer=renderer or StubRenderer(),  # type: ignore[arg-type]
        artifact_presenter=artifact_presenter,
        source_filename=source_filename,
        output_dirname=output_dirname,
    )


class StubRenderer:
    """A ``StepRenderer`` that writes placeholder artifacts instead of rendering.

    ``skip_styles`` drops perspective styles the way a partial render does, so
    the manifest can be checked for reporting only what actually exists.
    """

    def __init__(self, *, skip_styles: tuple[str, ...] = ()) -> None:
        self.skip_styles = skip_styles
        self.calls: list[tuple[Path, TechdrawPaths, Render3dPaths]] = []

    def render_many(self, requests: Sequence[RenderRequest]) -> list[RenderReport]:
        return [
            self.render(
                request.step_path,
                request.techdraw_paths,
                request.render3d_paths,
            )
            for request in requests
        ]

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


def _artifact_presenter(feedback_render3d: str = "path") -> ArtifactPresenter:
    return ArtifactPresenter(
        input_render3d_mode="none",
        input_render3d_styles=(),
        feedback_render3d_mode=feedback_render3d,  # type: ignore[arg-type]
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


def test_construction_prepares_an_output_directory_the_sandbox_cannot_write(
    tmp_path: Path,
) -> None:
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)

    verifier = _create_verifier(executor, workdir, source_filename="candidate.py")

    assert verifier.source_path == tmp_path / "candidate.py"
    assert (tmp_path / "attempts").is_dir()
    assert workdir.read_only_subdirs == [PurePosixPath("attempts")]


@pytest.mark.parametrize(
    "source_filename",
    ["", ".", "..", "../model.py", "nested/model.py", "/work/model.py", "model.txt"],
)
def test_rejects_a_source_filename_outside_the_workdir_root_or_not_python(
    tmp_path: Path,
    source_filename: str,
) -> None:
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)

    with pytest.raises(
        ValueError, match="source_filename must be a Python file basename"
    ):
        _create_verifier(executor, workdir, source_filename=source_filename)


def test_the_tool_takes_no_arguments_and_names_the_file_it_builds(
    tmp_path: Path,
) -> None:
    """An agent that must ask needs no parameters: the program is on disk."""
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)

    verify_output = create_verify_output_tool(
        executor,  # type: ignore[arg-type]
        workdir,
        renderer=StubRenderer(),  # type: ignore[arg-type]
        artifact_presenter=None,
        source_filename="candidate.py",
    )

    assert verify_output.name == "verify_output"
    assert verify_output.get_input_jsonschema()["properties"] == {}
    assert "/work/candidate.py" in verify_output.description
    assert (tmp_path / "attempts").is_dir()


def test_the_tool_result_is_what_the_model_reads(tmp_path: Path) -> None:
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")
    verify_output = create_verify_output_tool(
        executor,  # type: ignore[arg-type]
        workdir,
        renderer=StubRenderer(),  # type: ignore[arg-type]
        artifact_presenter=_artifact_presenter(),
    )

    result = verify_output.invoke({})

    assert _report_json(result)["status"] == "VERIFIED"
    # The source stays in the report the workflow keeps, never in the context.
    assert "source" not in _report_json(result)
    assert "techdraw.dxf" in _text(result)


def test_delegates_paths_and_returns_json_safe_mapping(tmp_path: Path) -> None:
    executor = StubCadQueryExecutor(
        _execution_report(
            returncode=0,
            stdout="construction log",
            stderr="construction warning",
        )
    )
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")
    verifier = _create_verifier(executor, workdir)

    result = verifier.feedback()

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
        "shape": "",
    }
    report = _report_json(result)
    assert isinstance(report["returncode"], int)
    assert "source" not in report
    assert (tmp_path / "attempts" / "000" / "model.py").read_text(
        encoding="utf-8"
    ) == VALID_SOURCE


def test_preserves_failed_attempt_and_execution_report(tmp_path: Path) -> None:
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
    verifier = _create_verifier(executor, workdir)

    result = verifier.feedback()

    assert _report_json(result) == {
        "verification_id": "000",
        "status": "FAILED",
        "returncode": 1,
        "stdout": "partial output",
        "stderr": "execution failed",
        "executor_error": "output.step was not generated",
        "shape": "",
    }
    attempt_dir = tmp_path / "attempts" / "000"
    assert (attempt_dir / "model.py").read_text(encoding="utf-8") == VALID_SOURCE
    assert not (attempt_dir / "output.step").exists()


def test_the_intermediate_returns_are_kept_beside_their_attempt(
    tmp_path: Path,
) -> None:
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")
    verifier = _create_verifier(executor, workdir)

    report, _ = verifier.verify()

    assert executor.intermediate_returns_dirs == [
        tmp_path / "attempts" / report.verification_id / "intermediate_returns"
    ]


def test_assigns_incrementing_verification_ids(tmp_path: Path) -> None:
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")
    verifier = _create_verifier(executor, workdir)

    first = verifier.feedback()
    second = verifier.feedback()

    assert _report_json(first)["verification_id"] == "000"
    assert _report_json(second)["verification_id"] == "001"
    assert (tmp_path / "attempts" / "000").is_dir()
    assert (tmp_path / "attempts" / "001").is_dir()


def test_rejects_missing_source_without_issuing_id(tmp_path: Path) -> None:
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    verifier = _create_verifier(executor, workdir)

    result = verifier.feedback()

    assert _report_json(result) == {
        "verification_id": None,
        "status": "REJECTED",
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "executor_error": "model.py was not found",
        "shape": "",
    }
    assert executor.calls == []
    assert list((tmp_path / "attempts").iterdir()) == []


def test_rejects_source_symlink_without_issuing_id(tmp_path: Path) -> None:
    real_source = tmp_path / "real-model.py"
    real_source.write_text(VALID_SOURCE, encoding="utf-8")
    (tmp_path / "model.py").symlink_to(real_source)
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    verifier = _create_verifier(executor, workdir)

    result = verifier.feedback()

    assert _report_json(result)["verification_id"] is None
    assert _report_json(result)["status"] == "REJECTED"
    assert _report_json(result)["executor_error"] == "model.py must not be a symlink"
    assert executor.calls == []
    assert list((tmp_path / "attempts").iterdir()) == []


def test_preserves_executor_rejection_without_source_snapshot(
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
    verifier = _create_verifier(executor, workdir)

    result = verifier.feedback()

    assert _report_json(result)["verification_id"] == "000"
    assert _report_json(result)["status"] == "REJECTED"
    assert _report_json(result)["executor_error"] == "model.py must be valid UTF-8"
    assert not (tmp_path / "attempts" / "000" / "model.py").exists()
    assert len(executor.calls) == 1


def test_the_report_keeps_the_source_that_feedback_leaves_out(tmp_path: Path) -> None:
    """`verify` is what the workflow stores; `feedback` is what a model reads."""
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")
    verifier = _create_verifier(executor, workdir)

    report, _ = verifier.verify()

    assert report == VerifyOutputResult(
        verification_id="000",
        status=ExecutionStatus.VERIFIED,
        source=VALID_SOURCE,
        returncode=0,
        stdout="construction log",
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
def test_rejects_output_dirname_outside_workdir_root(
    tmp_path: Path,
    output_dirname: PurePosixPath,
) -> None:
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)

    with pytest.raises(ValueError, match="output_dirname must be a directory basename"):
        _create_verifier(executor, workdir, output_dirname=output_dirname)


def test_rejects_symlink_output_directory(tmp_path: Path) -> None:
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    (tmp_path / "attempts").symlink_to(outside_dir, target_is_directory=True)
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)

    with pytest.raises(ValueError, match="output directory must not be a symlink"):
        _create_verifier(executor, workdir)


def test_verified_output_is_rendered_and_offered_to_the_model(tmp_path: Path) -> None:
    """A verified STEP must yield a drawing plus one render per style, and the
    tool result must point the model at all of them in sandbox coordinates."""
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")
    renderer = StubRenderer()
    verifier = _create_verifier(
        executor,
        workdir,
        renderer=renderer,
        artifact_presenter=_artifact_presenter(),
    )

    text = _text(verifier.feedback())

    verification_dir = tmp_path / "attempts" / "000"
    (rendered_step, _, _) = renderer.calls[0]
    assert rendered_step == verification_dir / "output.step"
    sandbox_dir = f"{workdir.sandbox_bind_dir}/attempts/000"
    assert f"{sandbox_dir}/techdraw.dxf" in text
    for style in RENDER3D_STYLES:
        assert f"{sandbox_dir}/render_3d/{style}.png" in text


def test_rendered_artifacts_stay_inside_the_verification_directory(
    tmp_path: Path,
) -> None:
    """Nothing may be written where another attempt, or the agent, could see it."""
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")
    verifier = _create_verifier(
        executor,
        workdir,
        renderer=StubRenderer(),
        artifact_presenter=_artifact_presenter(),
    )

    verifier.feedback()

    verification_dir = tmp_path / "attempts" / "000"
    written = {p for p in tmp_path.rglob("*") if p.is_file()}
    assert written == {
        tmp_path / "model.py",
        verification_dir / "model.py",
        verification_dir / "output.step",
        verification_dir / "techdraw.dxf",
        *(verification_dir / "render_3d" / f"{style}.png" for style in RENDER3D_STYLES),
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
    verifier = _create_verifier(
        executor,
        workdir,
        renderer=renderer,
        artifact_presenter=_artifact_presenter(),
    )

    result = verifier.feedback()

    assert renderer.calls == []
    assert _report_json(result)["status"] == "FAILED"
    assert "techdraw.dxf" not in _text(result)


def test_partial_render_offers_only_existing_styles_and_explains_the_rest(
    tmp_path: Path,
) -> None:
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")
    renderer = StubRenderer(skip_styles=("transparent_shaded_edges_perspective",))
    verifier = _create_verifier(
        executor,
        workdir,
        renderer=renderer,
        artifact_presenter=_artifact_presenter(),
    )

    result = verifier.feedback()
    text = _text(result)

    assert "transparent_shaded_edges_perspective.png" not in text
    assert "hlg_perspective.png" in text
    # The reason belongs where the render would have been, not in the report.
    assert "render_errors" not in _report_json(result)
    assert (
        "- transparent_shaded_edges_perspective: unavailable "
        "(RuntimeError: transparent_shaded_edges_perspective failed)"
    ) in text


def test_result_carries_paths_but_never_the_drawing_itself(
    tmp_path: Path,
) -> None:
    """The model is handed a path to open deliberately, not the DXF body."""
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")
    verifier = _create_verifier(
        executor,
        workdir,
        renderer=StubRenderer(),
        artifact_presenter=_artifact_presenter(),
    )

    text = _text(verifier.feedback())

    dxf_body = (tmp_path / "attempts" / "000" / "techdraw.dxf").read_text(
        encoding="utf-8"
    )
    assert "techdraw.dxf" in text
    assert dxf_body not in text


def test_images_are_embedded_only_when_the_presenter_asks_for_them(
    tmp_path: Path,
) -> None:
    """The feedback presentation mode decides whether images are embedded."""
    executor = StubCadQueryExecutor(_execution_report())
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")

    def block_types(mode: str) -> list[str]:
        workdir = SandboxWorkdir(host_bind_dir=tmp_path)
        result = _create_verifier(
            executor,
            workdir,
            renderer=StubRenderer(),
            artifact_presenter=_artifact_presenter(mode),
        ).feedback()
        assert isinstance(result, list)
        return [block["type"] for block in result]

    assert "image" not in block_types("path")
    assert "image" in block_types("image")
    assert "image" not in block_types("none")


def test_without_an_artifact_presenter_the_model_sees_only_the_report(
    tmp_path: Path,
) -> None:
    """Rendering for later evaluation must not leak artifacts into the context."""
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")
    verifier = _create_verifier(executor, workdir, renderer=StubRenderer())

    result = verifier.feedback()

    assert _report_json(result)["status"] == "VERIFIED"
    assert "techdraw.dxf" not in _text(result)
    assert (tmp_path / "attempts" / "000" / "techdraw.dxf").is_file()


def test_the_table_states_each_return_and_its_change() -> None:
    returns = (
        IntermediateReturn(
            "ret_base",
            census=ShapeCensus(1, 6000.0, Counter({"Plane": 6}), Counter({"Line": 12})),
        ),
        IntermediateReturn(
            "ret_hole",
            census=ShapeCensus(
                1,
                5880.0,
                Counter({"Plane": 10}),
                Counter({"Line": 20, "Circle": 4}),
            ),
        ),
    )

    assert _census_table(returns).splitlines() == [
        "ret_base  volume 6000.0; faces 6 (Plane 6); edges 12 (Line 12)",
        (
            "ret_hole  volume 5880.0 (-120.0); faces 10 (+4: Plane +4); "
            "edges 24 (+12: Line +8, Circle +4)"
        ),
    ]


def test_an_operation_that_built_nothing_shows_no_change() -> None:
    census = ShapeCensus(1, 6000.0, Counter({"Plane": 6}), Counter({"Line": 12}))
    returns = (
        IntermediateReturn("ret_base", census=census),
        IntermediateReturn("ret_cleaned", census=census),
    )

    assert _census_table(returns).splitlines()[1] == (
        "ret_cleaned  volume 6000.0 (+0.0); faces 6 (+0); edges 12 (+0)"
    )


def test_a_return_that_was_not_exported_carries_the_reason() -> None:
    returns = (
        IntermediateReturn(
            "ret_base",
            census=ShapeCensus(1, 6000.0, Counter({"Plane": 6}), Counter({"Line": 12})),
        ),
        IntermediateReturn("ret_count", error="TypeError: not a shape"),
        IntermediateReturn(
            "ret_grown",
            census=ShapeCensus(1, 9000.0, Counter({"Plane": 6}), Counter({"Line": 12})),
        ),
    )

    lines = _census_table(returns).splitlines()

    assert lines[1] == "ret_count  not exported: TypeError: not a shape"
    # The change is measured from the last return that reached a STEP.
    assert lines[2].startswith("ret_grown  volume 9000.0 (+3000.0)")


def test_the_table_of_nothing_is_empty() -> None:
    assert _census_table(()) == ""


def test_every_kept_return_is_drawn_beside_its_step(tmp_path: Path) -> None:
    executor = StubCadQueryExecutor(
        _execution_report(), return_names=("ret_base", "ret_hole")
    )
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")
    verifier = _create_verifier(executor, workdir)

    report, _ = verifier.verify()

    assert report.verification_id is not None
    returns_dir = (
        tmp_path / "attempts" / report.verification_id / "intermediate_returns"
    )
    for name in ("ret_base", "ret_hole"):
        assert (returns_dir / name / "output.step").is_file()
        assert (returns_dir / name / "techdraw.dxf").is_file()
        for style in RENDER3D_STYLES:
            assert (returns_dir / name / "render_3d" / f"{style}.png").is_file()


def test_the_feedback_states_the_returns_and_where_they_were_drawn(
    tmp_path: Path,
) -> None:
    executor = StubCadQueryExecutor(
        _execution_report(), return_names=("ret_base", "ret_hole")
    )
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")
    verifier = _create_verifier(executor, workdir)

    text = _text(verifier.feedback())

    assert "ret_base  volume 100.0; faces 6 (Plane 6); edges 12 (Line 12)" in text
    assert "ret_hole  volume 200.0 (+100.0); faces 6 (+0); edges 12 (+0)" in text
    # Sandbox paths, and one sentence for a layout every return shares.
    assert "/work/attempts/000/intermediate_returns/<name>/" in text
    # The table is a block of its own, not a JSON string full of escapes.
    assert "intermediate_returns" not in _report_json(verifier.feedback())


def test_a_return_whose_views_failed_is_named(tmp_path: Path) -> None:
    executor = StubCadQueryExecutor(_execution_report(), return_names=("ret_base",))
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")
    verifier = _create_verifier(
        executor,
        workdir,
        renderer=StubRenderer(skip_styles=("hlg_perspective",)),
    )

    text = _text(verifier.feedback())

    assert "Views that could not be drawn:" in text
    assert "ret_base: RuntimeError: hlg_perspective failed" in text


def test_a_program_that_kept_nothing_says_nothing_about_returns(
    tmp_path: Path,
) -> None:
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")
    verifier = _create_verifier(executor, workdir)

    assert "Intermediate returns" not in _text(verifier.feedback())


def test_the_returns_are_neither_kept_nor_drawn_when_switched_off(
    tmp_path: Path,
) -> None:
    executor = StubCadQueryExecutor(
        _execution_report(), return_names=("ret_base", "ret_hole")
    )
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")
    renderer = StubRenderer()
    verifier = OutputVerifier(
        executor,  # type: ignore[arg-type]
        workdir,
        renderer=renderer,  # type: ignore[arg-type]
        artifact_presenter=None,
        show_intermediate_returns=False,
    )

    text = _text(verifier.feedback())

    # The executor is never asked for them, so nothing downstream can run.
    assert executor.intermediate_returns_dirs == [None]
    assert not (tmp_path / "attempts" / "000" / "intermediate_returns").exists()
    assert "Intermediate returns" not in text
    # One render, for the attempt itself.
    assert len(renderer.calls) == 1


def test_a_result_that_fails_still_reports_what_the_returns_built(
    tmp_path: Path,
) -> None:
    executor = StubCadQueryExecutor(
        _execution_report(status=ExecutionStatus.FAILED, returncode=0),
        return_names=("ret_base", "ret_hole"),
    )
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")
    verifier = _create_verifier(executor, workdir)

    text = _text(verifier.feedback())

    # The program ran, so its returns are the diagnostic for why `result` is not
    # one valid solid.
    assert "ret_base  volume 100.0" in text
    assert "ret_hole  volume 200.0 (+100.0)" in text
    returns_dir = tmp_path / "attempts" / "000" / "intermediate_returns"
    assert (returns_dir / "ret_base" / "techdraw.dxf").is_file()
    # No STEP of its own, so the attempt's own views are absent.
    assert "attempts/000/techdraw.dxf" not in text


def test_a_part_that_broke_apart_says_how_many_pieces() -> None:
    returns = (
        IntermediateReturn(
            "ret_whole",
            census=ShapeCensus(1, 6000.0, Counter({"Plane": 6}), Counter({"Line": 12})),
        ),
        IntermediateReturn(
            "ret_split",
            census=ShapeCensus(
                3, 5000.0, Counter({"Plane": 18}), Counter({"Line": 36})
            ),
        ),
    )

    lines = _census_table(returns).splitlines()

    # One solid is the normal case and goes unsaid.
    assert lines[0].startswith("ret_whole  volume 6000.0")
    assert lines[1].startswith("ret_split  solids 3 (+2); volume 5000.0 (-1000.0)")
