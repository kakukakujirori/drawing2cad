import json
from pathlib import Path, PurePosixPath

import pytest
from langchain_core.tools import BaseTool

from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.tools.verify_output import (
    VerifyOutputResult,
    create_verify_output_tool,
)
from zeroshot.pipeline.verification.run_cadquery import (
    CadQueryExecutionReport,
    ExecutionStatus,
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
    render_views: bool = False,
    source_filename: str = "model.py",
    output_dirname: PurePosixPath = PurePosixPath("attempts"),
    serialize_output: bool = True,
) -> BaseTool:
    return create_verify_output_tool(
        executor,  # type: ignore[arg-type]
        workdir,
        render_views=render_views,
        source_filename=source_filename,
        output_dirname=output_dirname,
        serialize_output=serialize_output,
    )


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
    assert result == {
        "verification_id": "000",
        "status": "VERIFIED",
        "returncode": 0,
        "stdout": "construction log",
        "stderr": "construction warning",
        "executor_error": None,
    }
    assert json.loads(json.dumps(result)) == result
    assert isinstance(result["returncode"], int)
    assert "source" not in result
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

    assert result == {
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

    assert first["verification_id"] == "000"
    assert second["verification_id"] == "001"
    assert (tmp_path / "attempts" / "000").is_dir()
    assert (tmp_path / "attempts" / "001").is_dir()


def test_tool_rejects_missing_source_without_issuing_id(tmp_path: Path) -> None:
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    verify_output = _create_tool(executor, workdir)

    result = verify_output.invoke({})

    assert result == {
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

    assert result["verification_id"] is None
    assert result["status"] == "REJECTED"
    assert result["executor_error"] == "model.py must not be a symlink"
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

    assert result["verification_id"] == "000"
    assert result["status"] == "REJECTED"
    assert result["executor_error"] == "model.py must be valid UTF-8"
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


def test_render_views_guard_runs_after_successful_verification(
    tmp_path: Path,
) -> None:
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")
    verify_output = _create_tool(
        executor,
        workdir,
        render_views=True,
    )

    with pytest.raises(
        NotImplementedError,
        match="View rendering not implemented yet",
    ):
        verify_output.invoke({})

    assert len(executor.calls) == 1
