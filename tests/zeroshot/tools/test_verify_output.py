import errno
import json
from pathlib import Path

import pytest
from langchain_core.tools import BaseTool

from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.tools.verify_output import create_verify_output_tool
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
        self.sources: list[str] = []

    def execute(self, source: str) -> CadQueryExecutionReport:
        self.sources.append(source)
        return self.report


def _execution_report(
    *,
    status: ExecutionStatus = ExecutionStatus.VERIFIED,
    returncode: int | None = 0,
    stdout: str = "construction log",
    stderr: str = "",
    executor_error: str | None = None,
) -> CadQueryExecutionReport:
    return CadQueryExecutionReport(
        exec_id="verification-1",
        source_path=Path("/trusted/artifacts/verification-1/model.py"),
        source_sha256="source-hash",
        status=status,
        step_path=Path("/trusted/artifacts/verification-1/output.step"),
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
) -> BaseTool:
    return create_verify_output_tool(
        executor,  # type: ignore[arg-type]
        workdir,
        render_views=render_views,
        source_filename=source_filename,
    )


def test_tool_schema_exposes_no_runtime_arguments(tmp_path: Path) -> None:
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


def test_tool_executes_exact_source_and_returns_json_safe_mapping(
    tmp_path: Path,
) -> None:
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

    assert executor.sources == [VALID_SOURCE]
    assert result == {
        "status": "VERIFIED",
        "execution_id": "verification-1",
        "source_sha256": "source-hash",
        "returncode": 0,
        "stdout": "construction log",
        "stderr": "construction warning",
        "executor_error": None,
    }
    assert json.loads(json.dumps(result)) == result
    assert isinstance(result["returncode"], int)
    assert result["executor_error"] is None
    assert "source_path" not in result
    assert "step_path" not in result


def test_tool_preserves_failed_execution_report(tmp_path: Path) -> None:
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
        "status": "FAILED",
        "execution_id": "verification-1",
        "source_sha256": "source-hash",
        "returncode": 1,
        "stdout": "partial output",
        "stderr": "execution failed",
        "executor_error": "output.step was not generated",
    }
    assert executor.sources == [VALID_SOURCE]


def test_tool_rejects_missing_source_without_calling_executor(
    tmp_path: Path,
) -> None:
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    verify_output = _create_tool(executor, workdir)

    result = verify_output.invoke({})

    assert result == {
        "status": "REJECTED",
        "execution_id": None,
        "source_sha256": None,
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "executor_error": "model.py was not found",
    }
    assert executor.sources == []


def test_tool_rejects_source_symlink_without_calling_executor(
    tmp_path: Path,
) -> None:
    real_source = tmp_path / "real-model.py"
    real_source.write_text(VALID_SOURCE, encoding="utf-8")
    (tmp_path / "model.py").symlink_to(real_source)
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    verify_output = _create_tool(executor, workdir)

    result = verify_output.invoke({})

    assert result["status"] == "REJECTED"
    assert result["executor_error"] == "model.py must not be a symlink"
    assert executor.sources == []


def test_tool_rejects_non_utf8_source_without_calling_executor(
    tmp_path: Path,
) -> None:
    (tmp_path / "model.py").write_bytes(b"\xff")
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    verify_output = _create_tool(executor, workdir)

    result = verify_output.invoke({})

    assert result["status"] == "REJECTED"
    assert result["executor_error"] == "model.py must be valid UTF-8"
    assert executor.sources == []


def test_tool_maps_permission_error_to_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.py"
    model_path.write_text(VALID_SOURCE, encoding="utf-8")
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    verify_output = _create_tool(executor, workdir)

    def raise_permission_error(
        self: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        del self, encoding, errors
        raise PermissionError

    monkeypatch.setattr(Path, "read_text", raise_permission_error)

    result = verify_output.invoke({})

    assert result["status"] == "REJECTED"
    assert result["executor_error"] == "model.py is not readable"
    assert executor.sources == []


def test_tool_maps_unexpected_read_error_without_exposing_host_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.py"
    model_path.write_text(VALID_SOURCE, encoding="utf-8")
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    verify_output = _create_tool(executor, workdir)
    read_error = OSError(
        errno.EIO,
        "Input/output error",
        str(model_path),
    )

    def raise_read_error(
        self: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        del self, encoding, errors
        raise read_error

    monkeypatch.setattr(Path, "read_text", raise_read_error)

    result = verify_output.invoke({})

    assert result["status"] == "INFRA_ERROR"
    assert result["executor_error"] == "Failed to read model.py: Input/output error"
    assert str(tmp_path) not in str(result["executor_error"])
    assert executor.sources == []


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

    assert executor.sources == [VALID_SOURCE]
