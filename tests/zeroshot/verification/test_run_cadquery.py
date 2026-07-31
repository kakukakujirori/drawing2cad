import errno
import sys
from collections.abc import Callable
from dataclasses import FrozenInstanceError
from pathlib import Path

import cadquery as cq
import pytest

from zeroshot.pipeline.sandbox import (
    SandboxResult,
    SandboxRunner,
    SandboxStatus,
    SandboxWorkdir,
)
from zeroshot.pipeline.verification.run_cadquery import (
    CadQueryExecutionReport,
    CadQueryExecutor,
    ExecutionStatus,
    StepVerificationError,
)

VALID_BOX_SOURCE = """\
import cadquery as cq

result = cq.Workplane("XY").box(10, 20, 30)
"""


class StubSandboxRunner:
    def __init__(
        self,
        result: SandboxResult,
        step_writer: Callable[[Path], None] | None = None,
    ) -> None:
        self.result = result
        self.step_writer = step_writer
        self.calls: list[tuple[str, SandboxWorkdir]] = []

    def run(
        self,
        command: str,
        workdir: SandboxWorkdir,
        timeout_s: float | None = None,
    ) -> SandboxResult:
        del timeout_s
        self.calls.append((command, workdir))
        if self.step_writer is not None:
            self.step_writer(workdir.host_bind_dir / "output.step")
        return self.result


def _sandbox_result(
    *,
    status: SandboxStatus = SandboxStatus.COMPLETED,
    returncode: int | None = 0,
    stdout: str = "",
    stderr: str = "",
) -> SandboxResult:
    return SandboxResult(
        status=status,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _executor(
    sandbox_result: SandboxResult,
    step_writer: Callable[[Path], None] | None = None,
) -> tuple[CadQueryExecutor, StubSandboxRunner]:
    runner = StubSandboxRunner(sandbox_result, step_writer)
    executor = CadQueryExecutor(
        sandbox_runner=runner,  # type: ignore[arg-type]
    )
    return executor, runner


def _write_model(tmp_path: Path, source: str = VALID_BOX_SOURCE) -> Path:
    model_path = tmp_path / "model.py"
    model_path.write_text(source, encoding="utf-8")
    return model_path


def _write_valid_box_step(path: Path) -> None:
    cq.exporters.export(
        cq.Workplane("XY").box(10, 20, 30),
        str(path),
        exportType="STEP",
    )


@pytest.mark.parametrize(
    ("status", "serialized_value"),
    [
        (ExecutionStatus.VERIFIED, "VERIFIED"),
        (ExecutionStatus.REJECTED, "REJECTED"),
        (ExecutionStatus.FAILED, "FAILED"),
        (ExecutionStatus.TIMEOUT, "TIMEOUT"),
        (ExecutionStatus.INFRA_ERROR, "INFRA_ERROR"),
    ],
)
def test_execution_status_has_stable_serialized_value(
    status: ExecutionStatus,
    serialized_value: str,
) -> None:
    assert status.value == serialized_value


def test_execution_report_is_immutable() -> None:
    report = CadQueryExecutionReport(
        source="result = None",
        status=ExecutionStatus.REJECTED,
    )

    with pytest.raises(FrozenInstanceError):
        report.status = ExecutionStatus.VERIFIED  # type: ignore[misc]


def test_source_validation_accepts_valid_cadquery_source() -> None:
    CadQueryExecutor.validate_source(VALID_BOX_SOURCE)


def test_source_validation_rejects_syntax_error() -> None:
    with pytest.raises(SyntaxError):
        CadQueryExecutor.validate_source("result = (")


def test_verify_step_accepts_one_valid_solid(tmp_path: Path) -> None:
    step_path = tmp_path / "box.step"
    _write_valid_box_step(step_path)

    CadQueryExecutor.verify_step(step_path)


def test_verify_step_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(StepVerificationError, match="not found"):
        CadQueryExecutor.verify_step(tmp_path / "missing.step")


def test_verify_step_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "box.step"
    link = tmp_path / "box-link.step"
    _write_valid_box_step(target)
    link.symlink_to(target)

    with pytest.raises(StepVerificationError, match="symlink"):
        CadQueryExecutor.verify_step(link)


def test_verify_step_rejects_invalid_step(tmp_path: Path) -> None:
    step_path = tmp_path / "invalid.step"
    step_path.write_text("not a STEP file", encoding="utf-8")

    with pytest.raises(StepVerificationError, match="Failed to import STEP"):
        CadQueryExecutor.verify_step(step_path)


def test_verify_step_rejects_zero_solids(tmp_path: Path) -> None:
    step_path = tmp_path / "open-shell.step"
    box = cq.Workplane("XY").box(10, 10, 10).val()
    open_shell = cq.Shell.makeShell(box.Faces()[:5])
    cq.exporters.export(open_shell, str(step_path), exportType="STEP")

    with pytest.raises(StepVerificationError, match="found 0"):
        CadQueryExecutor.verify_step(step_path)


def test_verify_step_rejects_multiple_solids(tmp_path: Path) -> None:
    step_path = tmp_path / "multi-solid.step"
    first = cq.Workplane("XY").box(10, 10, 10)
    second = cq.Workplane("XY").box(10, 10, 10).translate((10, 10, 0))
    multi_solid = first.union(second, glue=False)
    cq.exporters.export(multi_solid, str(step_path), exportType="STEP")

    with pytest.raises(StepVerificationError, match="found 2"):
        CadQueryExecutor.verify_step(step_path)


def test_execute_writes_verified_step_to_requested_path(tmp_path: Path) -> None:
    executor, runner = _executor(
        _sandbox_result(
            stdout="construction log",
            stderr="construction warning",
        ),
        _write_valid_box_step,
    )
    model_path = _write_model(tmp_path)
    output_step_path = tmp_path / "saved" / "output.step"
    output_step_path.parent.mkdir()

    report = executor.execute(model_path, output_step_path)

    assert report == CadQueryExecutionReport(
        source=VALID_BOX_SOURCE,
        status=ExecutionStatus.VERIFIED,
        returncode=0,
        stdout="construction log",
        stderr="construction warning",
    )
    assert output_step_path.is_file()
    assert runner.calls[0][0] == "python model.py"
    CadQueryExecutor.verify_step(output_step_path)


def test_execute_can_validate_without_persisting_step(tmp_path: Path) -> None:
    executor, _ = _executor(_sandbox_result(), _write_valid_box_step)
    model_path = _write_model(tmp_path)

    report = executor.execute(model_path)

    assert report.status is ExecutionStatus.VERIFIED
    assert report.source == VALID_BOX_SOURCE
    assert not (tmp_path / "output.step").exists()


def test_execute_runs_valid_box_in_real_sandbox(tmp_path: Path) -> None:
    executor = CadQueryExecutor(
        sandbox_runner=SandboxRunner(
            python_executable=Path(sys.executable),
            default_timeout_s=30.0,
        ),
    )
    model_path = _write_model(tmp_path)
    output_step_path = tmp_path / "output.step"

    report = executor.execute(model_path, output_step_path)

    assert report.status is ExecutionStatus.VERIFIED
    assert report.returncode == 0
    assert report.source == VALID_BOX_SOURCE
    CadQueryExecutor.verify_step(output_step_path)


def test_execute_rejects_non_utf8_source_before_running_sandbox(
    tmp_path: Path,
) -> None:
    executor, runner = _executor(_sandbox_result())
    model_path = tmp_path / "model.py"
    model_path.write_bytes(b"\xff")

    report = executor.execute(model_path)

    assert report.status is ExecutionStatus.REJECTED
    assert report.source is None
    assert report.executor_error == "model.py must be valid UTF-8"
    assert runner.calls == []


def test_execute_maps_permission_error_before_running_sandbox(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, runner = _executor(_sandbox_result())
    model_path = _write_model(tmp_path)

    def raise_permission_error(
        self: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        del self, encoding, errors
        raise PermissionError

    monkeypatch.setattr(Path, "read_text", raise_permission_error)

    report = executor.execute(model_path)

    assert report.status is ExecutionStatus.REJECTED
    assert report.source is None
    assert report.executor_error == "model.py is not readable"
    assert runner.calls == []


def test_execute_maps_read_error_without_exposing_host_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, runner = _executor(_sandbox_result())
    model_path = _write_model(tmp_path)
    read_error = OSError(errno.EIO, "Input/output error", str(model_path))

    def raise_read_error(
        self: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        del self, encoding, errors
        raise read_error

    monkeypatch.setattr(Path, "read_text", raise_read_error)

    report = executor.execute(model_path)

    assert report.status is ExecutionStatus.INFRA_ERROR
    assert report.source is None
    assert report.executor_error == "Failed to read model.py: Input/output error"
    assert str(tmp_path) not in str(report.executor_error)
    assert runner.calls == []


def test_execute_maps_timeout_and_preserves_raw_output(tmp_path: Path) -> None:
    executor, _ = _executor(
        _sandbox_result(
            status=SandboxStatus.TIMEOUT,
            returncode=None,
            stdout="partial output",
            stderr="",
        ),
    )

    report = executor.execute(_write_model(tmp_path))

    assert report.status is ExecutionStatus.TIMEOUT
    assert report.source == VALID_BOX_SOURCE
    assert report.returncode is None
    assert report.stdout == "partial output"
    assert report.stderr == ""


def test_execute_maps_sandbox_infra_error(tmp_path: Path) -> None:
    executor, _ = _executor(
        _sandbox_result(
            status=SandboxStatus.INFRA_ERROR,
            returncode=None,
            stderr="bwrap failed",
        ),
    )

    report = executor.execute(_write_model(tmp_path))

    assert report.status is ExecutionStatus.INFRA_ERROR
    assert report.returncode is None
    assert report.stderr == "bwrap failed"


def test_execute_preserves_process_failure_and_empty_stderr(tmp_path: Path) -> None:
    executor, _ = _executor(
        _sandbox_result(
            returncode=7,
            stdout="diagnostic",
            stderr="",
        ),
    )

    report = executor.execute(_write_model(tmp_path))

    assert report.status is ExecutionStatus.FAILED
    assert report.returncode == 7
    assert report.executor_error is None
    assert report.stdout == "diagnostic"
    assert report.stderr == ""


def test_execute_rejects_missing_step_without_inventing_stderr(
    tmp_path: Path,
) -> None:
    executor, _ = _executor(_sandbox_result(returncode=0, stderr=""))

    report = executor.execute(_write_model(tmp_path))

    assert report.status is ExecutionStatus.FAILED
    assert report.returncode == 0
    assert report.executor_error == "output.step was not generated"
    assert report.stderr == ""


def test_execute_rejects_syntax_before_running_sandbox(tmp_path: Path) -> None:
    executor, runner = _executor(_sandbox_result())
    invalid_source = "result = ("

    report = executor.execute(_write_model(tmp_path, invalid_source))

    assert report.status is ExecutionStatus.REJECTED
    assert report.source == invalid_source
    assert report.executor_error is not None
    assert "was never closed" in report.executor_error
    assert report.stderr == ""
    assert runner.calls == []


def test_execute_preserves_process_output_on_step_verification_failure(
    tmp_path: Path,
) -> None:
    def write_invalid_step(path: Path) -> None:
        path.write_text("not a STEP file", encoding="utf-8")

    executor, _ = _executor(
        _sandbox_result(
            stdout="construction log",
            stderr="",
        ),
        write_invalid_step,
    )

    report = executor.execute(_write_model(tmp_path))

    assert report.status is ExecutionStatus.FAILED
    assert report.executor_error is not None
    assert "Failed to import STEP" in report.executor_error
    assert report.stdout == "construction log"
    assert report.stderr == ""


def test_execute_rejects_sandbox_output_symlink(tmp_path: Path) -> None:
    outside_step_path = tmp_path / "outside.step"
    _write_valid_box_step(outside_step_path)

    def write_output_symlink(path: Path) -> None:
        path.symlink_to(outside_step_path)

    executor, _ = _executor(_sandbox_result(), write_output_symlink)
    requested_output_path = tmp_path / "saved.step"

    report = executor.execute(_write_model(tmp_path), requested_output_path)

    assert report.status is ExecutionStatus.FAILED
    assert report.executor_error is not None
    assert "symlink" in report.executor_error
    assert not requested_output_path.exists()
