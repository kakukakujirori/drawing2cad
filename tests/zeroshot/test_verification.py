import sys
from collections.abc import Callable
from dataclasses import FrozenInstanceError
from hashlib import sha256
from pathlib import Path
from uuid import UUID

import cadquery as cq
import pytest

from zeroshot.pipeline.verification.run_cadquery import (
    CadQueryExecutionReport,
    CadQueryExecutor,
    ExecutionStatus,
    StepVerificationError,
)
from zeroshot.pipeline.sandbox import (
    SandboxResult,
    SandboxRunner,
    SandboxStatus,
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
        self.calls: list[tuple[str, Path]] = []

    def run(
        self,
        command: str,
        work_dir: Path,
        timeout_s: float | None = None,
    ) -> SandboxResult:
        del timeout_s
        self.calls.append((command, work_dir))
        if self.step_writer is not None:
            self.step_writer(work_dir / "output.step")
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
    tmp_path: Path,
    sandbox_result: SandboxResult,
    step_writer: Callable[[Path], None] | None = None,
) -> tuple[CadQueryExecutor, StubSandboxRunner, Path]:
    artifact_root = tmp_path / "artifacts"
    runner = StubSandboxRunner(sandbox_result, step_writer)
    executor = CadQueryExecutor(
        artifact_root=artifact_root,
        sandbox_runner=runner,  # type: ignore[arg-type]
    )
    return executor, runner, artifact_root


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


def test_execution_report_is_immutable(tmp_path: Path) -> None:
    report = CadQueryExecutionReport(
        exec_id="exec-1",
        source_path=tmp_path / "model.py",
        source_sha256="source-hash",
        status=ExecutionStatus.REJECTED,
        step_path=None,
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


def test_execute_creates_verified_artifacts(tmp_path: Path) -> None:
    executor, runner, artifact_root = _executor(
        tmp_path,
        _sandbox_result(
            stdout="construction log",
            stderr="construction warning",
        ),
        _write_valid_box_step,
    )

    report = executor.execute(source=VALID_BOX_SOURCE)

    assert UUID(report.exec_id)
    assert report.status is ExecutionStatus.VERIFIED
    assert report.returncode == 0
    assert report.source_path == artifact_root / report.exec_id / "model.py"
    assert report.source_path and report.source_path.read_text(encoding="utf-8") == VALID_BOX_SOURCE
    assert report.source_sha256 == sha256(VALID_BOX_SOURCE.encode("utf-8")).hexdigest()
    assert report.step_path == artifact_root / report.exec_id / "output.step"
    assert report.step_path and report.step_path.is_file()
    assert report.executor_error is None
    assert report.stdout == "construction log"
    assert report.stderr == "construction warning"
    assert runner.calls[0][0] == "python model.py"

    imported = cq.importers.importStep(str(report.step_path))
    solids = imported.solids().vals()
    assert len(solids) == 1
    assert cq.Shape(solids[0].wrapped).isValid()  # type: ignore[union-attr]


def test_execute_runs_valid_box_in_real_sandbox(tmp_path: Path) -> None:
    executor = CadQueryExecutor(
        artifact_root=tmp_path / "artifacts",
        sandbox_runner=SandboxRunner(
            python_executable=Path(sys.executable),
            default_timeout_s=30.0,
        ),
    )

    report = executor.execute(source=VALID_BOX_SOURCE)

    assert report.status is ExecutionStatus.VERIFIED
    assert report.returncode == 0
    assert report.step_path is not None
    CadQueryExecutor.verify_step(report.step_path)


def test_execute_maps_timeout_and_preserves_raw_output(tmp_path: Path) -> None:
    executor, _, _ = _executor(
        tmp_path,
        _sandbox_result(
            status=SandboxStatus.TIMEOUT,
            returncode=None,
            stdout="partial output",
            stderr="",
        ),
    )

    report = executor.execute(source=VALID_BOX_SOURCE)

    assert report.status is ExecutionStatus.TIMEOUT
    assert report.returncode is None
    assert report.stdout == "partial output"
    assert report.stderr == ""
    assert report.step_path is None


def test_execute_maps_sandbox_infra_error(tmp_path: Path) -> None:
    executor, _, _ = _executor(
        tmp_path,
        _sandbox_result(
            status=SandboxStatus.INFRA_ERROR,
            returncode=None,
            stderr="bwrap failed",
        ),
    )

    report = executor.execute(source=VALID_BOX_SOURCE)

    assert report.status is ExecutionStatus.INFRA_ERROR
    assert report.returncode is None
    assert report.stderr == "bwrap failed"
    assert report.step_path is None


def test_execute_preserves_process_failure_and_empty_stderr(tmp_path: Path) -> None:
    executor, _, _ = _executor(
        tmp_path,
        _sandbox_result(
            returncode=7,
            stdout="diagnostic",
            stderr="",
        ),
    )

    report = executor.execute(source=VALID_BOX_SOURCE)

    assert report.status is ExecutionStatus.FAILED
    assert report.returncode == 7
    assert report.executor_error is None
    assert report.stdout == "diagnostic"
    assert report.stderr == ""
    assert report.step_path is None


def test_execute_rejects_missing_step_without_inventing_stderr(
    tmp_path: Path,
) -> None:
    executor, _, _ = _executor(
        tmp_path,
        _sandbox_result(returncode=0, stderr=""),
    )

    report = executor.execute(source=VALID_BOX_SOURCE)

    assert report.status is ExecutionStatus.FAILED
    assert report.returncode == 0
    assert report.executor_error == "output.step was not generated"
    assert report.stderr == ""
    assert report.step_path is None


def test_execute_rejects_syntax_before_running_sandbox(tmp_path: Path) -> None:
    executor, runner, artifact_root = _executor(
        tmp_path,
        _sandbox_result(),
    )

    report = executor.execute(source="result = (")

    assert report.status is ExecutionStatus.REJECTED
    assert report.executor_error is not None
    assert "was never closed" in report.executor_error
    assert report.stderr == ""
    assert report.step_path is None
    assert report.source_path == artifact_root / report.exec_id / "model.py"
    assert runner.calls == []


def test_execute_preserves_process_output_on_step_verification_failure(
    tmp_path: Path,
) -> None:
    def write_invalid_step(path: Path) -> None:
        path.write_text("not a STEP file", encoding="utf-8")

    executor, _, _ = _executor(
        tmp_path,
        _sandbox_result(
            stdout="construction log",
            stderr="",
        ),
        write_invalid_step,
    )

    report = executor.execute(source=VALID_BOX_SOURCE)

    assert report.status is ExecutionStatus.FAILED
    assert report.executor_error is not None
    assert "Failed to import STEP" in report.executor_error
    assert report.stdout == "construction log"
    assert report.stderr == ""


def test_execute_refuses_to_overwrite_existing_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed_exec_id = UUID("00000000-0000-0000-0000-000000000001")
    monkeypatch.setattr(
        "zeroshot.pipeline.verification.run_cadquery.uuid4",
        lambda: fixed_exec_id,
    )
    executor, _, artifact_root = _executor(
        tmp_path,
        _sandbox_result(),
        _write_valid_box_step,
    )

    first_report = executor.execute(source=VALID_BOX_SOURCE)

    with pytest.raises(FileExistsError):
        executor.execute(source="result = 'must not overwrite'")

    assert first_report.status is ExecutionStatus.VERIFIED
    assert (artifact_root / str(fixed_exec_id) / "model.py").read_text(
        encoding="utf-8"
    ) == VALID_BOX_SOURCE
