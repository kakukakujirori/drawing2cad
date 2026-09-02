import errno
import sys
from collections import Counter
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
    _returned_names,
)
from zeroshot.pipeline.verification.shape_census import ShapeCensus

VALID_BOX_SOURCE = """\
import cadquery as cq

result = cq.Workplane("XY").box(10, 20, 30)
"""

TWO_RESULT_SOURCE = """\
import cadquery as cq

ret_base = cq.Workplane("XY").box(10, 20, 30)  # the block
ret_hole = ret_base.cut(
    cq.Workplane("XY").box(2, 2, 100)
)
result = ret_hole
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


def test_source_validation_rejects_a_swallowed_failure() -> None:
    """A chamfer the kernel refused leaves a script that still exports, still
    verifies, and still reports success, with the feature missing from the
    solid.  Every later stage sees a plausible part, so this is the only place
    the difference is observable."""
    with pytest.raises(ValueError, match="Try-except is not allowed"):
        CadQueryExecutor.validate_source(
            "import cadquery as cq\n"
            'part = cq.Workplane("XY").box(10, 20, 30)\n'
            "try:\n"
            "    part = part.edges().chamfer(0.4)\n"
            "except Exception:\n"
            "    pass\n"
            "result = part\n"
        )


def test_source_validation_accepts_the_word_except_in_a_comment() -> None:
    """Generated models do write it: one baseline sample carries `# ... floor,
    except` above its cut.  Scanning for the word alone rejects the whole
    sample and asks the coder to remove a `try` it never wrote."""
    CadQueryExecutor.validate_source(
        "import cadquery as cq\n"
        "# hollowed all the way down to a 3.775 floor, except the rib\n"
        'result = cq.Workplane("XY").box(10, 20, 30)\n'
    )


def test_source_validation_accepts_a_finally_that_reraises() -> None:
    """`finally` without handlers cleans up and lets the exception through."""
    CadQueryExecutor.validate_source(
        "import cadquery as cq\n"
        "try:\n"
        '    result = cq.Workplane("XY").box(10, 20, 30)\n'
        "finally:\n"
        "    pass\n"
    )


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
        # A 10x20x30 box is six planes and twelve straight edges. The census
        # is here so that a coder handed this report can see when its arcs came
        # back as a hundred of them.
        census=ShapeCensus(1, 6000.0, Counter({"Plane": 6}), Counter({"Line": 12})),
    )
    assert output_step_path.is_file()
    assert runner.calls[0][0] == "python /work/_run_program.py /work/model.py"
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


def test_the_returned_names_are_the_ones_the_program_assigns_in_order() -> None:
    assert _returned_names(TWO_RESULT_SOURCE, "model.py") == ["ret_base", "ret_hole"]


def test_a_nested_assignment_is_not_kept() -> None:
    source = """\
def build():
    ret_inner = object()
    return ret_inner

result = build()
"""

    assert _returned_names(source, "model.py") == []


def test_execute_keeps_every_named_output_when_asked_for_them(
    tmp_path: Path,
) -> None:
    executor = CadQueryExecutor(
        sandbox_runner=SandboxRunner(
            python_executable=Path(sys.executable),
            default_timeout_s=60.0,
        ),
    )
    model_path = _write_model(tmp_path, TWO_RESULT_SOURCE)
    output_step_path = tmp_path / "output.step"
    intermediate_returns_dir = tmp_path / "intermediate_returns"

    report = executor.execute(
        model_path, output_step_path, intermediate_returns_dir=intermediate_returns_dir
    )

    assert report.status is ExecutionStatus.VERIFIED
    # The reported source stays what the coder wrote, not what was run.
    assert report.source == TWO_RESULT_SOURCE
    assert [output.name for output in report.intermediate_returns] == [
        "ret_base",
        "ret_hole",
    ]
    for output in report.intermediate_returns:
        assert output.error is None
        assert output.step_path is not None
        CadQueryExecutor.verify_step(output.step_path)


def test_a_program_that_does_not_build_keeps_nothing(tmp_path: Path) -> None:
    executor = CadQueryExecutor(
        sandbox_runner=SandboxRunner(
            python_executable=Path(sys.executable),
            default_timeout_s=60.0,
        ),
    )
    source = """\
import cadquery as cq

ret_base = cq.Workplane("XY").box(10, 20, 30)
ret_hole = ret_base.no_such_method()
result = ret_hole
"""
    model_path = _write_model(tmp_path, source)

    report = executor.execute(
        model_path, intermediate_returns_dir=tmp_path / "intermediate_returns"
    )

    assert report.status is ExecutionStatus.FAILED
    assert report.intermediate_returns == ()
    # What that run needs said is the line the coder wrote.
    assert "line 4" in report.stderr


def test_a_named_output_that_is_not_a_shape_is_reported_rather_than_kept(
    tmp_path: Path,
) -> None:
    executor = CadQueryExecutor(
        sandbox_runner=SandboxRunner(
            python_executable=Path(sys.executable),
            default_timeout_s=60.0,
        ),
    )
    source = """\
import cadquery as cq

ret_base = cq.Workplane("XY").box(10, 20, 30)
ret_count = 3
result = ret_base
"""
    model_path = _write_model(tmp_path, source)

    report = executor.execute(
        model_path, intermediate_returns_dir=tmp_path / "intermediate_returns"
    )

    assert report.status is ExecutionStatus.VERIFIED
    reported = {output.name: output.error for output in report.intermediate_returns}
    assert reported["ret_base"] is None
    # Whatever the exporter said about an int, said back rather than raised.
    assert reported["ret_count"]
    assert not (tmp_path / "intermediate_returns" / "ret_count.step").exists()


def test_the_staged_program_is_the_one_the_coder_wrote(tmp_path: Path) -> None:
    # Read inside the run: the sandbox workdir is gone once execute returns.
    seen: dict[str, object] = {}

    def capture(step_path: Path) -> None:
        seen["source"] = (step_path.parent / "model.py").read_text(encoding="utf-8")
        _write_valid_box_step(step_path)

    executor, runner = _executor(_sandbox_result(), capture)
    model_path = _write_model(tmp_path, TWO_RESULT_SOURCE)

    report = executor.execute(model_path)
    command, _ = runner.calls[0]

    assert report.intermediate_returns == ()
    assert seen["source"] == TWO_RESULT_SOURCE
    # No returns was asked for, so the runner is given no output to keep.
    assert command == "python /work/_run_program.py /work/model.py"


def test_the_runner_is_told_which_outputs_to_keep(tmp_path: Path) -> None:
    executor, runner = _executor(_sandbox_result(), _write_valid_box_step)
    model_path = _write_model(tmp_path, TWO_RESULT_SOURCE)

    executor.execute(
        model_path, intermediate_returns_dir=tmp_path / "intermediate_returns"
    )
    command, _ = runner.calls[0]

    assert command == "python /work/_run_program.py /work/model.py ret_base ret_hole"


def test_a_kept_return_is_counted(tmp_path: Path) -> None:
    executor = CadQueryExecutor(
        sandbox_runner=SandboxRunner(
            python_executable=Path(sys.executable),
            default_timeout_s=60.0,
        ),
    )
    model_path = _write_model(tmp_path, TWO_RESULT_SOURCE)

    report = executor.execute(
        model_path, intermediate_returns_dir=tmp_path / "intermediate_returns"
    )
    counted = {output.name: output.census for output in report.intermediate_returns}

    assert counted["ret_base"] is not None
    assert counted["ret_base"].volume == pytest.approx(6000.0)
    assert counted["ret_base"].faces == Counter({"Plane": 6})
    # A 2x2 bar cut through the 30mm depth of the block.
    assert counted["ret_hole"] is not None
    assert counted["ret_hole"].volume == pytest.approx(5880.0)


def test_a_syntax_error_is_reported_with_the_line_and_the_caret(
    tmp_path: Path,
) -> None:
    executor, _ = _executor(_sandbox_result())
    model_path = _write_model(tmp_path, 'result = cq.Workplane("XY".box(1, 2, 3)\n')

    report = executor.execute(model_path)

    assert report.status is ExecutionStatus.REJECTED
    assert report.executor_error is not None
    assert 'result = cq.Workplane("XY".box(1, 2, 3)' in report.executor_error
    assert "^" in report.executor_error
    assert "SyntaxError: '(' was never closed" in report.executor_error


def test_an_export_failure_carries_the_traceback_that_explains_it(
    tmp_path: Path,
) -> None:
    executor = CadQueryExecutor(
        sandbox_runner=SandboxRunner(
            python_executable=Path(sys.executable),
            default_timeout_s=60.0,
        ),
    )
    source = """\
import cadquery as cq

ret_base = cq.Workplane("XY").box(10, 20, 30)
result = 3
"""

    report = executor.execute(_write_model(tmp_path, source))

    assert report.status is ExecutionStatus.FAILED
    assert "Failed to export `result` to STEP:" in report.stderr
    # The exporter's own frames, not one line saying it raised.
    assert "cadquery/occ_impl/exporters" in report.stderr
    assert "TypeError" in report.stderr


def test_a_traceback_names_only_files_the_coder_can_edit(tmp_path: Path) -> None:
    executor = CadQueryExecutor(
        sandbox_runner=SandboxRunner(
            python_executable=Path(sys.executable),
            default_timeout_s=60.0,
        ),
    )
    source = """\
import cadquery as cq

result = cq.Workplane("XY").box(10, 20, 30).faces(">Z").fillet(100.0)
"""

    report = executor.execute(_write_model(tmp_path, source))

    assert report.status is ExecutionStatus.FAILED
    assert report.stderr.startswith("Traceback (most recent call last):\n  File")
    assert "_run_program.py" not in report.stderr
    assert "/work/model.py" in report.stderr


def test_a_return_in_several_pieces_is_counted_whole(tmp_path: Path) -> None:
    executor = CadQueryExecutor(
        sandbox_runner=SandboxRunner(
            python_executable=Path(sys.executable),
            default_timeout_s=60.0,
        ),
    )
    source = """\
import cadquery as cq

ret_apart = cq.Workplane("XY").box(10, 10, 10).union(
    cq.Workplane("XY").box(10, 10, 10).translate((100, 0, 0))
)
result = ret_apart
"""

    report = executor.execute(
        _write_model(tmp_path, source),
        intermediate_returns_dir=tmp_path / "intermediate_returns",
    )
    counted = report.intermediate_returns[0].census

    assert counted is not None
    assert counted.solids == 2
    # Both boxes, not just the one `.val()` would have returned.
    assert counted.volume == pytest.approx(2000.0)
