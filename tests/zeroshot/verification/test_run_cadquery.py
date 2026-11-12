import errno
import math
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
from zeroshot.pipeline.verification._run_program import PROGRAM_LOG
from zeroshot.pipeline.verification.program_log import (
    describe_ineffective_booleans,
    describe_inside_out,
)
from zeroshot.pipeline.verification.run_cadquery import (
    CadQueryExecutionReport,
    CadQueryExecutor,
    ExecutionStatus,
    StepVerificationError,
)
from zeroshot.pipeline.verification.shape_census import ShapeCensus
from zeroshot.pipeline.verification.solid_mesh import UnjoinedMeshError

VALID_BOX_SOURCE = """\
import cadquery as cq

result = cq.Workplane("XY").box(10, 20, 30)
"""

TWO_RESULT_SOURCE = """\
import cadquery as cq

base = cq.Workplane("XY").box(10, 20, 30)  # the block
hole = base.cut(
    cq.Workplane("XY").box(2, 2, 100)
)
result = hole
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


def test_verify_step_rejects_a_solid_the_scorer_cannot_mesh(tmp_path: Path) -> None:
    # The slot's end arc bulges inward and crosses both walls; isValid still passes.
    slot = (
        cq.Workplane("XY")
        .moveTo(0, 20)
        .lineTo(1, 0)
        .threePointArc((5, 8), (9, 0))
        .lineTo(10, 20)
        .close()
        .extrude(3)
    )
    block = cq.Workplane("XY").box(30, 40, 10, centered=False).translate((-10, -10, 0))
    step_path = tmp_path / "self-crossing-cut.step"
    cq.exporters.export(block.cut(slot), str(step_path), exportType="STEP")

    with pytest.raises(StepVerificationError, match="fails to join near x ") as caught:
        CadQueryExecutor.verify_step(step_path)
    # The smallest region named lies on the slot, whose walls the arc crosses.
    unjoined = caught.value.__cause__
    assert isinstance(unjoined, UnjoinedMeshError)
    low, high = unjoined.regions[0]
    assert (low > -0.01).all() and (high < (10.01, 20.01, 3.01)).all()


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
        step_path=output_step_path,
        stdout="construction log",
        stderr="construction warning",
        # A 10x20x30 box is six planes and twelve straight edges. The census
        # is here so that a coder handed this report can see when its arcs came
        # back as a hundred of them.
        census=ShapeCensus(
            1,
            6000.0,
            (10.0, 20.0, 30.0),
            Counter({"Plane": 6}),
            Counter({"Line": 12}),
            warnings=(),
        ),
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


def test_a_freeform_spline_is_verified_and_only_requests_construction_review(tmp_path):
    source = (
        "import cadquery as cq\n"
        "result = (cq.Workplane().moveTo(0, 0)\n"
        "    .spline([(5, 2), (10, 0)], includeCurrent=True)\n"
        "    .lineTo(10, -5).lineTo(0, -5).close().extrude(5))\n"
    )
    model = tmp_path / "freeform.py"
    model.write_text(source)
    executor = CadQueryExecutor(
        SandboxRunner(python_executable=Path(sys.executable), default_timeout_s=30.0)
    )

    report = executor.execute(model, tmp_path / "output.step")

    assert report.status is ExecutionStatus.VERIFIED
    assert report.returncode == 0
    assert len(report.census.warnings) == 1
    assert "freeform.py:3 (spline)" in report.census.warnings[0]
    assert str(tmp_path) not in report.census.warnings[0]
    assert report.census.edges["BSplineCurve"] > 0
    CadQueryExecutor.verify_step(report.step_path)


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


def test_the_staged_program_is_the_one_the_coder_wrote(tmp_path: Path) -> None:
    # Read inside the run: the sandbox workdir is gone once execute returns.
    seen: dict[str, object] = {}

    def capture(step_path: Path) -> None:
        seen["source"] = (step_path.parent / "model.py").read_text(encoding="utf-8")
        _write_valid_box_step(step_path)

    executor, runner = _executor(_sandbox_result(), capture)
    model_path = _write_model(tmp_path, TWO_RESULT_SOURCE)

    executor.execute(model_path)
    command, _ = runner.calls[0]

    assert seen["source"] == TWO_RESULT_SOURCE
    assert command == "python /work/_run_program.py /work/model.py"


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

base = cq.Workplane("XY").box(10, 20, 30)
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


def test_a_result_in_several_pieces_is_counted_whole(tmp_path: Path) -> None:
    executor = CadQueryExecutor(
        sandbox_runner=SandboxRunner(
            python_executable=Path(sys.executable),
            default_timeout_s=60.0,
        ),
    )
    source = """\
import cadquery as cq

result = cq.Workplane("XY").box(10, 10, 10).union(
    cq.Workplane("XY").box(10, 10, 10).translate((100, 0, 0))
)
"""

    report = executor.execute(
        _write_model(tmp_path, source), output_step_path=tmp_path / "output.step"
    )
    counted = report.census

    # Rejected as two solids, but still counted for the coder to read.
    assert report.status is ExecutionStatus.FAILED
    assert counted is not None
    assert counted.solids == 2
    # Both boxes, not just the one `.val()` would have returned.
    assert counted.volume == pytest.approx(2000.0)


def _build(tmp_path: Path, source: str) -> CadQueryExecutionReport:
    executor = CadQueryExecutor(
        sandbox_runner=SandboxRunner(
            python_executable=Path(sys.executable),
            default_timeout_s=60.0,
        ),
    )
    return executor.execute(
        _write_model(tmp_path, source), output_step_path=tmp_path / "output.step"
    )


def test_a_cut_is_logged_with_the_volumes_and_boxes_around_it(tmp_path: Path) -> None:
    source = """\
import cadquery as cq

body = cq.Workplane("XY").box(10, 10, 10)
result = body.cut(cq.Workplane("XY").box(2, 2, 2).translate((50, 0, 0)))
"""

    report = _build(tmp_path, source)

    assert report.status is ExecutionStatus.VERIFIED
    assert report.program_log is not None
    (call,) = report.program_log.booleans
    assert (call.operation, call.line, call.caller_line) == ("cut", 4, 4)
    assert call.volume_before == pytest.approx(1000.0)
    assert call.volume_after == pytest.approx(1000.0)
    assert call.body_box == pytest.approx((-5, -5, -5, 5, 5, 5), abs=1e-6)
    assert call.tool_box == pytest.approx((49, -1, -1, 51, 1, 1), abs=1e-6)
    assert report.program_log.result_volumes == pytest.approx((1000.0,))


def test_workplane_operations_are_logged_at_their_lines(tmp_path: Path) -> None:
    source = """\
import cadquery as cq

body = cq.Workplane("XY").box(20, 20, 10)
body = body.faces(">Z").workplane().hole(2)
body = body.faces(">Z").workplane().center(5, 5).rect(2, 2).cutBlind(-3)
body = body.union(cq.Workplane("XY").box(2, 2, 2).translate((-5, -5, 5.5)))
result = body
"""

    report = _build(tmp_path, source)

    assert report.status is ExecutionStatus.VERIFIED
    assert report.program_log is not None
    assert [(call.operation, call.line) for call in report.program_log.booleans] == [
        ("cut", 4),
        ("cut", 5),
        ("fuse", 6),
    ]
    removed = [
        call.volume_before - call.volume_after  # type: ignore[operator]
        for call in report.program_log.booleans
    ]
    assert removed == pytest.approx([10 * math.pi, 12.0, -6.0], rel=1e-6)


def test_a_call_inside_a_helper_keeps_the_line_that_called_the_helper(
    tmp_path: Path,
) -> None:
    source = """\
import cadquery as cq


def drill(body, x):
    return body.cut(cq.Workplane("XY").cylinder(20, 1).translate((x, 0, 0)))


result = drill(drill(cq.Workplane("XY").box(10, 10, 10), -2), 2)
"""

    report = _build(tmp_path, source)

    assert report.program_log is not None
    assert [(call.line, call.caller_line) for call in report.program_log.booleans] == [
        (5, 8),
        (5, 8),
    ]


def test_booleans_without_a_solid_or_a_tool_are_not_logged(tmp_path: Path) -> None:
    source = """\
import cadquery as cq

pair = [cq.Solid.makeBox(1, 1, 1), cq.Solid.makeBox(1, 1, 1, pnt=cq.Vector(0.5, 0, 0))]
merged = cq.Compound.makeCompound(pair).fuse()
sketch = cq.Sketch().rect(10, 10).circle(2, mode="s")
result = cq.Workplane("XY").placeSketch(sketch).extrude(1)
"""

    report = _build(tmp_path, source)

    assert report.status is ExecutionStatus.VERIFIED
    assert report.program_log.booleans == ()


def test_a_program_that_raises_inside_a_boolean_logs_nothing(tmp_path: Path) -> None:
    source = """\
import cadquery as cq

box = cq.Solid.makeBox(1, 1, 1)
result = box.cut(None)
"""

    report = _build(tmp_path, source)

    assert report.status is ExecutionStatus.FAILED
    assert report.program_log is None
    assert "Traceback (most recent call last):" in report.stderr
    assert "_run_program.py" not in report.stderr
    assert '"/work/model.py", line 4' in report.stderr
    assert "cadquery/occ_impl/shapes.py" in report.stderr


def test_booleans_without_effect_are_named_from_a_real_build(tmp_path: Path) -> None:
    source = """\
import cadquery as cq

body = cq.Workplane("XY").box(10, 10, 10)
body = body.cut(cq.Workplane("XY").box(2, 2, 2).translate((50, 0, 0)))
body = body.cut(cq.Workplane("XY").box(2, 2, 20))
body = body.cut(cq.Workplane("XY").box(1, 1, 4))
body = body.union(cq.Workplane("XY").box(2, 2, 2).translate((3, 3, 0)))
body = body.union(cq.Workplane("XY").box(2, 2, 2).translate((3, 3, 5.5)))
result = body
"""

    report = _build(tmp_path, source)

    assert report.status is ExecutionStatus.VERIFIED
    assert report.program_log is not None
    entries = describe_ineffective_booleans(report.program_log.booleans, "model.py")
    assert [entry.split("\n")[0] for entry in entries] == [
        "model.py:4 cut removed no material.",
        (
            "model.py:6 cut removed no material: the cutter's region holds no "
            "material at this point."
        ),
        (
            "model.py:7 union added no material: the tool's region is already "
            "material at this point."
        ),
    ]


def test_an_empty_intersect_and_a_cut_of_everything_are_named(tmp_path: Path) -> None:
    source = """\
import cadquery as cq

body = cq.Solid.makeBox(10, 10, 10)
empty = body.intersect(cq.Solid.makeBox(2, 2, 2, pnt=cq.Vector(50, 0, 0)))
gone = body.cut(cq.Solid.makeBox(30, 30, 30, pnt=cq.Vector(-10, -10, -10)))
result = body
"""

    report = _build(tmp_path, source)

    assert report.program_log is not None
    entries = describe_ineffective_booleans(report.program_log.booleans, "model.py")
    assert [entry.split("\n")[0] for entry in entries] == [
        "model.py:4 intersect left no material.",
        "model.py:5 cut removed all material: the cutter covers the whole body.",
    ]


def test_an_inside_out_result_is_logged_with_its_negative_volume(
    tmp_path: Path,
) -> None:
    source = """\
import cadquery as cq

box = cq.Solid.makeBox(10, 10, 10)
result = cq.Solid(box.wrapped.Reversed())
"""

    report = _build(tmp_path, source)

    assert report.program_log is not None
    assert report.program_log.result_volumes == pytest.approx((-1000.0,))
    assert describe_inside_out(report.program_log, "model.py") == (
        "`result` is an inside-out solid (one whose faces point inward): its "
        "volume is -1000.000 mm³."
    )


def test_an_inside_out_result_fails_the_build_that_its_step_would_pass(
    tmp_path: Path,
) -> None:
    source = """\
import cadquery as cq

result = cq.Solid(cq.Solid.makeBox(10, 10, 10).wrapped.Reversed())
"""

    report = _build(tmp_path, source)

    # The STEP alone passes: it reads back as an ordinary box.
    assert report.step_path is not None
    CadQueryExecutor.verify_step(report.step_path)
    assert report.status is ExecutionStatus.FAILED
    assert report.executor_error == (
        "`result` is an inside-out solid (one whose faces point inward): its "
        "volume is -1000.000 mm³."
    )


def test_a_boolean_given_an_inside_out_body_fails_the_build_at_its_line(
    tmp_path: Path,
) -> None:
    source = """\
import cadquery as cq

body = cq.Solid(cq.Solid.makeBox(10, 10, 10).wrapped.Reversed())
result = body.cut(cq.Solid.makeBox(2, 2, 2))
"""

    report = _build(tmp_path, source)

    assert report.status is ExecutionStatus.FAILED
    assert report.executor_error is not None
    assert report.executor_error.startswith(
        "model.py:4 cut was given an inside-out body (one whose faces point "
        "inward): its volume is -1000.000 mm³.\n  body"
    )


def test_an_inside_out_solid_is_named_before_the_step_error_it_causes(
    tmp_path: Path,
) -> None:
    def write_two_solids_and_an_inside_out_log(path: Path) -> None:
        cq.exporters.export(
            cq.Workplane("XY")
            .box(1, 1, 1)
            .union(cq.Workplane("XY").box(1, 1, 1).translate((5, 0, 0))),
            str(path),
            exportType="STEP",
        )
        path.with_name(PROGRAM_LOG).write_text(
            '{"booleans": [], "result_volumes": [1.0, -1.0]}', encoding="utf-8"
        )

    executor, _ = _executor(_sandbox_result(), write_two_solids_and_an_inside_out_log)

    report = executor.execute(_write_model(tmp_path))

    assert report.status is ExecutionStatus.FAILED
    assert report.executor_error == (
        "1 of the 2 solids in `result` is inside out (faces pointing inward): "
        "volume -1.000 mm³.\n"
        "Expected exactly one solid, found 2"
    )
