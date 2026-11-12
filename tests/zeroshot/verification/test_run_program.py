from pathlib import Path
from typing import Any

import cadquery as cq
import pytest

from zeroshot.pipeline.verification import _run_program as runner


def _run_logged(tmp_path: Path, source: str) -> tuple[list[dict[str, Any]], dict]:
    """Run `source` as the program, with the booleans logged, in this process."""
    program = tmp_path / "model.py"
    log: list[dict[str, Any]] = []
    namespace: dict[str, Any] = {}
    undo = runner._record_booleans(program, log)
    try:
        exec(compile(source, str(program), "exec"), namespace)  # noqa: S102 - the program under test
    finally:
        undo()
    return log, namespace


def test_a_boolean_called_inside_another_is_logged_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cq.Compound, "cut", lambda self, *tools, **kw: cq.Shape.cut(self, *tools, **kw)
    )
    source = (
        "import cadquery as cq\n"
        "body = cq.Compound.makeCompound([cq.Solid.makeBox(2, 2, 2)])\n"
        "result = body.cut(cq.Solid.makeBox(1, 1, 1))\n"
    )

    log, _ = _run_logged(tmp_path, source)

    assert [(entry["operation"], entry["line"]) for entry in log] == [("cut", 3)]


def test_a_failed_measurement_leaves_the_call_and_its_result_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unmeasurable(self: cq.Shape) -> float:
        raise RuntimeError("no volume")

    monkeypatch.setattr(cq.Shape, "Volume", unmeasurable)
    source = (
        "import cadquery as cq\n"
        "result = cq.Solid.makeBox(2, 2, 2).cut(cq.Solid.makeBox(1, 1, 1))\n"
    )

    log, namespace = _run_logged(tmp_path, source)

    assert len(namespace["result"].Faces()) == 9
    assert log[0]["volume_before"] is None
    assert log[0]["volume_after"] is None


def test_a_failure_while_logging_drops_only_the_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(self: cq.Shape) -> list:
        raise RuntimeError("no solids")

    monkeypatch.setattr(cq.Shape, "Solids", broken)
    source = (
        "import cadquery as cq\n"
        "result = cq.Solid.makeBox(2, 2, 2).cut(cq.Solid.makeBox(1, 1, 1))\n"
    )

    log, namespace = _run_logged(tmp_path, source)

    assert len(namespace["result"].Faces()) == 9
    assert log == []


def test_an_error_inside_a_boolean_propagates_unchanged(tmp_path: Path) -> None:
    source = "import cadquery as cq\nresult = cq.Solid.makeBox(1, 1, 1).cut(None)\n"

    with pytest.raises(AttributeError, match="wrapped"):
        _run_logged(tmp_path, source)


def test_the_booleans_are_restored_after_the_run(tmp_path: Path) -> None:
    before = {
        (cls, name): cls.__dict__[name]
        for cls in (cq.Shape, cq.Compound)
        for name in ("cut", "fuse", "intersect")
    }

    _run_logged(tmp_path, "result = None\n")

    assert all(cls.__dict__[name] is fn for (cls, name), fn in before.items())


def test_a_program_that_does_not_compile_is_reported_before_logging_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    started = []
    monkeypatch.setattr(runner, "_record_booleans", lambda *args: started.append(args))
    monkeypatch.chdir(tmp_path)
    program = tmp_path / "model.py"
    program.write_text("result = (\n", encoding="utf-8")

    assert runner.main(["_run_program.py", str(program)]) == 1
    assert started == []
    assert "SyntaxError" in capsys.readouterr().err
    assert not (tmp_path / runner.PROGRAM_LOG).exists()
