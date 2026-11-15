import json
import sys
from pathlib import Path
from typing import Any

import cadquery as cq
import pytest
from cadquery.occ_impl.shapes import Mixin3D

from zeroshot.pipeline.verification import _run_program as runner


def _run_logged(tmp_path: Path, source: str) -> tuple[list[dict[str, Any]], dict]:
    """Run `source` as the program, with its calls logged, in this process."""
    program = tmp_path / "model.py"
    log: list[dict[str, Any]] = []
    namespace: dict[str, Any] = {}
    undo = runner._record_calls(program, log)
    try:
        exec(compile(source, str(program), "exec"), namespace)  # noqa: S102 - the program under test
    finally:
        undo()
    return log, namespace


def _run_raising(tmp_path: Path, source: str) -> tuple[list[dict[str, Any]], Exception]:
    """Run `source`, which raises, with its calls logged; return the log and the error."""
    program = tmp_path / "model.py"
    log: list[dict[str, Any]] = []
    undo = runner._record_calls(program, log)
    try:
        exec(compile(source, str(program), "exec"), {})  # noqa: S102 - the program under test
    except Exception as error:  # noqa: BLE001 - returned for the test to inspect
        return log, error
    finally:
        undo()
    raise AssertionError("the program did not raise")


def _unlogged_error(source: str) -> Exception:
    try:
        exec(compile(source, "unlogged.py", "exec"), {})  # noqa: S102 - the same program, not logged
    except Exception as error:  # noqa: BLE001 - returned for the test to compare
        return error
    raise AssertionError("the program did not raise")


def _counts(entry: dict[str, Any]) -> tuple:
    """The number of solids of an entry's body, of each tool and of its output."""

    def count(solids: list | None) -> int | None:
        return None if solids is None else len(solids)

    return (
        count(entry["body"]),
        tuple(count(tool) for tool in entry["tools"]),
        count(entry["output"]),
    )


def _given_a_tool(log: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [entry for entry in log if entry["tools"]]


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


def test_an_extrusion_and_a_cut_are_logged_with_their_solids(tmp_path: Path) -> None:
    source = (
        "import cadquery as cq\n"
        "body = cq.Workplane('XY').rect(10, 10).extrude(10)\n"
        "result = body.cut(cq.Workplane('XY').rect(2, 2).extrude(40))\n"
    )
    unlogged: dict[str, Any] = {}
    exec(compile(source, "unlogged.py", "exec"), unlogged)  # noqa: S102 - the same program, not logged

    log, namespace = _run_logged(tmp_path, source)

    extrude = log[0]
    assert (extrude["operation"], extrude["line"], extrude["caller_line"]) == (
        "extrude",
        2,
        2,
    )
    assert (extrude["body"], extrude["tools"], extrude["error"]) == (None, [], None)
    (solid,) = extrude["output"]
    assert solid["valid"] is True
    assert solid["volume"] == pytest.approx(1000.0)
    assert solid["box"] == pytest.approx([-5, -5, 0, 5, 5, 10], abs=1e-6)
    (cut,) = _given_a_tool(log)
    assert (cut["operation"], cut["line"]) == ("cut", 3)
    assert _counts(cut) == (1, (1,), 1)
    assert cut["tools"][0][0]["volume"] == pytest.approx(160.0)
    assert cut["output"][0]["volume"] == pytest.approx(960.0)
    # Logging changes nothing the program builds.
    built, expected = namespace["result"].val(), unlogged["result"].val()
    assert built.Volume() == pytest.approx(expected.Volume())
    assert len(built.Faces()) == len(expected.Faces())


def test_the_booleans_that_make_a_builders_solid_are_part_of_it(
    tmp_path: Path,
) -> None:
    source = (
        "import cadquery as cq\n"
        "result = cq.Workplane('XY').rect(4, 4).extrude(1, both=True)\n"
    )

    log, _ = _run_logged(tmp_path, source)

    assert log[0]["operation"] == "extrude"
    assert log[0]["output"][0]["volume"] == pytest.approx(32.0)
    assert _given_a_tool(log) == []


def test_an_extrusion_onto_a_body_is_logged_with_the_union_that_joins_it(
    tmp_path: Path,
) -> None:
    source = (
        "import cadquery as cq\n"
        "body = cq.Workplane('XY').box(10, 10, 2)\n"
        "result = body.faces('>Z').workplane().rect(2, 2).extrude(3)\n"
    )

    log, _ = _run_logged(tmp_path, source)

    extrude = next(entry for entry in log if entry["operation"] == "extrude")
    (fuse,) = _given_a_tool(log)
    assert (extrude["line"], fuse["operation"], fuse["line"]) == (3, "fuse", 3)
    assert fuse["tools"] == [extrude["output"]]
    assert fuse["output"][0]["volume"] == pytest.approx(212.0)


def test_the_solid_builders_a_program_calls_itself_are_logged(tmp_path: Path) -> None:
    source = (
        "import cadquery as cq\n"
        "square = cq.Wire.makePolygon([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], close=True)\n"
        "prism = cq.Solid.extrudeLinear(cq.Face.makeFromWires(square), cq.Vector(0, 0, 2))\n"
        "loft = cq.Workplane().rect(2, 2).workplane(offset=3).circle(1).loft()\n"
        "ring = cq.Workplane('XZ').moveTo(2, 0).rect(1, 1).revolve(360, (0, 0, 0), (0, 1, 0))\n"
        "result = prism\n"
    )

    log, _ = _run_logged(tmp_path, source)

    built = [
        (entry["operation"], entry["line"]) for entry in log if entry["body"] is None
    ]
    assert built == [("extrudeLinear", 3), ("loft", 4), ("revolve", 5)]
    assert log[0]["output"][0]["volume"] == pytest.approx(2.0)


def test_a_fillet_and_a_chamfer_are_logged_with_the_solid_they_work_on(
    tmp_path: Path,
) -> None:
    source = (
        "import cadquery as cq\n"
        "body = cq.Workplane('XY').box(10, 10, 10)\n"
        "body = body.edges('|Z').fillet(1)\n"
        "box = cq.Solid.makeBox(4, 4, 4)\n"
        "result = box.chamfer(0.5, None, box.Edges()[:1])\n"
    )

    log, _ = _run_logged(tmp_path, source)

    fillet, chamfer = (e for e in log if e["operation"] in ("fillet", "chamfer"))
    assert (fillet["operation"], fillet["line"], _counts(fillet)) == (
        "fillet",
        3,
        (1, (), 1),
    )
    assert fillet["body"][0]["volume"] == pytest.approx(1000.0)
    assert fillet["output"][0]["volume"] < 1000.0
    assert (chamfer["operation"], chamfer["line"], _counts(chamfer)) == (
        "chamfer",
        5,
        (1, (), 1),
    )


def test_only_a_boolean_is_logged_with_tools(tmp_path: Path) -> None:
    # A Workplane of edges given to a fillet is not a tool.
    source = (
        "import cadquery as cq\n"
        "body = cq.Workplane('XY').box(10, 10, 10)\n"
        "result = body.fillet(1, body.edges('|Z'))\n"
    )

    log, error = _run_raising(tmp_path, source)

    assert isinstance(error, TypeError)
    fillet = log[-1]
    assert (fillet["operation"], _counts(fillet)) == ("fillet", (1, (), None))


def test_an_invalid_extrusion_is_logged_as_invalid(tmp_path: Path) -> None:
    # The wire passes OCCT's check; the face and the solid made from it do not.
    source = (
        "import cadquery as cq\n"
        "bowtie = cq.Workplane('XY').polyline([(0, 0), (10, 10), (10, 0), (0, 10)]).close()\n"
        "result = bowtie.extrude(2)\n"
    )

    log, _ = _run_logged(tmp_path, source)

    (solid,) = log[0]["output"]
    assert solid["valid"] is False


def test_an_empty_intersection_has_no_solids_and_no_box(tmp_path: Path) -> None:
    source = (
        "import cadquery as cq\n"
        "body = cq.Workplane('XY').box(2, 2, 2)\n"
        "result = body.intersect(cq.Workplane('XY').box(2, 2, 2).translate((50, 0, 0)))\n"
    )

    log, _ = _run_logged(tmp_path, source)

    (intersect,) = _given_a_tool(log)
    # Measured, and empty: not the None of solids that could not be measured.
    assert (intersect["operation"], intersect["output"]) == ("intersect", [])
    assert _counts(intersect) == (1, (1,), 0)


def test_each_solid_of_a_separated_result_is_logged(tmp_path: Path) -> None:
    source = (
        "import cadquery as cq\n"
        "body = cq.Workplane('XY').box(2, 2, 2)\n"
        "result = body.union(cq.Workplane('XY').box(1, 1, 1).translate((10, 0, 0)))\n"
    )

    log, _ = _run_logged(tmp_path, source)

    (union,) = _given_a_tool(log)
    assert union["operation"] == "union"
    assert sorted(solid["volume"] for solid in union["output"]) == pytest.approx(
        [1.0, 8.0]
    )
    assert sorted(solid["box"][0] for solid in union["output"]) == pytest.approx(
        [-1.0, 9.5]
    )


def test_an_inside_out_solid_keeps_its_negative_volume_beside_its_twin(
    tmp_path: Path,
) -> None:
    # The same solid in the other orientation is measured on its own, not reused.
    source = (
        "import cadquery as cq\n"
        "box = cq.Solid.makeBox(10, 10, 10)\n"
        "inverted = cq.Solid(box.wrapped.Reversed())\n"
        "outward = box.cut(cq.Solid.makeBox(1, 1, 1, pnt=cq.Vector(50, 0, 0)))\n"
        "inward = inverted.cut(cq.Solid.makeBox(1, 1, 1, pnt=cq.Vector(50, 0, 0)))\n"
        "result = box\n"
    )

    log, _ = _run_logged(tmp_path, source)

    assert [entry["body"][0]["volume"] for entry in log] == pytest.approx(
        [1000.0, -1000.0]
    )


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
    (entry,) = log
    assert entry["body"][0]["volume"] is None
    assert entry["output"][0]["volume"] is None
    # The rest is still measured.
    assert entry["output"][0]["valid"] is True
    assert entry["output"][0]["box"] == pytest.approx([0, 0, 0, 2, 2, 2], abs=1e-6)


def test_a_shape_whose_solids_cannot_be_listed_is_logged_unmeasured(
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
    (entry,) = log
    assert (entry["operation"], entry["line"]) == ("cut", 2)
    assert _counts(entry) == (None, (None,), None)


def test_a_boolean_that_raises_is_logged_with_its_body_and_error(
    tmp_path: Path,
) -> None:
    source = "import cadquery as cq\nresult = cq.Solid.makeBox(1, 1, 1).cut(None)\n"

    log, error = _run_raising(tmp_path, source)

    unlogged = _unlogged_error(source)
    assert (type(error), str(error)) == (type(unlogged), str(unlogged))
    (entry,) = log
    assert (entry["operation"], entry["line"]) == ("cut", 2)
    assert entry["body"][0]["volume"] == pytest.approx(1.0)
    # A tool that is no shape is not one of its tools.
    assert (entry["tools"], entry["output"]) == ([], None)
    assert entry["error"] == f"AttributeError: {error}"


def test_a_union_refused_for_an_empty_tool_is_logged_with_that_tool(
    tmp_path: Path,
) -> None:
    # The Workplane checks the tool before any shape boolean runs.
    source = (
        "import cadquery as cq\n"
        "body = cq.Workplane('XY').box(2, 2, 2)\n"
        "empty = cq.Workplane('XY').box(1, 1, 1).intersect(\n"
        "    cq.Workplane('XY').box(1, 1, 1).translate((9, 0, 0))\n"
        ")\n"
        "result = body.union(empty)\n"
    )

    log, error = _run_raising(tmp_path, source)

    unlogged = _unlogged_error(source)
    assert (type(error), str(error)) == (type(unlogged), str(unlogged))
    union = log[-1]
    assert (union["operation"], union["line"]) == ("union", 6)
    # The body has its solid; the tool is the one with none.
    assert _counts(union) == (1, (0,), None)
    assert union["error"] == (
        "ValueError: Workplane object must have at least one solid on the stack "
        "to union!"
    )


def test_a_cut_with_no_solid_to_cut_is_logged_with_an_empty_body(
    tmp_path: Path,
) -> None:
    source = (
        "import cadquery as cq\n"
        "result = cq.Workplane('XY').cut(cq.Workplane('XY').box(1, 1, 1))\n"
    )

    log, error = _run_raising(tmp_path, source)

    assert isinstance(error, ValueError)
    cut = log[-1]
    assert (cut["operation"], _counts(cut)) == ("cut", (0, (1,), None))
    assert cut["error"].startswith("ValueError: Cannot find a solid")


def test_a_generator_given_as_a_tool_is_not_consumed(tmp_path: Path) -> None:
    source = (
        "import cadquery as cq\n"
        "tools = (cq.Solid.makeBox(1, 1, 1) for _ in range(2))\n"
        "body = cq.Workplane('XY').box(2, 2, 2)\n"
        "try:\n"
        "    body.union(tools)\n"
        "except ValueError:\n"
        "    pass\n"
        "result = next(tools)\n"
    )

    log, namespace = _run_logged(tmp_path, source)

    assert isinstance(namespace["result"], cq.Solid)
    assert log[-1]["operation"] == "union"
    assert log[-1]["error"].startswith("ValueError: Cannot union type")


def test_the_patched_methods_are_restored_after_a_program_that_raised(
    tmp_path: Path,
) -> None:
    module = sys.modules[cq.Workplane.__module__]
    patched = [
        (cq.Shape, "fuse"),
        (cq.Compound, "cut"),
        (Mixin3D, "fillet"),
        (cq.Workplane, "union"),
        (cq.Workplane, "_extrude"),
        (module, "loft"),
        (cq.Solid, "makeLoft"),
    ]
    before = {(owner, name): vars(owner)[name] for owner, name in patched}

    _run_raising(tmp_path, "raise RuntimeError('stop')\n")

    assert all(vars(owner)[name] is fn for (owner, name), fn in before.items())


def _main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str) -> int:
    monkeypatch.chdir(tmp_path)
    program = tmp_path / "model.py"
    program.write_text(source, encoding="utf-8")
    return runner.main(["_run_program.py", str(program)])


def _written_log(tmp_path: Path) -> dict[str, Any]:
    return json.loads((tmp_path / runner.PROGRAM_LOG).read_text(encoding="utf-8"))


def test_a_program_that_raises_leaves_the_calls_it_made(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    source = (
        "import cadquery as cq\n"
        "body = cq.Solid.makeBox(2, 2, 2).cut(cq.Solid.makeBox(1, 1, 1))\n"
        "raise RuntimeError('after the cut')\n"
    )

    assert _main(tmp_path, monkeypatch, source) == 1

    log = _written_log(tmp_path)
    assert (log["completed"], log["result_volumes"]) == (False, None)
    (cut,) = log["calls"]
    assert (cut["operation"], cut["error"]) == ("cut", None)
    assert cut["output"][0]["volume"] == pytest.approx(7.0)
    stderr = capsys.readouterr().err
    assert "RuntimeError: after the cut" in stderr
    assert "_run_program.py" not in stderr


def test_a_program_without_a_result_still_leaves_its_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    source = "import cadquery as cq\nbody = cq.Workplane('XY').rect(1, 1).extrude(1)\n"

    assert _main(tmp_path, monkeypatch, source) == 1

    log = _written_log(tmp_path)
    assert (log["completed"], log["result_volumes"]) == (True, None)
    assert log["calls"][0]["operation"] == "extrude"
    assert "`result` not defined" in capsys.readouterr().err


def test_a_result_that_fails_to_export_still_leaves_its_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    source = (
        "import cadquery as cq\n"
        "box = cq.Workplane('XY').rect(1, 1).extrude(1)\n"
        "result = 5\n"
    )

    assert _main(tmp_path, monkeypatch, source) == 1

    log = _written_log(tmp_path)
    assert (log["completed"], log["result_volumes"]) == (True, None)
    assert log["calls"][0]["operation"] == "extrude"
    assert "Failed to export `result` to STEP:" in capsys.readouterr().err
    assert not (tmp_path / runner.RESULT_STEP).exists()


def test_a_program_that_does_not_compile_is_reported_before_logging_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    started = []
    monkeypatch.setattr(runner, "_record_calls", lambda *args: started.append(args))

    assert _main(tmp_path, monkeypatch, "result = (\n") == 1
    assert started == []
    assert "SyntaxError" in capsys.readouterr().err
    assert not (tmp_path / runner.PROGRAM_LOG).exists()
