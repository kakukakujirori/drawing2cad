"""Scoring a native run, which leaves a program rather than a verified STEP."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from zeroshot.evaluation.aggregate_run import Terminal, collect, summarize
from zeroshot.evaluation.run_scoring import (
    NATIVE_BUILD_DIRNAME,
    StepScorer,
    is_native_run,
    score_run,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
# Supervision checks need no ECCV or Ortho2CAD columns.
_FAST = {"include_eccv": False, "include_ortho2cad": False}
_BOX = 'import cadquery as cq\nresult = cq.Workplane("XY").box(30, 20, 10)\n'
_TWO_BOXES = (
    "import cadquery as cq\n"
    'result = cq.Workplane("XY").box(1, 1, 1)'
    '.union(cq.Workplane("XY").box(1, 1, 1).translate((5, 0, 0)))\n'
)


@pytest.fixture
def native_run(tmp_path: Path):
    """A sample directory as `zeroshot.pipeline_native` leaves it."""

    def build(program: str | None, name: str = "run") -> Path:
        run_dir = tmp_path / name
        (run_dir / "workspace").mkdir(parents=True)
        (run_dir / "events.jsonl").touch()
        (run_dir / "run.json").write_text(
            json.dumps({"config": {"workflow": {"name": "native"}}})
        )
        if program is not None:
            (run_dir / "workspace/model.py").write_text(program)
        return run_dir

    return build


def test_a_native_program_is_built_and_scored(native_run, solids) -> None:
    run_dir = native_run(_BOX)
    document = score_run(run_dir, solids["box"], StepScorer(**_FAST))

    assert document["status"] == "OK"
    assert document["native_build_status"] == "VERIFIED"
    assert document["pred_step"] == str(run_dir / NATIVE_BUILD_DIRNAME / "output.step")
    assert document["metrics"]["mesh_iou"] == pytest.approx(1.0, abs=1e-3)
    # The workspace stays as the agent left it.
    assert sorted(p.name for p in (run_dir / "workspace").iterdir()) == ["model.py"]


def test_a_native_program_may_catch_exceptions(native_run, solids) -> None:
    """The staged coder is told not to; the native agent never was."""
    program = (
        "import cadquery as cq\n"
        "try:\n"
        '    result = cq.Workplane("XY").box(30, 20, 10).edges("|Z").fillet(50)\n'
        "except Exception:\n"
        '    result = cq.Workplane("XY").box(30, 20, 10)\n'
    )
    document = score_run(native_run(program), solids["box"], StepScorer(**_FAST))
    assert document["status"] == "OK"


@pytest.mark.parametrize(
    ("program", "reason"),
    [
        # What the native runner writes when the turn limit ran out first.
        ("import cadquery as cq\nresult = None\n", "FAILED"),
        # The staged attempts are held to exactly one solid, and so is this.
        (_TWO_BOXES, "Expected exactly one solid"),
    ],
)
def test_a_native_program_without_one_solid_is_no_prediction(
    native_run, solids, program, reason
) -> None:
    document = score_run(native_run(program), solids["box"], StepScorer(**_FAST))
    assert document["status"] == "NO_PREDICTION"
    assert document["pred_step"] is None
    assert document["native_build_status"] == "FAILED"
    assert reason in document["skipped_attempts"]["workspace/model.py"]


def test_a_native_run_without_a_program_is_no_prediction(native_run, solids) -> None:
    document = score_run(native_run(None), solids["box"], StepScorer(**_FAST))
    assert document["status"] == "NO_PREDICTION"
    assert document["native_build_status"] is None
    assert document["skipped_attempts"] == {}


def test_a_staged_program_that_never_verified_is_not_built(
    make_run_dir, solids
) -> None:
    """A staged run with no attempt submitted nothing, whatever model.py holds."""
    run_dir = make_run_dir({})
    (run_dir / "workspace/model.py").write_text(_BOX)

    assert not is_native_run(run_dir)
    document = score_run(run_dir, solids["box"], StepScorer(**_FAST))
    assert document["status"] == "NO_PREDICTION"
    assert "native_build_status" not in document
    assert not (run_dir / NATIVE_BUILD_DIRNAME).exists()


def test_the_cli_scores_a_native_run(native_run, solids, tmp_path) -> None:
    run_dir = native_run(_BOX, name="000364")
    targets = tmp_path / "targets"
    targets.mkdir()
    (targets / "000364.step").write_bytes(solids["box"].read_bytes())

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "zeroshot.evaluation.run_scoring",
            "--run-dir",
            str(run_dir),
            "--target-dir",
            str(targets),
            "--skip-eccv",
            "--skip-ortho2cad",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads((run_dir / "score.json").read_text())["status"] == "OK"


def _native_log(run_dir: Path, *events: tuple[str, dict, int]) -> None:
    with (run_dir / "events.jsonl").open("w") as handle:
        for index, (name, data, at) in enumerate(events):
            record = {"timestamp": at, "schema_version": 1, "event_index": index}
            handle.write(json.dumps({**record, "event": name, "data": data}) + "\n")


def test_aggregation_reads_a_native_run(native_run, tmp_path) -> None:
    """Calls and tokens come from the transcript, not from streamed deltas."""
    run_dir = native_run(_BOX, name="000364")
    _native_log(
        run_dir,
        ("run_started", {}, 1_000),
        ("messages", [{"event": "delta"}, {}], 1_500),
        ("run_completed", {"model_path": "workspace/model.py"}, 61_000),
    )
    usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    (run_dir / "messages.json").write_text(
        json.dumps(
            [
                {"type": "human", "content": "drawing"},
                {
                    "type": "ai",
                    "content": "",
                    "tool_calls": [{"name": "run_shell", "args": {}, "id": "a"}],
                    "usage_metadata": usage,
                },
                {"type": "tool", "content": "ok"},
                {"type": "ai", "content": "Saved.", "usage_metadata": usage},
            ]
        )
    )
    (run_dir / "score.json").write_text(
        json.dumps({"status": "OK", "native_build_status": "VERIFIED"})
    )

    (row,) = collect(tmp_path)
    assert row.terminal is Terminal.COMPLETED
    assert row.verified
    assert row.wall_ms == 60_000
    assert (row.model_calls, row.calls_with_usage) == (2, 2)
    assert (row.tokens.input, row.tokens.output) == (20, 10)
    assert row.tool_calls == {"run_shell:generator": 1}


def test_aggregation_counts_a_native_turn_limit_as_budget_exhausted(
    native_run, tmp_path
) -> None:
    run_dir = native_run("import cadquery as cq\nresult = None\n")
    _native_log(
        run_dir,
        ("run_started", {}, 0),
        ("turn_limit_exceeded", {"max_turns": 150}, 10),
        ("run_completed", {"status": "turn_limit_exceeded"}, 20),
    )
    (run_dir / "score.json").write_text(
        json.dumps({"status": "NO_PREDICTION", "native_build_status": "FAILED"})
    )

    rows = collect(tmp_path)
    assert rows[0].stop_reasons == {"generator": "BUDGET_EXHAUSTED"}
    assert not rows[0].verified
    assert summarize(rows).budget_exhausted == 1


def test_aggregation_survives_a_native_run_killed_mid_stream(
    native_run, tmp_path
) -> None:
    run_dir = native_run(None)
    _native_log(run_dir, ("run_started", {}, 0))
    with (run_dir / "events.jsonl").open("a") as handle:
        handle.write('{"event": "messages", "data":')

    (row,) = collect(tmp_path)
    assert row.terminal is Terminal.INTERRUPTED
    assert row.model_calls == 0
