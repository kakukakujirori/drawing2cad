"""Scoring a native run, which leaves a program rather than a verified STEP."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.errors import InstantiationException
from PIL import Image

import zeroshot.pipeline_native.__main__ as entry
from zeroshot.evaluation.aggregate_run import Terminal, Tokens, collect, summarize
from zeroshot.evaluation.run_scoring import (
    NATIVE_BUILD_DIRNAME,
    StepScorer,
    is_native_run,
    score_run,
)
from zeroshot.pipeline_native import runner
from zeroshot.pipeline_native.event_logging import EventLog

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

    def build(program: str | None, name: str = "run", **sandbox: object) -> Path:
        run_dir = tmp_path / name
        (run_dir / "workspace").mkdir(parents=True)
        (run_dir / "events.jsonl").touch()
        config = {"sandbox_runner": sandbox}
        (run_dir / "run.json").write_text(
            json.dumps({"pipeline": "native", "config": config})
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


def test_a_native_run_recorded_before_its_marker_is_still_native(
    native_run,
) -> None:
    """Earlier runs said so only through `workflow.name` in their config."""
    run_dir = native_run(_BOX)
    (run_dir / "run.json").write_text(
        json.dumps({"config": {"workflow": {"name": "native"}}})
    )

    assert is_native_run(run_dir)


def test_a_native_build_gets_the_time_its_run_gave_the_agent(
    native_run, solids
) -> None:
    program = "import time\n" + _BOX + "time.sleep(3)\n"
    run_dir = native_run(program, default_timeout_s=1.0)
    document = score_run(run_dir, solids["box"], StepScorer(**_FAST))

    assert document["native_build_status"] == "TIMEOUT"
    assert document["status"] == "NO_PREDICTION"


def test_a_sandbox_that_cannot_run_is_the_evaluators_failure(
    native_run, solids, monkeypatch
) -> None:
    """Not the model's: the scorer reports it, rather than a zero for the arm."""
    from zeroshot.pipeline.sandbox import SandboxResult, SandboxRunner, SandboxStatus

    monkeypatch.setattr(
        SandboxRunner,
        "run",
        lambda self, **kwargs: SandboxResult(SandboxStatus.INFRA_ERROR, None, "", "x"),
    )
    document = score_run(native_run(_BOX), solids["box"], StepScorer(**_FAST))

    assert document["native_build_status"] == "INFRA_ERROR"
    assert document["status"] == "FAILED"
    assert document["errors"]["build"]["kind"] == "evaluation_error"


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


_USAGE = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}


def test_aggregation_reads_a_native_run(native_run, tmp_path) -> None:
    """A streamed call counts once, by its finish; a whole message counts once."""
    run_dir = native_run(_BOX, name="000364")
    _native_log(
        run_dir,
        ("run_started", {}, 1_000),
        ("messages", [{"event": "message-start"}, {}], 1_100),
        ("messages", [{"event": "content-block-delta", "index": 0}, {}], 1_200),
        ("messages", [{"event": "message-finish", "usage": _USAGE}, {}], 1_300),
        ("tools", {"event": "tool-started", "tool_name": "run_shell"}, 1_400),
        ("tools", {"event": "tool-finished", "tool_call_id": "a"}, 1_500),
        ("messages", [{"type": "ai", "usage_metadata": _USAGE}, {}], 1_600),
        ("run_completed", {"model_path": "workspace/model.py"}, 61_000),
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


def test_aggregation_counts_a_failed_native_run_that_left_no_transcript(
    native_run, tmp_path
) -> None:
    """A run that raised writes no messages.json; its spending is in the log."""
    run_dir = native_run(None)
    _native_log(
        run_dir,
        ("run_started", {}, 0),
        ("messages", [{"event": "message-finish", "usage": _USAGE}, {}], 10),
        ("run_failed", {"error": "usage limit"}, 20),
    )

    (row,) = collect(tmp_path)
    assert row.terminal is Terminal.FAILED
    assert (row.model_calls, row.tokens.input) == (1, 10)


def test_aggregation_reads_claude_usage_left_in_response_metadata(
    native_run, tmp_path
) -> None:
    """Claude runs from before usage_metadata kept Anthropic's counts, which
    leave cache reads and writes out of input_tokens."""
    run_dir = native_run(None)
    usage = {
        "input_tokens": 10,
        "cache_read_input_tokens": 80,
        "cache_creation_input_tokens": 5,
        "output_tokens": 7,
        "output_tokens_details": {"thinking_tokens": 4},
    }
    finish = {"event": "message-finish", "metadata": {"usage": usage}}
    _native_log(
        run_dir,
        ("run_started", {}, 0),
        ("messages", [finish, {}], 10),
        ("run_failed", {"error": "usage limit"}, 20),
    )

    (row,) = collect(tmp_path)
    assert row.calls_with_usage == 1
    assert row.tokens == Tokens(
        input=95, output=7, reasoning=4, cache_read=80, cache_creation=5
    )


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


def _entrypoint_config(tmp_path: Path, target: Path | None, *overrides: str):
    drawing = tmp_path / "drawing.png"
    Image.new("RGB", (16, 16), "white").save(drawing)
    with initialize_config_dir(
        config_dir=str(REPO_ROOT / "zeroshot/configs"), version_base="1.3"
    ):
        return compose(
            config_name="native",
            overrides=[
                f"artifact_root={tmp_path}",
                "sample.sample_id=000364",
                f"sample.drawing.sheets.0.file={drawing}",
                f"sample.target_step_path={'null' if target is None else target}",
                "evaluation.scorer.include_eccv=false",
                "evaluation.scorer.include_ortho2cad=false",
                *overrides,
            ],
        )


def _earlier_run_with_score(run_dir: Path, terminal: str) -> str:
    """A sample an earlier run ended and scored; returns its score.json."""
    run_dir.mkdir(parents=True)
    with EventLog(run_dir / "events.jsonl") as log:
        log.write("run_started", {"run_id": "earlier"})
        log.write(terminal, {})
    (run_dir / "score.json").write_text('{"status": "EARLIER"}')
    return (run_dir / "score.json").read_text()


def test_the_native_entrypoint_scores_what_it_generated(
    native_run, solids, tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(entry, "run", lambda config: native_run(_BOX, "000364"))

    entry.main(_entrypoint_config(tmp_path, solids["box"]))

    document = json.loads((tmp_path / "000364" / "score.json").read_text())
    assert document["status"] == "OK"
    assert document["metrics"]["mesh_iou"] == pytest.approx(1.0, abs=1e-3)
    assert "  mesh_iou" in capsys.readouterr().out


def test_a_native_run_that_raised_is_scored_then_reraised(
    native_run, solids, tmp_path, monkeypatch
) -> None:
    def crash(config):
        native_run(None, "000364")
        raise RuntimeError("stream dropped")

    monkeypatch.setattr(entry, "run", crash)

    with pytest.raises(RuntimeError, match="stream dropped"):
        entry.main(_entrypoint_config(tmp_path, solids["box"]))
    document = json.loads((tmp_path / "000364" / "score.json").read_text())
    assert document["status"] == "NO_PREDICTION"


def test_a_scoring_error_does_not_hide_why_the_run_failed(
    native_run, solids, tmp_path, monkeypatch
) -> None:
    def crash(config):
        native_run(None, "000364")
        raise RuntimeError("stream dropped")

    def broken_scorer(*args):
        raise ValueError("scorer bug")

    monkeypatch.setattr(entry, "run", crash)
    monkeypatch.setattr(entry, "score_run", broken_scorer)

    with pytest.raises(RuntimeError, match="stream dropped"):
        entry.main(_entrypoint_config(tmp_path, solids["box"]))


@pytest.mark.parametrize(
    ("terminal", "on_existing", "refused"),
    [
        ("run_completed", "fail", True),
        ("run_completed", "skip", False),
        ("run_completed", "retry", False),
        ("run_failed", "fail", True),
        ("run_failed", "skip", True),
    ],
)
def test_a_sample_the_runner_does_not_start_keeps_its_earlier_score(
    solids, tmp_path, monkeypatch, terminal, on_existing, refused
) -> None:
    """Refused or skipped, the sample's directory is an earlier run's."""
    before = _earlier_run_with_score(tmp_path / "000364", terminal)
    monkeypatch.setattr(runner, "instantiate", pytest.fail)
    config = _entrypoint_config(tmp_path, solids["box"], f"on_existing={on_existing}")

    if refused:
        with pytest.raises(FileExistsError):
            entry.main(config)
    else:
        entry.main(config)
    assert (tmp_path / "000364" / "score.json").read_text() == before


@pytest.mark.parametrize(
    ("override", "error"),
    [
        ("sample.target_step_path=/nonexistent/000364.step", FileNotFoundError),
        ("evaluation.scorer.timeout_s=0", InstantiationException),
    ],
    ids=["missing_target", "bad_scorer"],
)
def test_a_bad_target_or_scorer_fails_before_generation(
    solids, tmp_path, monkeypatch, override, error
) -> None:
    monkeypatch.setattr(entry, "run", pytest.fail)

    with pytest.raises(error):
        entry.main(_entrypoint_config(tmp_path, solids["box"], override))
    assert not (tmp_path / "000364").exists()


def test_an_untargeted_native_run_is_not_scored(
    native_run, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(entry, "run", lambda config: native_run(_BOX, "000364"))

    entry.main(_entrypoint_config(tmp_path, None))

    assert (tmp_path / "000364" / "workspace/model.py").is_file()
    assert not (tmp_path / "000364" / "score.json").exists()
