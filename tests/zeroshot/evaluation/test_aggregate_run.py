import json
import sys
from pathlib import Path
from typing import Any

from zeroshot.evaluation import aggregate_run
from zeroshot.evaluation.aggregate_run import (
    SampleRow,
    Terminal,
    collect,
    format_report,
    headline_columns,
    notes,
    read_events,
    summarize,
)
from zeroshot.evaluation.run_scoring import StepScorer

FAMILIES = list(StepScorer().families())


def _event(name: str, timestamp_ms: int = 0, **data: Any) -> dict[str, Any]:
    return {"event": name, "timestamp_ms": timestamp_ms, "data": data}


def _ai(usage: dict[str, Any] | None = None) -> dict[str, Any]:
    return _event(
        "message", node="model", messages=[{"type": "ai", "usage_metadata": usage}]
    )


def _write_sample(
    run_dir: Path,
    sample_id: str,
    events: list[dict[str, Any]],
    score: dict[str, Any] | None = None,
) -> Path:
    sample_dir = run_dir / sample_id
    (sample_dir / ".hydra").mkdir(parents=True)
    (sample_dir / ".hydra" / "config.yaml").write_text("a: 1\n", encoding="utf-8")
    (sample_dir / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    if score is not None:
        (sample_dir / "score.json").write_text(json.dumps(score), encoding="utf-8")
    return sample_dir


def _completed_events(usage: dict[str, Any] | None) -> list[dict[str, Any]]:
    return [
        _event("run_started"),
        _event("node_started", timestamp_ms=1000, node="model"),
        _ai(usage),
        _event("node_finished", timestamp_ms=3000, node="model"),
        _event("tool_started", tool_name="run_shell", caller="model"),
        _event("node_started", timestamp_ms=3000, node="verify_final"),
        _event("verification", node="verify_final", report={"status": "VERIFIED"}),
        _event("stop_reason", node="verify_final", reason="COMPLETED"),
        _event("node_finished", timestamp_ms=3500, node="verify_final"),
        _event("run_completed", duration_ms=4000),
    ]


_USAGE = {
    "input_tokens": 100,
    "output_tokens": 20,
    "input_token_details": {"cache_read": 64},
    "output_token_details": {"reasoning": 8},
}


def test_read_events_reports_what_the_run_did(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text(
        "".join(json.dumps(e) + "\n" for e in _completed_events(_USAGE)),
        encoding="utf-8",
    )

    row = read_events(path, "000364")

    assert row.sample_id == "000364"
    assert row.terminal is Terminal.COMPLETED
    assert row.verify == "VERIFIED"
    assert row.stop_reason == "COMPLETED"
    assert row.agent_turns == 1
    assert row.model_calls == 1
    assert row.tool_calls == {"run_shell:model": 1}
    assert row.tokens.input == 100
    assert row.tokens.cache_read == 64
    assert row.tokens.reasoning == 8
    assert row.wall_ms == 4000
    assert row.node_ms == {"model": 2000, "verify_final": 500}


def test_a_backend_that_reports_no_usage_is_not_recorded_as_free(
    tmp_path: Path,
) -> None:
    """Ollama returns none at all; a zero would read as a free run."""
    path = tmp_path / "events.jsonl"
    path.write_text(
        "".join(json.dumps(e) + "\n" for e in _completed_events(None)),
        encoding="utf-8",
    )

    row = read_events(path, "quiet")

    assert row.model_calls == 1
    assert row.calls_with_usage == 0
    assert row.tokens.input is None
    assert row.tokens.output is None


def test_a_crashed_sample_is_a_row_not_an_omission(tmp_path: Path) -> None:
    """Dropping it would let every rate flatter itself by shrinking its own
    denominator."""
    _write_sample(tmp_path, "ok", _completed_events(_USAGE), {"status": "OK"})
    _write_sample(
        tmp_path,
        "crashed",
        [
            _event("run_started"),
            _ai(_USAGE),
            _event("run_failed", duration_ms=900, error="APIConnectionError"),
        ],
    )

    rows = collect(tmp_path)
    summary = summarize(rows)

    assert [row.sample_id for row in rows] == ["crashed", "ok"]
    assert rows[0].terminal is Terminal.FAILED
    assert rows[0].error == "APIConnectionError"
    assert rows[0].verify is None
    assert summary.samples == 2
    assert summary.execution_success == 1
    assert summary.terminal == {"failed": 1, "completed": 1}


def test_the_two_metric_means_answer_different_questions(tmp_path: Path) -> None:
    """One is shape quality, the other is the system's score. Only reporting
    `mean_scored` would hide that half the samples produced nothing."""
    _write_sample(
        tmp_path,
        "scored",
        _completed_events(_USAGE),
        {"status": "OK", "metrics": {"voxel_iou": 0.5}},
    )
    _write_sample(
        tmp_path,
        "empty",
        _completed_events(_USAGE),
        {"status": "NO_PREDICTION", "metrics": {}},
    )

    summary = summarize(collect(tmp_path))

    assert summary.scored == 1
    assert summary.metrics["voxel_iou"].scored == 0.5
    assert summary.metrics["voxel_iou"].overall == 0.25


def test_a_directory_without_events_is_not_a_sample(tmp_path: Path) -> None:
    _write_sample(tmp_path, "ok", _completed_events(_USAGE), {"status": "OK"})
    (tmp_path / "multirun.yaml").write_text("x: 1\n", encoding="utf-8")
    (tmp_path / "stray").mkdir()

    assert [row.sample_id for row in collect(tmp_path)] == ["ok"]


def test_notes_carry_the_exceptions_the_table_leaves_out(tmp_path: Path) -> None:
    budget = _completed_events(_USAGE)
    budget[7] = _event("stop_reason", node="verify_final", reason="BUDGET_EXHAUSTED")
    _write_sample(tmp_path, "budget", budget, {"status": "OK"})
    _write_sample(tmp_path, "slow", _completed_events(_USAGE), {"status": "TIMEOUT"})
    _write_sample(tmp_path, "quiet", _completed_events(None), {"status": "OK"})

    rows = collect(tmp_path)
    lines = notes(rows)

    assert any("BUDGET_EXHAUSTED" in line for line in lines)
    assert any("score TIMEOUT" in line for line in lines)
    assert any("no token usage" in line for line in lines)
    assert summarize(rows).budget_exhausted == 1


def test_the_table_stays_narrow_enough_to_read(tmp_path: Path) -> None:
    _write_sample(
        tmp_path,
        "000364",
        _completed_events(_USAGE),
        {"status": "OK", "metrics": {"eccv_surface_f1": 0.04, "voxel_iou": 0.113}},
    )

    rows = collect(tmp_path)
    report = format_report(rows, summarize(rows), headline_columns(rows, FAMILIES))
    table = [
        line for line in report.splitlines() if line.startswith(("sample", "000364"))
    ]

    assert len(table[0].split()) == 7
    assert all(len(line) < 80 for line in report.splitlines())
    assert "0.040" in table[1]


def test_a_family_is_represented_by_the_column_it_returned_first() -> None:
    """Not by the alphabetically first one, and not by a list kept here."""
    rows = [
        SampleRow(
            sample_id="s",
            terminal=Terminal.COMPLETED,
            metrics={"zeta_recall": 0.1, "zeta_f1": 0.2, "iou_score": 0.3},
        )
    ]

    assert headline_columns(rows, ["zeta", "iou"]) == ["zeta_recall", "iou_score"]


def test_a_family_that_produced_no_column_takes_none() -> None:
    """A family can fail on its own without emptying the table."""
    rows = [
        SampleRow(
            sample_id="s", terminal=Terminal.COMPLETED, metrics={"voxel_iou": 0.3}
        )
    ]

    assert headline_columns(rows, ["eccv", "voxel"]) == ["voxel_iou"]


def test_the_scorer_decides_which_metrics_the_table_leads_with(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """A second list of metric names here would drift the moment a family does.

    Adding, renaming or reordering a family has to move the column with no edit
    in this module.
    """
    _write_sample(
        tmp_path,
        "000364",
        _completed_events(_USAGE),
        {"status": "OK", "metrics": {"renamed_iou": 0.25}},
    )

    class OneFamilyScorer:
        def families(self) -> dict[str, Any]:
            return {"renamed": object()}

    monkeypatch.setattr(aggregate_run, "StepScorer", OneFamilyScorer)
    monkeypatch.setattr(sys, "argv", ["aggregate_run", "--run-dir", str(tmp_path)])

    aggregate_run.main()

    header = capsys.readouterr().out.splitlines()[0]
    assert "renamed_iou" in header
    assert not any(name in header for name in FAMILIES)
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))


def test_no_metric_name_is_written_into_this_module() -> None:
    """The scorer owns them; a literal here is a duplicate by definition."""
    source = Path(aggregate_run.__file__).read_text(encoding="utf-8")

    assert not [name for name in FAMILIES if f"{name}_" in source]
