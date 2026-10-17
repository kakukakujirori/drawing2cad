"""Join what a run recorded about itself with what scoring said about it.

`events.jsonl` describes every attempted sample, crashes included, and needs no
ground truth. `score.json` records evaluation and generation failures wherever
a target was configured. Neither alone says whether the pipeline works.

    python -m zeroshot.evaluation.aggregate_run --run-dir outputs/<run>
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any

from zeroshot.evaluation.metrics.itercad import aggregate_itercad
from zeroshot.evaluation.metrics.surface_distance import DISTANCE_METRICS
from zeroshot.evaluation.run_scoring import (
    DISTANCE_DECIMALS,
    Headline,
    ScoreStatus,
    StepScorer,
    format_metric,
    is_native_run,
)
from zeroshot.pipeline.models.claude import usage_metadata_from_anthropic
from zeroshot.pipeline.workflow import StopReason


class Terminal(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"  # the run raised
    INTERRUPTED = "interrupted"  # killed before it recorded either


@dataclass(frozen=True)
class Tokens:
    """`None` throughout when the backend reported no usage; never 0."""

    input: int | None = None  # already includes `cache_read` and `cache_creation`
    output: int | None = None
    reasoning: int | None = None
    cache_read: int | None = None
    cache_creation: int | None = None

    @classmethod
    def summed(cls, usages: Sequence[Mapping[str, Any]]) -> Tokens:
        if not usages:
            return cls()

        def total(key: str, group: str | None = None) -> int:
            return sum(
                (usage.get(group) or {} if group else usage).get(key, 0)
                for usage in usages
            )

        return cls(
            input=total("input_tokens"),
            output=total("output_tokens"),
            reasoning=total("reasoning", "output_token_details"),
            cache_read=total("cache_read", "input_token_details"),
            cache_creation=total("cache_creation", "input_token_details"),
        )


def _call_usage(
    usage: Mapping[str, Any] | None, metadata: Mapping[str, Any] | None
) -> Mapping[str, Any] | None:
    """A call's LangChain usage, or its Anthropic usage in those terms.

    ChatClaude reported only the provider's `usage` until it filled
    `usage_metadata`; runs from then have the counts in their response metadata.
    """
    if usage:
        return usage
    raw = (metadata or {}).get("usage")
    if isinstance(raw, Mapping) and (
        "cache_read_input_tokens" in raw or "cache_creation_input_tokens" in raw
    ):
        return usage_metadata_from_anthropic(raw)
    return None


@dataclass(frozen=True)
class SampleRow:
    sample_id: str
    terminal: Terminal
    error: str | None = None
    verify: str | None = None
    stop_reasons: Mapping[str, str] = field(default_factory=dict)
    agent_turns: int = 0
    model_calls: int = 0
    calls_with_usage: int = 0
    tool_calls: Mapping[str, int] = field(default_factory=dict)
    tokens: Tokens = field(default_factory=Tokens)
    wall_ms: int | None = None
    node_ms: Mapping[str, int] = field(default_factory=dict)
    score_status: str | None = None
    metrics: Mapping[str, float | int] = field(default_factory=dict)
    build_valid: bool | None = None
    score_errors: Mapping[str, Any] = field(default_factory=dict)
    protocol: Mapping[str, Any] = field(default_factory=dict)
    auc_options: Mapping[str, Any] = field(default_factory=dict)

    @property
    def verified(self) -> bool:
        return self.verify == "VERIFIED"

    @property
    def scored(self) -> bool:
        return self.score_status in {ScoreStatus.OK.value, ScoreStatus.PARTIAL.value}


@dataclass(frozen=True)
class MetricMeans:
    """Available mean; overall counts only explicit generation failures as zero.

    Distances have no overall mean. An evaluator error also makes overall
    undefined, instead of improving the metric by substituting a zero.
    """

    scored: float | None
    overall: float | None
    measured: int = 0


@dataclass(frozen=True)
class RunSummary:
    samples: int
    terminal: Mapping[str, int]
    execution_success: int
    scored: int
    reported_tokens: int
    budget_exhausted: int
    tokens: Tokens
    wall_ms: int
    metrics: Mapping[str, MetricMeans]
    itercad: Mapping[str, Any] = field(default_factory=dict)


def read_events(events_path: Path, sample_id: str) -> SampleRow:
    """What the run recorded about itself, ground truth not involved."""
    row = SampleRow(sample_id=sample_id, terminal=Terminal.INTERRUPTED)
    usages: list[Mapping[str, Any]] = []
    tool_calls: Counter[str] = Counter()
    node_ms: Counter[str] = Counter()
    started: dict[str, int] = {}

    with events_path.open(encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            name, data, at = event["event"], event["data"], event["timestamp_ms"]
            if name == "message" and data.get("node") == "model":
                calls = [m for m in data["messages"] if m.get("type") == "ai"]
                usages += [
                    usage
                    for m in calls
                    if (
                        usage := _call_usage(
                            m.get("usage_metadata"), m.get("response_metadata")
                        )
                    )
                ]
                row = replace(
                    row,
                    agent_turns=row.agent_turns + 1,
                    model_calls=row.model_calls + len(calls),
                )
            elif name == "tool_started":
                tool_calls[f"{data.get('tool_name')}:{data.get('caller')}"] += 1
            elif name == "verification":
                report = data["report"]
                execution = report.get("exec_report", report) or {}
                row = replace(row, verify=execution.get("status"))
            elif name == "stop_reason" and (role := data.get("role")):
                row = replace(
                    row, stop_reasons={**row.stop_reasons, role: data["reason"]}
                )
            elif name == "node_started":
                started[data["node"]] = at
            elif name == "node_finished":
                if (start := started.pop(data["node"], None)) is not None:
                    node_ms[data["node"]] += at - start
            elif name in {"run_completed", "run_failed"}:
                terminal = (
                    Terminal.COMPLETED if name == "run_completed" else Terminal.FAILED
                )
                row = replace(
                    row,
                    terminal=terminal,
                    wall_ms=data.get("duration_ms"),
                    error=data.get("error"),
                )

    return replace(
        row,
        calls_with_usage=len(usages),
        tokens=Tokens.summed(usages),
        tool_calls=dict(tool_calls),
        node_ms=dict(node_ms),
    )


# A native log line carrying a streamed delta, which nothing here reads; most
# lines are these, so they are dropped before being parsed.
_NATIVE_DELTA = '"data": [{"event": "content-block-'


def read_native_events(events_path: Path, sample_id: str) -> SampleRow:
    """What a native run (`zeroshot.pipeline_native`) recorded, in its own schema.

    A model call is counted once: a streamed call by its `message-finish`, which
    carries its usage, and a call that did not stream by its complete message.
    The log has both even for a run that failed before writing messages.json.
    """
    row = SampleRow(sample_id=sample_id, terminal=Terminal.INTERRUPTED)
    started = None
    usages: list[Mapping[str, Any]] = []
    calls = 0
    tool_calls: Counter[str] = Counter()
    with events_path.open(encoding="utf-8") as handle:
        for line in handle:
            if _NATIVE_DELTA in line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:  # the record a kill cut short
                continue
            name, data, at = event["event"], event["data"], event["timestamp"]
            if name == "messages":
                payload = data[0] if isinstance(data, list) and data else {}
                if payload.get("event") == "message-finish":
                    usage = _call_usage(payload.get("usage"), payload.get("metadata"))
                elif payload.get("type") == "ai":
                    usage = _call_usage(
                        payload.get("usage_metadata"), payload.get("response_metadata")
                    )
                else:
                    continue
                calls += 1
                if usage:
                    usages.append(usage)
            elif name == "tools" and data.get("event") == "tool-started":
                tool_calls[f"{data.get('tool_name')}:generator"] += 1
            elif name == "run_started":
                started = at
            elif name == "turn_limit_exceeded":
                row = replace(
                    row,
                    stop_reasons={"generator": StopReason.BUDGET_EXHAUSTED.value},
                )
            elif name in {"run_completed", "run_failed"}:
                row = replace(
                    row,
                    terminal=(
                        Terminal.COMPLETED
                        if name == "run_completed"
                        else Terminal.FAILED
                    ),
                    wall_ms=None if started is None else at - started,
                    error=data.get("error"),
                )

    return replace(
        row,
        agent_turns=calls,
        model_calls=calls,
        calls_with_usage=len(usages),
        tokens=Tokens.summed(usages),
        tool_calls=dict(tool_calls),
    )


def collect(run_dir: Path) -> list[SampleRow]:
    """One row per attempted sample. Dropping the crashed ones would let every
    rate below flatter itself by shrinking its own denominator."""
    rows = []
    for sample_dir in sorted(run_dir.iterdir()):
        events_path = sample_dir / "events.jsonl"
        if not events_path.is_file():
            continue
        score_path = sample_dir / "score.json"
        score = (
            json.loads(score_path.read_text("utf-8")) if score_path.is_file() else {}
        )
        native = is_native_run(sample_dir)
        row = (read_native_events if native else read_events)(
            events_path, sample_dir.name
        )
        if native:
            # Its program is built only when scored, so that is its verification.
            row = replace(row, verify=score.get("native_build_status"))
        rows.append(
            replace(
                row,
                score_status=score.get("status"),
                metrics=score.get("metrics", {}),
                build_valid=score.get("build_valid"),
                score_errors=score.get("errors", {}),
                protocol=score.get("protocol", {}),
                auc_options={
                    key.removeprefix("auc_tr_"): value
                    for key, value in score.get("scorer", {}).items()
                    if key.startswith("auc_tr_")
                },
            )
        )
    return rows


def _exhausted(row: SampleRow) -> list[str]:
    """The roles in this sample that ran out of turns, in the order they ran."""
    return [
        role
        for role, reason in row.stop_reasons.items()
        if reason == StopReason.BUDGET_EXHAUSTED.value
    ]


def _mean(values: Iterable[float]) -> float | None:
    collected = list(values)
    return sum(collected) / len(collected) if collected else None


def summarize(rows: Sequence[SampleRow]) -> RunSummary:
    scored = [row for row in rows if row.scored]
    # A sample without score.json, e.g. one with no target, has nothing to average.
    evaluated = [row for row in rows if row.score_status is not None]
    counted = [row for row in rows if row.tokens.input is not None]
    protocols = [row.protocol for row in evaluated if row.metrics]
    if any(protocols) and not all(protocols):
        raise ValueError("some scores predate the scoring protocol; rescore the run")
    auc_options = [row.auc_options for row in rows if row.auc_options]
    if auc_options and any(options != auc_options[0] for options in auc_options):
        raise ValueError("different AUC configurations are mixed; rescore the run")
    distance_summary = {}
    if evaluated:
        distance_summary = aggregate_itercad(
            [
                {
                    "build_valid": row.build_valid,
                    "metrics": row.metrics,
                    "errors": row.score_errors,
                    "protocol": row.protocol,
                }
                for row in evaluated
            ],
            **(auc_options[0] if auc_options else {}),
        )

    return RunSummary(
        samples=len(rows),
        terminal=dict(Counter(row.terminal.value for row in rows)),
        execution_success=sum(row.verified for row in rows),
        scored=len(scored),
        reported_tokens=len(counted),
        budget_exhausted=sum(bool(_exhausted(row)) for row in rows),
        # Reflected, so a token kind added to `Tokens` is totalled here too.
        tokens=Tokens(
            **{
                name: sum(getattr(row.tokens, name) or 0 for row in counted)
                for name in Tokens.__dataclass_fields__
            }
        ),
        wall_ms=sum(row.wall_ms or 0 for row in rows),
        metrics={
            name: MetricMeans(
                scored=_mean(
                    float(row.metrics[name]) for row in scored if name in row.metrics
                ),
                overall=None
                if name in DISTANCE_METRICS
                or any(
                    name not in row.metrics
                    and row.build_valid is not False
                    and row.score_status
                    not in {
                        ScoreStatus.NO_PREDICTION.value,
                        ScoreStatus.GENERATION_FAILED.value,
                    }
                    for row in evaluated
                )
                else _mean(float(row.metrics.get(name, 0)) for row in evaluated),
                measured=sum(name in row.metrics for row in scored),
            )
            for name in sorted({name for row in scored for name in row.metrics})
        },
        itercad=distance_summary,
    )


def headline_columns(
    rows: Sequence[SampleRow], headlines: Mapping[str, Headline]
) -> dict[str, Headline]:
    """The scorer's headlines that some sample has, in the scorer's order.

    `headlines()` names the columns and how to show them, so the choice lives
    with the metrics. A list here would be a copy of it.
    """
    seen = {name for row in rows for name in row.metrics}
    return {name: headline for name, headline in headlines.items() if name in seen}


def notes(rows: Sequence[SampleRow]) -> list[str]:
    """Whatever did not go the ordinary way, so the table need not carry it."""
    lines = []
    any_scored = any(row.score_status for row in rows)
    for row in rows:
        if row.terminal is not Terminal.COMPLETED:
            lines.append(f"{row.sample_id}  {row.terminal.value}: {row.error}")
        elif not row.verified:
            lines.append(f"{row.sample_id}  final verification {row.verify}")
        if exhausted := _exhausted(row):
            lines.append(f"{row.sample_id}  budget exhausted: {', '.join(exhausted)}")
        if row.score_status not in {None, ScoreStatus.OK.value}:
            lines.append(f"{row.sample_id}  score {row.score_status}")
        if any_scored and row.score_status is None:
            lines.append(f"{row.sample_id}  no score.json, left out of the metrics")
        for stage, error in row.score_errors.items():
            if isinstance(error, Mapping):
                lines.append(
                    f"{row.sample_id}  {error.get('kind', 'evaluation_error')} [{stage}]: {error.get('message')}"
                )
            else:
                lines.append(f"{row.sample_id}  score error [{stage}]: {error}")
        if row.model_calls and not row.calls_with_usage:
            lines.append(f"{row.sample_id}  backend reported no token usage")
    return lines


def _cell(value: object, decimals: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return format_metric(value, decimals)
    return f"{value:,}" if isinstance(value, int) else str(value)


def _aligned(rows: Sequence[Sequence[str]]) -> list[str]:
    widths = [max(map(len, column)) for column in zip(*rows, strict=True)]
    return [
        "  ".join(cell.rjust(width) for cell, width in zip(row, widths, strict=True))
        for row in rows
    ]


def format_report(
    rows: Sequence[SampleRow],
    summary: RunSummary,
    headlines: Mapping[str, Headline],
) -> str:
    lines = _aligned(
        [
            (
                "sample",
                "turns",
                "verify",
                *(headline.label for headline in headlines.values()),
                "in_tok",
                "wall_s",
            ),
            *(
                (
                    row.sample_id,
                    _cell(row.agent_turns),
                    row.verify or "-",
                    *(
                        _cell(row.metrics.get(name), headline.decimals)
                        for name, headline in headlines.items()
                    ),
                    _cell(row.tokens.input),
                    _cell(round((row.wall_ms or 0) / 1000)),
                )
                for row in rows
            ),
        ]
    )
    rule = "-" * len(lines[0])
    lines.insert(1, rule)

    total = summary.samples
    evaluated = sum(row.score_status is not None for row in rows)
    rates = ("execution_success", "scored", "reported_tokens", "budget_exhausted")
    lines += [
        rule,
        "   ".join(f"{name} {getattr(summary, name)}/{total}" for name in rates),
        *(
            f"{name}  mean {_cell(means.scored, headline.decimals)}"
            f" (measured n={means.measured})"
            + (
                " (valid only)"
                if name in DISTANCE_METRICS
                else f"   {_cell(means.overall, headline.decimals)}"
                f" (all n={evaluated}, invalid=0)"
            )
            for name, headline in headlines.items()
            if (means := summary.metrics.get(name)) is not None
        ),
        (
            f"in_tok {_cell(summary.tokens.input)}"
            f"   out_tok {_cell(summary.tokens.output)}"
            f"   wall {round(summary.wall_ms / 1000):,}s"
        ),
        (
            f"  of which cache_read {_cell(summary.tokens.cache_read)}"
            f"   cache_creation {_cell(summary.tokens.cache_creation)}"
            f"   reasoning {_cell(summary.tokens.reasoning)}"
        ),
    ]
    if summary.itercad:
        stats = summary.itercad
        lines += [
            (
                f"AUC-TR {_cell(stats['auc_tr'])}"
                f"   Mean CD {_cell(stats['mean_cd'], DISTANCE_DECIMALS)}"
                f"   Med. CD {_cell(stats['median_cd'], DISTANCE_DECIMALS)}"
            ),
            (
                f"  CD in GT bbox diagonals; tolerances [{stats['auc_tr_min_cd']:g},"
                f" {stats['auc_tr_max_cd']:g}] over {stats['auc_tr_num_points']} points"
            ),
            (
                f"  invalid {stats['invalid_predictions']},"
                f" evaluator errors {stats['cd_evaluation_errors']}"
            ),
        ]
    if remarks := notes(rows):
        lines += ["", "notes:", *(f"  {line}" for line in remarks)]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-name", default="summary.json")
    args = parser.parse_args()

    rows = collect(args.run_dir)
    if not rows:
        raise SystemExit(f"no sample with events.jsonl under {args.run_dir}")
    summary = summarize(rows)
    headlines = headline_columns(rows, StepScorer.headlines())

    print(format_report(rows, summary, headlines))

    output = args.output or args.run_dir / args.summary_name
    document = {"summary": asdict(summary), "samples": [asdict(row) for row in rows]}
    output.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", "utf-8")
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
