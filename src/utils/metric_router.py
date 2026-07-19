"""Route one metric event to independent persistence and progress sinks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class MetricLogSink(Protocol):
    def log(self, metrics: Mapping[str, Any], *, step: int) -> None: ...

    def log_artifact(
        self, path: str | Path, *, name: str | None = None, kind: str = "artifact"
    ) -> None: ...

    def finish(self) -> None: ...


class ProgressMetricSink(Protocol):
    def update_metrics(self, metrics: Mapping[str, str]) -> None: ...


@dataclass(frozen=True)
class LoggedMetric:
    """A scalar metric plus optional progress-bar presentation metadata."""

    value: Any
    prog_bar: bool = False
    display_name: str | None = None
    format_spec: str | None = None

    def __post_init__(self) -> None:
        if self.display_name is not None and not self.display_name:
            raise ValueError("metric display_name must be non-empty when provided")


def _display_scalar(value: Any, *, key: str, format_spec: str | None) -> str:
    if hasattr(value, "numel"):
        if value.numel() != 1:
            raise TypeError(f"progress metric {key!r} must be scalar")
        value = value.detach().cpu().item()
    if hasattr(value, "item") and not isinstance(value, (bool, int, float, str)):
        value = value.item()
    if value is None:
        return "—"
    if format_spec is not None:
        try:
            return format(value, format_spec)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"invalid progress format {format_spec!r} for metric {key!r}"
            ) from error
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


class MetricRouter:
    """Mediator analogous to Lightning's ``self.log(..., prog_bar=True)``.

    The persistence logger and progress bar implement separate protocols and
    never reference one another. The training orchestration emits one event;
    this router persists every value and forwards only explicitly marked
    values to the progress display.
    """

    def __init__(
        self,
        logger: MetricLogSink,
        progress: ProgressMetricSink | None = None,
    ) -> None:
        self.logger = logger
        self.progress = progress

    def log(
        self,
        metrics: Mapping[str, Any | LoggedMetric],
        *,
        step: int,
    ) -> None:
        persisted: dict[str, Any] = {}
        displayed: dict[str, str] = {}
        for key, raw in metrics.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"metric keys must be non-empty strings, got {key!r}")
            metric = raw if isinstance(raw, LoggedMetric) else LoggedMetric(raw)
            persisted[key] = metric.value
            if metric.prog_bar:
                display_name = metric.display_name or key
                displayed[display_name] = _display_scalar(
                    metric.value,
                    key=key,
                    format_spec=metric.format_spec,
                )
        self.logger.log(persisted, step=step)
        if displayed and self.progress is not None:
            self.progress.update_metrics(displayed)

    def log_artifact(
        self, path: str | Path, *, name: str | None = None, kind: str = "artifact"
    ) -> None:
        self.logger.log_artifact(path, name=name, kind=kind)

    def finish(self) -> None:
        self.logger.finish()


__all__ = [
    "LoggedMetric",
    "MetricLogSink",
    "MetricRouter",
    "ProgressMetricSink",
]
