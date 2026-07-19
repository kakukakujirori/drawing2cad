"""Rank-aware Rich progress display for explicit training loops."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta
from typing import Any

from rich.console import Console, Group
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    Task,
)
from rich.text import Text


class _EpochColumn(ProgressColumn):
    def __init__(self, style: str) -> None:
        super().__init__()
        self.style = style

    def render(self, task: Task) -> Text:
        return Text(f"[epoch {task.fields['epoch']}] |", style=self.style)


class _StepColumn(ProgressColumn):
    def __init__(self, style: str) -> None:
        super().__init__()
        self.style = style

    def render(self, task: Task) -> Text:
        completed = min(int(task.completed), int(task.total or 0))
        total = int(task.total or 0)
        return Text(f"|[{completed}/{total}]", style=self.style)


class _MetricsColumn(ProgressColumn):
    def __init__(self, style: str) -> None:
        super().__init__()
        self.style = style

    def render(self, task: Task) -> Text:
        metrics = task.fields.get("metrics", {})
        return Text(
            " ".join(f"{key}: {value}" for key, value in metrics.items()),
            style=self.style,
        )


class _LightningTimeColumn(ProgressColumn):
    """Match Lightning's Rich ``elapsed • remaining`` time column."""

    max_refresh = 0.5

    def __init__(self, style: str) -> None:
        super().__init__()
        self.style = style

    @staticmethod
    def _duration(seconds: float | None) -> str:
        if seconds is None:
            return "-:--:--"
        return str(timedelta(seconds=int(seconds)))

    def render(self, task: Task) -> Text:
        elapsed = task.finished_time if task.finished else task.elapsed
        return Text(
            f"{self._duration(elapsed)} • {self._duration(task.time_remaining)}",
            style=self.style,
        )


class _ProcessingSpeedColumn(ProgressColumn):
    """Match Lightning's optimizer-iteration speed presentation."""

    def __init__(self, style: str) -> None:
        super().__init__()
        self.style = style

    def render(self, task: Task) -> Text:
        speed = f"{task.speed:>.2f}" if task.speed is not None else "0.00"
        return Text(f"{speed}it/s", style=self.style)


class RichEpochProgressBar:
    """Render one optimizer-step bar per epoch on the main process.

    The visual contract is deliberately small and stable::

        [epoch N] |━━━━      |[step/steps] loss: ... 0:00:12 • 0:00:38 1.86it/s

    ``refresh_rate`` follows Lightning's RichProgressBar convention: zero
    disables display, while a positive value is the maximum refresh frequency
    per second. ``leave=False`` removes completed bars instead of accumulating
    one terminal row per epoch.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        is_main_process: bool = True,
        refresh_rate: float = 10.0,
        leave: bool = False,
        styles: Mapping[str, str] | None = None,
        console: Console | None = None,
        auto_refresh: bool = True,
    ) -> None:
        if refresh_rate < 0:
            raise ValueError("progress refresh_rate must be non-negative")
        theme = {
            "epoch": "bold cyan",
            "bar": "grey37",
            "complete": "cyan",
            "finished": "green",
            "pulse": "cyan",
            "count": "bold cyan",
            "metrics": "italic",
            "time": "dim",
            "speed": "dim underline",
        }
        if styles is not None:
            unknown = set(styles) - set(theme)
            if unknown:
                raise ValueError(f"unknown progress styles: {sorted(unknown)}")
            theme.update({key: str(value) for key, value in styles.items()})

        self.enabled = bool(enabled and is_main_process and refresh_rate > 0)
        self.leave = bool(leave)
        self._started = False
        self._task_id: int | None = None
        self._total_steps: int | None = None
        self._progress = Progress(
            _EpochColumn(theme["epoch"]),
            BarColumn(
                bar_width=None,
                style=theme["bar"],
                complete_style=theme["complete"],
                finished_style=theme["finished"],
                pulse_style=theme["pulse"],
            ),
            _StepColumn(theme["count"]),
            _MetricsColumn(theme["metrics"]),
            _LightningTimeColumn(theme["time"]),
            _ProcessingSpeedColumn(theme["speed"]),
            console=console,
            auto_refresh=auto_refresh,
            refresh_per_second=max(float(refresh_rate), 1.0),
            transient=not self.leave,
            disable=not self.enabled,
            expand=True,
        )

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        is_main_process: bool,
        console: Console | None = None,
    ) -> "RichEpochProgressBar":
        raw = config
        styles = raw.get("styles")
        if styles is not None and not isinstance(styles, Mapping):
            raise TypeError("utils.progress_bar.styles must be a mapping")
        return cls(
            enabled=bool(raw.get("enabled", True)),
            is_main_process=is_main_process,
            refresh_rate=float(raw.get("refresh_rate", 10.0)),
            leave=bool(raw.get("leave", False)),
            styles=styles,
            console=console,
        )

    def start(self) -> None:
        if not self._started:
            self._progress.start()
            self._started = True

    def start_epoch(
        self,
        epoch: int,
        *,
        total_steps: int,
        completed_steps: int = 0,
    ) -> None:
        if epoch <= 0:
            raise ValueError("displayed epoch must be positive")
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if not 0 <= completed_steps <= total_steps:
            raise ValueError("completed_steps must be within [0, total_steps]")
        self.start()
        if self._task_id is not None and not self.leave:
            self._progress.remove_task(self._task_id)
        self._task_id = self._progress.add_task(
            "",
            total=total_steps,
            completed=completed_steps,
            epoch=epoch,
            metrics={},
        )
        self._total_steps = total_steps

    def advance(self, steps: int = 1) -> None:
        if steps < 0:
            raise ValueError("progress advance must be non-negative")
        if self._task_id is None:
            raise RuntimeError("start_epoch must be called before advance")
        self._progress.advance(self._task_id, steps)

    def finish_epoch(self) -> None:
        if self._task_id is None:
            return
        assert self._total_steps is not None
        self._progress.update(self._task_id, completed=self._total_steps)
        self._progress.stop_task(self._task_id)
        self._progress.refresh()

    def update_metrics(self, metrics: Mapping[str, str]) -> None:
        if not isinstance(metrics, Mapping):
            raise TypeError("progress metrics must be a mapping")
        if self._task_id is None:
            return
        task = next(task for task in self._progress.tasks if task.id == self._task_id)
        current = dict(task.fields.get("metrics", {}))
        current.update({str(key): str(value) for key, value in metrics.items()})
        self._progress.update(self._task_id, metrics=current)

    def refresh(self) -> None:
        if self._started:
            self._progress.refresh()

    def stop(self) -> None:
        if self._started:
            self._progress.stop()
            self._started = False

    def __enter__(self) -> "RichEpochProgressBar":
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()

    def get_renderable(self) -> Group:
        """Return the current Rich renderable, primarily for deterministic tests."""

        return self._progress.get_renderable()


__all__ = ["RichEpochProgressBar"]
