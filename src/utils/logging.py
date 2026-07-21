"""Local-first metric logging with an optional Weights & Biases sink."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import re
import threading
from typing import Any, Mapping, Protocol

import torch

log = logging.getLogger(__name__)


class MetricSink(Protocol):
    def log(self, metrics: Mapping[str, Any], *, step: int) -> None: ...

    def log_artifact(
        self, path: str | Path, *, name: str | None = None, kind: str = "artifact"
    ) -> None: ...

    def finish(self) -> None: ...


def _scalar(value: Any, *, key: str) -> bool | int | float | str | None:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise TypeError(
                f"metric {key!r} must be scalar, got tensor {tuple(value.shape)}"
            )
        value = value.detach().cpu().item()
    try:
        import numpy as np
    except ImportError:
        np = None  # type: ignore[assignment]
    if np is not None and isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        # JSON has no portable NaN/Infinity representation. Preserve the key
        # as null; checkpoint monitoring performs its own strict finite check.
        return None
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(
        f"metric {key!r} must be a scalar, string, or None, got {type(value)}"
    )


def normalize_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(metrics, Mapping):
        raise TypeError(f"metrics must be a mapping, got {type(metrics)}")
    normalized: dict[str, Any] = {}
    for key, value in metrics.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"metric keys must be non-empty strings, got {key!r}")
        if key in {"step", "timestamp"}:
            raise ValueError(f"metric key {key!r} is reserved")
        normalized[key] = _scalar(value, key=key)
    return normalized


class JSONLMetricLogger:
    """Append one flushed JSON object per logging step."""

    def __init__(self, path: str | Path, *, enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = enabled
        self._lock = threading.Lock()
        self._handle = None
        if enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("a", encoding="utf-8")

    def log(self, metrics: Mapping[str, Any], *, step: int) -> None:
        if not self.enabled:
            return
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError(f"step must be a non-negative integer, got {step!r}")
        record = {
            "step": step,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **normalize_metrics(metrics),
        }
        line = json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False)
        with self._lock:
            assert self._handle is not None
            self._handle.write(line + "\n")
            self._handle.flush()

    def log_artifact(
        self, path: str | Path, *, name: str | None = None, kind: str = "artifact"
    ) -> None:
        del path, name, kind

    def finish(self) -> None:
        with self._lock:
            if self._handle is not None:
                self._handle.close()
                self._handle = None

    def __enter__(self) -> "JSONLMetricLogger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.finish()


def _wandb_run_id(run_dir: Path) -> str | None:
    """Extract the most recent wandb run ID from the run directory.

    Wandb stores runs under ``<run_dir>/wandb/run-YYYYMMDD_HHMMSS-<id>/``.
    This function finds the latest such directory and parses the run ID.
    """
    wandb_dir = run_dir / "wandb"
    if not wandb_dir.exists():
        return None
    candidates = [
        path
        for path in wandb_dir.iterdir()
        if path.is_dir() and path.name.startswith("run-")
    ]
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    match = re.match(r"run-\d{8}_\d{6}-(?P<id>.+)$", latest.name)
    return match.group("id") if match else None


class WandbMetricLogger:
    """Lazy W&B adapter; importing this module never requires ``wandb``."""

    def __init__(
        self,
        *,
        project: str,
        config: Mapping[str, Any] | None = None,
        entity: str | None = None,
        group: str | None = None,
        name: str | None = None,
        tags: list[str] | tuple[str, ...] = (),
        mode: str = "online",
        run_dir: str | Path | None = None,
        resume_id: str | None = None,
    ) -> None:
        if not project:
            raise ValueError("wandb project must not be empty")
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError(
                "W&B logging is enabled but wandb is not installed"
            ) from error
        self._wandb = wandb
        self._run = wandb.init(
            project=project,
            entity=entity,
            group=group,
            name=name,
            tags=list(tags),
            mode=mode,
            dir=None if run_dir is None else str(run_dir),
            config=None if config is None else dict(config),
            id=resume_id,
            resume="allow" if resume_id else None,
        )

    def log(self, metrics: Mapping[str, Any], *, step: int) -> None:
        self._run.log(normalize_metrics(metrics), step=step)

    def log_artifact(
        self, path: str | Path, *, name: str | None = None, kind: str = "artifact"
    ) -> None:
        artifact_path = Path(path)
        if not artifact_path.exists():
            raise FileNotFoundError(f"artifact does not exist: {artifact_path}")
        artifact = self._wandb.Artifact(name or artifact_path.name, type=kind)
        if artifact_path.is_dir():
            artifact.add_dir(str(artifact_path))
        else:
            artifact.add_file(str(artifact_path))
        self._run.log_artifact(artifact)

    def finish(self) -> None:
        self._run.finish()


class ExperimentLogger:
    """Fan metrics out to local JSONL and any enabled remote sinks."""

    def __init__(self, sinks: list[MetricSink] | tuple[MetricSink, ...]) -> None:
        self.sinks = tuple(sinks)

    @classmethod
    def from_config(
        cls,
        run_dir: str | Path,
        config: Mapping[str, Any],
        *,
        resolved_config: Mapping[str, Any] | None = None,
        is_main_process: bool = True,
        resume: bool = False,
    ) -> "ExperimentLogger":
        if not is_main_process:
            return cls(())
        run_dir = Path(run_dir)
        sinks: list[MetricSink] = []
        jsonl_config = config.get("jsonl", {})
        if bool(jsonl_config.get("enabled", True)):
            sinks.append(
                JSONLMetricLogger(
                    run_dir / jsonl_config.get("filename", "metrics.jsonl")
                )
            )
        wandb_config = config.get("wandb", {})
        if bool(wandb_config.get("enabled", False)):
            resume_id: str | None = None
            if resume:
                resume_id = _wandb_run_id(run_dir)
                if resume_id is None:
                    log.warning(
                        "No wandb run id found under %s/wandb; "
                        "starting a new wandb run.",
                        run_dir,
                    )
                else:
                    log.info("Resuming wandb run id %s", resume_id)
            sinks.append(
                WandbMetricLogger(
                    project=wandb_config.get("project", "drawing2cad"),
                    config=resolved_config,
                    entity=wandb_config.get("entity"),
                    group=wandb_config.get("group"),
                    name=wandb_config.get("name"),
                    tags=wandb_config.get("tags", ()),
                    mode=wandb_config.get("mode", "online"),
                    run_dir=run_dir,
                    resume_id=resume_id,
                )
            )
        return cls(sinks)

    def log(self, metrics: Mapping[str, Any], *, step: int) -> None:
        for sink in self.sinks:
            sink.log(metrics, step=step)

    def log_artifact(
        self, path: str | Path, *, name: str | None = None, kind: str = "artifact"
    ) -> None:
        for sink in self.sinks:
            sink.log_artifact(path, name=name, kind=kind)

    def finish(self) -> None:
        errors: list[Exception] = []
        for sink in reversed(self.sinks):
            try:
                sink.finish()
            except Exception as error:  # ensure every sink receives finish
                errors.append(error)
        if errors:
            raise RuntimeError(
                f"failed to finish {len(errors)} metric sink(s)"
            ) from errors[0]

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.finish()


__all__ = [
    "ExperimentLogger",
    "JSONLMetricLogger",
    "MetricSink",
    "WandbMetricLogger",
    "normalize_metrics",
]
