"""Rank-aware run-directory setup and reproducibility helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import random
import socket
import subprocess
from typing import Any, Mapping

import numpy as np
from omegaconf import OmegaConf
import torch
import yaml


@dataclass(frozen=True)
class RunContext:
    """Filesystem locations shared by the training components."""

    run_dir: Path
    config_path: Path
    metadata_path: Path
    metrics_path: Path
    checkpoints_dir: Path
    predictions_dir: Path
    is_main_process: bool
    rank: int
    world_size: int


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` derived from PyTorch's worker seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _to_resolved_container(config: Any) -> Any:
    if OmegaConf.is_config(config):
        return OmegaConf.to_container(config, resolve=True, enum_to_str=True)
    if isinstance(config, Mapping):
        return {
            str(key): _to_resolved_container(value) for key, value in config.items()
        }
    if isinstance(config, (list, tuple)):
        return [_to_resolved_container(value) for value in config]
    if isinstance(config, Path):
        return str(config)
    return config


def _nested_get(config: Any, *keys: str, default: Any = None) -> Any:
    current = config
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _git_revision(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = result.stdout.strip()
    return revision or None


def _run_metadata(
    config: Mapping[str, Any],
    *,
    project_root: Path,
    world_size: int,
    timestamp: datetime,
) -> dict[str, Any]:
    cuda_version = torch.version.cuda
    return {
        "created_at": timestamp.astimezone().isoformat(),
        "git_revision": _git_revision(project_root),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": cuda_version,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "world_size": world_size,
        "base_checkpoint": _nested_get(
            config, "model", "model_name_or_path", default=None
        ),
        "dataset_roots": {
            "train": _nested_get(config, "data", "train_root", default=None),
            "validation": _nested_get(config, "data", "val_root", default=None),
        },
    }


def setup_run(
    config: Any,
    *,
    output_dir: str | Path | None = None,
    project_root: str | Path | None = None,
    is_main_process: bool = True,
    rank: int | None = None,
    world_size: int | None = None,
    resume: bool = False,
    timestamp: datetime | None = None,
) -> RunContext:
    """Create a run layout and persist the fully resolved configuration.

    ``output_dir`` should normally be Hydra's runtime output directory. Resume
    callers pass the original run directory and set ``resume=True``.
    """

    resolved = _to_resolved_container(config)
    if not isinstance(resolved, Mapping):
        raise TypeError("config must resolve to a mapping")
    resolved = dict(resolved)
    timestamp = timestamp or datetime.now().astimezone()
    rank = int(os.environ.get("RANK", "0")) if rank is None else rank
    world_size = (
        int(os.environ.get("WORLD_SIZE", "1")) if world_size is None else world_size
    )
    if rank < 0 or world_size <= 0 or rank >= world_size:
        raise ValueError(f"invalid distributed rank/world size: {rank}/{world_size}")

    if project_root is None:
        project_root = _nested_get(
            resolved, "paths", "project_root", default=Path.cwd()
        )
    project_root_path = Path(project_root).expanduser().resolve()

    if output_dir is None:
        configured = _nested_get(resolved, "paths", "output_dir", default=None)
        if configured is not None:
            output_dir = configured
        else:
            log_root = _nested_get(
                resolved,
                "paths",
                "log_root",
                default=project_root_path / "logs" / "train_sft",
            )
            output_dir = Path(log_root) / timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(output_dir).expanduser().resolve()

    config_path = run_dir / "config.yaml"
    metadata_path = run_dir / "run_metadata.json"
    metrics_filename = _nested_get(
        resolved, "logger", "jsonl", "filename", default="metrics.jsonl"
    )
    checkpoint_dirname = _nested_get(
        resolved, "checkpoint", "root_dirname", default="checkpoints"
    )
    prediction_dirname = _nested_get(
        resolved,
        "evaluation",
        "predictions_dirname",
        default="val_predictions",
    )
    context = RunContext(
        run_dir=run_dir,
        config_path=config_path,
        metadata_path=metadata_path,
        metrics_path=run_dir / str(metrics_filename),
        checkpoints_dir=run_dir / str(checkpoint_dirname),
        predictions_dir=run_dir / str(prediction_dirname),
        is_main_process=is_main_process,
        rank=rank,
        world_size=world_size,
    )
    if not is_main_process:
        return context

    if resume and not run_dir.is_dir():
        raise FileNotFoundError(f"resume run directory does not exist: {run_dir}")
    # Hydra creates its runtime directory before invoking the task function.
    # Accept that directory, but never overwrite an existing run config unless
    # this is an explicit resume.
    if not resume and config_path.exists():
        raise FileExistsError(
            f"run directory already contains a resolved config: {config_path}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    context.checkpoints_dir.mkdir(exist_ok=True)
    if bool(_nested_get(resolved, "evaluation", "generation_enabled", default=True)):
        context.predictions_dir.mkdir(exist_ok=True)

    config_text = yaml.safe_dump(
        resolved,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
    _atomic_write_text(config_path, config_text)
    if not resume or not metadata_path.exists():
        metadata = _run_metadata(
            resolved,
            project_root=project_root_path,
            world_size=world_size,
            timestamp=timestamp,
        )
    else:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["resume_count"] = int(metadata.get("resume_count", 0)) + 1
        metadata["last_resumed_at"] = timestamp.astimezone().isoformat()
    _atomic_write_text(
        metadata_path,
        json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    return context


__all__ = ["RunContext", "seed_worker", "setup_run"]
