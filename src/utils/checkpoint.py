"""Complete latest-state and metric-driven top-K checkpoint management."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Mapping
from uuid import uuid4


SaveState = Callable[[Path], None]
LoadState = Callable[[Path], None]
Barrier = Callable[[], None]


@dataclass(frozen=True)
class CheckpointEntry:
    step: int
    value: float
    path: str


class CheckpointManager:
    """Manage a resumable ``latest`` state and an atomic top-K index.

    The callbacks deliberately receive directories. They can therefore be
    connected directly to ``Accelerator.save_state`` / ``load_state`` while
    tests and non-Accelerate callers can provide lightweight implementations.
    """

    def __init__(
        self,
        root_dir: str | Path,
        *,
        monitor: str,
        mode: str,
        top_k: int,
        latest_dirname: str = "latest",
        topk_dirname: str = "topk",
        index_filename: str = "topk.json",
        filename: str = "step_{step:08d}-{monitor_name}_{monitor_value:.4f}",
        is_main_process: bool = True,
        barrier: Barrier | None = None,
    ) -> None:
        if not monitor:
            raise ValueError("checkpoint monitor must not be empty")
        if mode not in {"min", "max"}:
            raise ValueError(f"checkpoint mode must be 'min' or 'max', got {mode!r}")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 0:
            raise ValueError(f"top_k must be a non-negative integer, got {top_k!r}")
        for label, value in {
            "latest_dirname": latest_dirname,
            "topk_dirname": topk_dirname,
            "index_filename": index_filename,
        }.items():
            if not value or Path(value).name != value:
                raise ValueError(f"{label} must be a single path component, got {value!r}")
        if latest_dirname == topk_dirname:
            raise ValueError("latest_dirname and topk_dirname must differ")
        self.root_dir = Path(root_dir)
        self.monitor = monitor
        self.mode = mode
        self.top_k = top_k
        self.latest_dirname = latest_dirname
        self.topk_dirname = topk_dirname
        self.index_filename = index_filename
        self.filename = filename
        self.is_main_process = is_main_process
        self.barrier = barrier
        self.latest_dir = self.root_dir / latest_dirname
        self.topk_dir = self.root_dir / topk_dirname
        self.index_path = self.root_dir / index_filename
        self._entries: list[CheckpointEntry] = []
        if is_main_process:
            self.root_dir.mkdir(parents=True, exist_ok=True)
            self.topk_dir.mkdir(exist_ok=True)
            self._entries = self._load_index()

    @classmethod
    def from_config(
        cls,
        run_dir: str | Path,
        config: Mapping[str, Any],
        *,
        is_main_process: bool = True,
        barrier: Barrier | None = None,
    ) -> "CheckpointManager":
        """Build a manager from the ``checkpoint`` Hydra config group."""

        return cls(
            Path(run_dir) / str(config.get("root_dirname", "checkpoints")),
            monitor=str(config["monitor"]),
            mode=str(config.get("mode", "max")),
            top_k=int(config.get("top_k", 3)),
            latest_dirname=str(config.get("latest_dirname", "latest")),
            topk_dirname=str(config.get("topk_dirname", "topk")),
            index_filename=str(config.get("index_filename", "topk.json")),
            filename=str(
                config.get(
                    "filename",
                    "step_{step:08d}-{monitor_name}_{monitor_value:.4f}",
                )
            ),
            is_main_process=is_main_process,
            barrier=barrier,
        )

    @property
    def entries(self) -> tuple[CheckpointEntry, ...]:
        return tuple(self._entries)

    def _sort(self, entries: list[CheckpointEntry]) -> list[CheckpointEntry]:
        if self.mode == "max":
            return sorted(entries, key=lambda item: (-item.value, -item.step))
        return sorted(entries, key=lambda item: (item.value, -item.step))

    def _load_index(self) -> list[CheckpointEntry]:
        if not self.index_path.exists():
            return []
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"invalid checkpoint index: {self.index_path}") from error
        if payload.get("monitor") != self.monitor or payload.get("mode") != self.mode:
            raise ValueError(
                "existing checkpoint index policy does not match configuration: "
                f"index={payload.get('monitor')!r}/{payload.get('mode')!r}, "
                f"config={self.monitor!r}/{self.mode!r}"
            )
        entries: list[CheckpointEntry] = []
        for raw in payload.get("checkpoints", []):
            entry = CheckpointEntry(
                step=int(raw["step"]), value=float(raw["value"]), path=str(raw["path"])
            )
            if (self.root_dir / entry.path).is_dir():
                entries.append(entry)
        return self._sort(entries)[: self.top_k]

    def _metric(self, metrics: Mapping[str, Any]) -> float:
        if self.monitor not in metrics:
            raise KeyError(
                f"checkpoint monitor {self.monitor!r} is absent from evaluation metrics; "
                f"available keys: {sorted(metrics)}"
            )
        raw = metrics[self.monitor]
        try:
            if hasattr(raw, "numel"):
                if raw.numel() != 1:
                    raise TypeError
                raw = raw.detach().cpu().item()
            value = float(raw)
        except (TypeError, ValueError, AttributeError) as error:
            raise TypeError(
                f"checkpoint monitor {self.monitor!r} must be a numeric scalar, got {raw!r}"
            ) from error
        if not math.isfinite(value):
            raise ValueError(
                f"checkpoint monitor {self.monitor!r} must be finite, got {value!r}"
            )
        return value

    def _checkpoint_name(self, *, step: int, value: float) -> str:
        monitor_name = self.monitor.replace("/", "_").replace(" ", "_")
        try:
            name = self.filename.format(
                step=step,
                monitor_name=monitor_name,
                monitor_value=value,
            )
        except (KeyError, ValueError, IndexError) as error:
            raise ValueError(f"invalid checkpoint filename template: {self.filename!r}") from error
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"checkpoint filename must produce one safe component, got {name!r}")
        return name

    @staticmethod
    def _write_metadata(path: Path, payload: Mapping[str, Any]) -> None:
        metadata_path = path / "checkpoint_metadata.json"
        metadata_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _json_scalar(value: Any, *, key: str) -> bool | int | float | str | None:
        if hasattr(value, "numel"):
            if value.numel() != 1:
                raise TypeError(f"checkpoint metric {key!r} must be scalar")
            value = value.detach().cpu().item()
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        raise TypeError(
            f"checkpoint metric {key!r} must be JSON-serializable scalar, got {type(value)}"
        )

    def _save_directory(
        self,
        target: Path,
        save_state: SaveState,
        metadata: Mapping[str, Any],
    ) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
        try:
            save_state(temporary)
            self._write_metadata(temporary, metadata)
            backup: Path | None = None
            if target.exists():
                backup = target.with_name(f".{target.name}.old-{uuid4().hex}")
                os.replace(target, backup)
            try:
                os.replace(temporary, target)
            except BaseException:
                if backup is not None and backup.exists():
                    os.replace(backup, target)
                raise
            if backup is not None:
                shutil.rmtree(backup)
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise

    @staticmethod
    def _commit_directory(source: Path, target: Path) -> None:
        """Replace ``target`` with an already-complete staging directory."""

        backup: Path | None = None
        if target.exists():
            backup = target.with_name(f".{target.name}.old-{uuid4().hex}")
            os.replace(target, backup)
        try:
            os.replace(source, target)
        except BaseException:
            if backup is not None and backup.exists():
                os.replace(backup, target)
            raise
        if backup is not None:
            shutil.rmtree(backup)

    def _save_latest_distributed(
        self,
        *,
        step: int,
        save_state: SaveState,
        metadata: Mapping[str, Any],
    ) -> None:
        """Run an Accelerate-style save callback on every process once."""

        assert self.barrier is not None
        staging = self.root_dir / f".{self.latest_dirname}.staging-step-{step:08d}"
        if self.is_main_process:
            if staging.exists():
                shutil.rmtree(staging)
            staging.mkdir(parents=True)
        self.barrier()
        save_state(staging)
        self.barrier()
        if self.is_main_process:
            self._write_metadata(staging, metadata)
            self._commit_directory(staging, self.latest_dir)
        self.barrier()

    @staticmethod
    def _link_or_copy(source: str, destination: str) -> str:
        try:
            os.link(source, destination)
        except OSError:
            return shutil.copy2(source, destination)
        return destination

    def _copy_directory(self, source: Path, target: Path) -> None:
        """Atomically snapshot immutable latest state into the top-K set."""

        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
        try:
            shutil.copytree(
                source,
                temporary,
                dirs_exist_ok=True,
                copy_function=self._link_or_copy,
            )
            backup: Path | None = None
            if target.exists():
                backup = target.with_name(f".{target.name}.old-{uuid4().hex}")
                os.replace(target, backup)
            try:
                os.replace(temporary, target)
            except BaseException:
                if backup is not None and backup.exists():
                    os.replace(backup, target)
                raise
            if backup is not None:
                shutil.rmtree(backup)
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise

    def _write_index(self) -> None:
        payload = {
            "monitor": self.monitor,
            "mode": self.mode,
            "top_k": self.top_k,
            "checkpoints": [asdict(entry) for entry in self._entries],
        }
        temporary = self.index_path.with_name(
            f".{self.index_path.name}.tmp-{os.getpid()}-{uuid4().hex}"
        )
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.index_path)

    def _candidate_entries(self, candidate: CheckpointEntry) -> list[CheckpointEntry]:
        without_same_step = [entry for entry in self._entries if entry.step != candidate.step]
        return self._sort([*without_same_step, candidate])[: self.top_k]

    def save(
        self,
        *,
        step: int,
        metrics: Mapping[str, Any],
        save_state: SaveState,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> bool:
        """Save latest state and, when qualified, a top-K copy.

        Returns ``True`` when this evaluation entered the retained top-K set.
        Non-main ranks perform no filesystem work and return ``False``.
        """

        if not self.is_main_process and self.barrier is None:
            return False
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError(f"step must be a non-negative integer, got {step!r}")
        value = self._metric(metrics)
        extra = {} if extra_metadata is None else dict(extra_metadata)
        reserved_metadata = {
            "step",
            "monitor",
            "monitor_value",
            "metrics",
        }.intersection(extra)
        if reserved_metadata:
            raise ValueError(
                "extra checkpoint metadata cannot override reserved keys: "
                f"{sorted(reserved_metadata)}"
            )
        metadata = {
            **extra,
            "step": step,
            "monitor": self.monitor,
            "monitor_value": value,
            "metrics": {
                key: self._json_scalar(metric, key=key)
                for key, metric in metrics.items()
            },
        }
        if self.barrier is None:
            self._save_directory(self.latest_dir, save_state, metadata)
        else:
            self._save_latest_distributed(
                step=step, save_state=save_state, metadata=metadata
            )
        if not self.is_main_process:
            return False
        if self.top_k == 0:
            return False

        name = self._checkpoint_name(step=step, value=value)
        relative_path = str(Path(self.topk_dirname) / name)
        candidate = CheckpointEntry(step=step, value=value, path=relative_path)
        retained = self._candidate_entries(candidate)
        if candidate not in retained:
            return False

        target = self.root_dir / relative_path
        # Snapshot the already-complete latest state. This invokes an
        # Accelerator save exactly once per evaluation and uses hardlinks when
        # supported; latest is replaced as a directory on the next save, so
        # retained snapshots remain immutable.
        self._copy_directory(self.latest_dir, target)
        previous = self._entries
        self._entries = retained
        self._write_index()
        retained_paths = {entry.path for entry in retained}
        for entry in previous:
            if entry.path not in retained_paths:
                obsolete = self.root_dir / entry.path
                if obsolete.is_dir() and obsolete != target:
                    shutil.rmtree(obsolete)
        return True

    def load_latest(self, load_state: LoadState) -> dict[str, Any]:
        """Restore the complete latest state and return its bookkeeping metadata."""

        if not self.latest_dir.is_dir():
            raise FileNotFoundError(f"latest checkpoint does not exist: {self.latest_dir}")
        metadata_path = self.latest_dir / "checkpoint_metadata.json"
        if not metadata_path.is_file():
            raise RuntimeError(f"latest checkpoint metadata is missing: {metadata_path}")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise RuntimeError(f"latest checkpoint metadata is invalid: {metadata_path}") from error
        load_state(self.latest_dir)
        return metadata


__all__ = [
    "Barrier",
    "CheckpointEntry",
    "CheckpointManager",
    "LoadState",
    "SaveState",
]
