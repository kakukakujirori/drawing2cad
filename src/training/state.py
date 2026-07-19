"""Serializable optimizer-boundary state shared by the loop and checkpoint IO."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainingProgress:
    global_step: int = 0
    epoch: int = 0
    batches_seen_in_epoch: int = 0

    def __post_init__(self) -> None:
        for name in ("global_step", "epoch", "batches_seen_in_epoch"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer, got {value!r}")


__all__ = ["TrainingProgress"]
