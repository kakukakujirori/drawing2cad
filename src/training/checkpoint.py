"""PEFT-aware adapter checkpoint state for SFT."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
import torch

from .state import TrainingProgress


def _rng_state(generator: torch.Generator | None) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if generator is not None:
        state["dataloader_generator"] = generator.get_state()
    return state


def _restore_rng_state(
    state: Mapping[str, Any], generator: torch.Generator | None
) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if generator is not None and "dataloader_generator" in state:
        generator.set_state(state["dataloader_generator"])


class AdapterCheckpointIO:
    """Save/load adapter-only model state plus optimizer, scheduler, and RNG."""

    def __init__(
        self,
        *,
        accelerator: Any,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        dataloader_generator: torch.Generator | None = None,
    ) -> None:
        self.accelerator = accelerator
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader_generator = dataloader_generator
        self.progress = TrainingProgress()

    def set_progress(self, progress: TrainingProgress) -> None:
        self.progress = progress

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        model = self.accelerator.unwrap_model(self.model)
        if not hasattr(model, "peft_config") or not hasattr(model, "save_pretrained"):
            raise TypeError("lightweight SFT checkpoints require a PEFT model")
        adapter_dir = directory / "adapter"
        model.save_pretrained(adapter_dir, safe_serialization=True)
        torch.save(self.optimizer.state_dict(), directory / "optimizer.pt")
        torch.save(self.scheduler.state_dict(), directory / "scheduler.pt")
        torch.save(_rng_state(self.dataloader_generator), directory / "rng_state.pt")
        (directory / "training_progress.json").write_text(
            json.dumps(asdict(self.progress), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def load(self, directory: Path) -> TrainingProgress:
        required = (
            directory / "adapter",
            directory / "optimizer.pt",
            directory / "scheduler.pt",
            directory / "rng_state.pt",
            directory / "training_progress.json",
        )
        missing = [path for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"incomplete SFT checkpoint; missing: {missing}")

        model = self.accelerator.unwrap_model(self.model)
        adapter_state = load_peft_weights(
            str(directory / "adapter"), device=str(self.accelerator.device)
        )
        load_result = set_peft_model_state_dict(model, adapter_state)
        unexpected = tuple(getattr(load_result, "unexpected_keys", ()))
        if unexpected:
            raise RuntimeError(f"unexpected adapter checkpoint keys: {unexpected}")
        self.optimizer.load_state_dict(
            torch.load(
                directory / "optimizer.pt",
                map_location=self.accelerator.device,
                weights_only=False,
            )
        )
        self.scheduler.load_state_dict(
            torch.load(
                directory / "scheduler.pt", map_location="cpu", weights_only=False
            )
        )
        rng = torch.load(
            directory / "rng_state.pt", map_location="cpu", weights_only=False
        )
        _restore_rng_state(rng, self.dataloader_generator)
        raw_progress = json.loads(
            (directory / "training_progress.json").read_text(encoding="utf-8")
        )
        self.progress = TrainingProgress(**raw_progress)
        return self.progress


__all__ = ["AdapterCheckpointIO"]
