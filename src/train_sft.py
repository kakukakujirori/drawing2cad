"""Hydra entrypoint for the explicit Accelerate SFT loop."""

from __future__ import annotations

import hydra
import rootutils
from omegaconf import DictConfig

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.training.sft import run_sft


CONFIG_DIR = str(ROOT / "configs")


@hydra.main(version_base="1.3", config_path=CONFIG_DIR, config_name="train_sft")
def main(config: DictConfig) -> None:
    run_sft(config)


if __name__ == "__main__":
    main()
