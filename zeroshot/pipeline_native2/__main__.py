"""Run the native coder and auditor with the shared Hydra config directory."""

import hydra
from omegaconf import DictConfig

from zeroshot.pipeline_native.__main__ import run_and_score
from zeroshot.pipeline_native2.runner import run


@hydra.main(version_base="1.3", config_path="../configs", config_name="native2")
def main(config: DictConfig) -> None:
    run_and_score(config, run)


if __name__ == "__main__":
    main()
