"""Run the native baseline with the shared Hydra config directory."""

import hydra
from omegaconf import DictConfig

from zeroshot.pipeline_native.runner import run


@hydra.main(version_base="1.3", config_path="../configs", config_name="workflow/native")
def main(config: DictConfig) -> None:
    run(config)


if __name__ == "__main__":
    main()
