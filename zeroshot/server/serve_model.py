import shlex

import hydra
import rootutils
from hydra.utils import instantiate
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from zeroshot.server import SGLangServer


def run(config: DictConfig) -> None:
    server: SGLangServer = instantiate(config.server)
    if config.dry_run:
        print(shlex.join(server.command()))
        return

    server.launch()


@hydra.main(version_base="1.3", config_path="configs", config_name="sglang")
def main(config: DictConfig) -> None:
    run(config)


if __name__ == "__main__":
    main()
