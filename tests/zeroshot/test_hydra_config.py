from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from zeroshot.pipeline.messages import MessageBuilder

CONFIG_DIR = Path(__file__).parents[2] / "zeroshot" / "configs"


def test_default_config_instantiates_message_builder() -> None:
    config_path = CONFIG_DIR / "default.yaml"
    assert config_path.is_file(), config_path

    with initialize_config_dir(
        config_dir=str(CONFIG_DIR.resolve()),
        version_base="1.3",
    ):
        config = compose(config_name="default")

    builder = instantiate(config.message_builder)
    assert isinstance(builder, MessageBuilder)
