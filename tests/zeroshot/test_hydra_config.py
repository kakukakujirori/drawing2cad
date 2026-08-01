from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from langchain_openai import ChatOpenAI

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


def test_gemma4_ollama_config_instantiates_chat_openai() -> None:
    with initialize_config_dir(
        config_dir=str(CONFIG_DIR.resolve()),
        version_base="1.3",
    ):
        config = compose(
            config_name="default",
            overrides=["model=gemma4_ollama"],
        )

    model = instantiate(config.model)

    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "gemma4:e2b"
    assert model.openai_api_base == "http://127.0.0.1:11434/v1"
    assert model.request_timeout == 600.0
    assert model.max_retries == 0
