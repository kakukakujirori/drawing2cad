from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.codex import _ChatOpenAICodex

from zeroshot.models import SGLangChatOpenAI
from zeroshot.pipeline.event_logging import ConsoleReporter
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
    console_reporter = instantiate(config.console)
    assert isinstance(builder, MessageBuilder)
    assert isinstance(console_reporter, ConsoleReporter)


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


def test_qwen3_6_sglang_config_instantiates_chat_openai() -> None:
    with initialize_config_dir(
        config_dir=str(CONFIG_DIR.resolve()),
        version_base="1.3",
    ):
        config = compose(
            config_name="default",
            overrides=["model=qwen3_6_sglang"],
        )

    model = instantiate(config.model)

    assert isinstance(model, SGLangChatOpenAI)
    assert model.model_name == "Qwen/Qwen3.6-35B-A3B-FP8"
    assert model.openai_api_base == "http://127.0.0.1:30000/v1"
    assert model.request_timeout == 600.0
    assert model.max_retries == 0
    assert model.streaming is True
    assert model.stream_usage is True
    assert model.max_tokens == 65536
    assert model.extra_body == {
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": True},
    }


def test_qwen3_6_sglang_thinking_and_output_limit_are_overridable() -> None:
    with initialize_config_dir(
        config_dir=str(CONFIG_DIR.resolve()),
        version_base="1.3",
    ):
        config = compose(
            config_name="default",
            overrides=[
                "model=qwen3_6_sglang",
                "model.extra_body.chat_template_kwargs.enable_thinking=false",
                "model.max_tokens=1024",
            ],
        )

    model = instantiate(config.model)

    assert model.extra_body["chat_template_kwargs"]["enable_thinking"] is False
    assert model.max_tokens == 1024


def test_gpt5_6_luna_codex_config_instantiates_oauth_model() -> None:
    with initialize_config_dir(
        config_dir=str(CONFIG_DIR.resolve()),
        version_base="1.3",
    ):
        config = compose(
            config_name="default",
            overrides=["model=gpt5_6_luna_codex"],
        )

    model = instantiate(config.model)

    assert isinstance(model, _ChatOpenAICodex)
    assert model.model_name == "gpt-5.6-luna"
    assert model.output_version == "responses/v1"
    assert model.streaming is True
    assert model.use_responses_api is True
    assert model.store is False
    assert model.openai_api_base == "https://chatgpt.com/backend-api/codex"
    assert model.request_timeout == 600.0
    assert model.max_retries == 0
    assert model.max_tokens is None
