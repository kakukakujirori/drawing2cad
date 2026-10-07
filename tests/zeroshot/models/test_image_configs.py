from pathlib import Path

import pytest
from hydra.utils import get_class
from langchain_core.messages import HumanMessage
from langchain_core.messages.content import create_image_block
from langchain_openai.chat_models.codex import _ChatOpenAICodex
from omegaconf import OmegaConf

from zeroshot.pipeline.models.image_history import INPUT_IMAGE_ID
from zeroshot.pipeline.models.openrouter import ChatOpenRouter

_CONFIGS = Path(__file__).parents[3] / "zeroshot/configs/model"


def _image_urls(value):
    if isinstance(value, dict):
        if value.get("type") == "image_url":
            return [value["image_url"]["url"]]
        if value.get("type") == "input_image":
            return [value["image_url"]]
        return [url for item in value.values() for url in _image_urls(item)]
    if isinstance(value, list):
        return [url for item in value for url in _image_urls(item)]
    return []


@pytest.mark.parametrize(
    "config_path", sorted(_CONFIGS.glob("*.yaml")), ids=lambda p: p.stem
)
def test_every_model_config_limits_request_images_to_eight(config_path, monkeypatch):
    config = OmegaConf.load(config_path).model
    assert config.image_history_limit == 8
    monkeypatch.setattr(_ChatOpenAICodex, "_codex_headers_sync", lambda self: {})
    model_class = get_class(config._target_)
    model = model_class(
        model=config.model,
        image_history_limit=config.image_history_limit,
        **({} if issubclass(model_class, _ChatOpenAICodex) else {"api_key": "EMPTY"}),
    )
    messages = [
        HumanMessage(
            content=[
                create_image_block(
                    url="https://example.test/drawing.png", id=INPUT_IMAGE_ID
                )
            ]
        ),
        *[
            HumanMessage(
                content=[create_image_block(url=f"https://example.test/{i}.png")]
            )
            for i in range(8)
        ],
    ]
    before = [message.model_dump() for message in messages]
    payload = (
        model._create_message_dicts(messages, None)[0]
        if isinstance(model, ChatOpenRouter)
        else model._get_request_payload(messages)
    )
    assert _image_urls(payload) == [
        "https://example.test/drawing.png",
        *[f"https://example.test/{i}.png" for i in range(1, 8)],
    ]
    assert [message.model_dump() for message in messages] == before
    assert "image_history_limit" not in model.model_kwargs
