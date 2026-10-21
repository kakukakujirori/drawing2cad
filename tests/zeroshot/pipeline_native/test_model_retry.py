"""Native retries recover incomplete calls without adding answer validation."""

import asyncio
from pathlib import Path

import pytest
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage

from tests.zeroshot.chat_models import ScriptedChatModel, tool_call
from tests.zeroshot.pipeline_native.test_native import _config, _events
from zeroshot.pipeline_native import runner
from zeroshot.pipeline_native.connection_retry import (
    ConnectionRetryMiddleware,
    ModelResponseError,
)


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr(
        "zeroshot.pipeline_native.connection_retry._backoff_delay", lambda _: 0
    )


def _response(reason):
    return AIMessage(
        content=[{"type": "reasoning", "reasoning": "Unfinished analysis."}],
        response_metadata={"finish_reason": reason, "id": "gen-incomplete"},
    )


def _call(middleware, responses, asynchronous):
    request = ModelRequest(
        model=ScriptedChatModel(responses=()),
        messages=[HumanMessage(content="Build the model.")],
    )
    seen = []
    replies = iter(responses)

    def handler(current):
        seen.append(current)
        return ModelResponse(result=[next(replies)])

    async def ahandler(current):
        return handler(current)

    if asynchronous:
        result = asyncio.run(middleware.awrap_model_call(request, ahandler))
    else:
        result = middleware.wrap_model_call(request, handler)
    return result, seen


@pytest.mark.parametrize("asynchronous", [False, True])
def test_error_response_retries_the_same_request(asynchronous, caplog):
    accepted = AIMessage(content="Done.")
    result, seen = _call(
        ConnectionRetryMiddleware(max_retries=1),
        [_response("error"), accepted],
        asynchronous,
    )
    assert result.result == [accepted]
    assert len(seen) == 2 and seen[0] is seen[1]
    retry = caplog.records[-1].args
    assert retry["request_adjusted"] is False
    assert "retry_prompt" not in retry


@pytest.mark.parametrize("asynchronous", [False, True])
def test_length_retry_adds_only_one_submission_reminder(asynchronous, caplog):
    prompt = "Write the current result now."
    accepted = tool_call("run_shell", {"command": "write model.py"}, "write")
    result, seen = _call(
        ConnectionRetryMiddleware(max_retries=2, length_retry_prompt=prompt),
        [_response("length"), _response("length"), accepted],
        asynchronous,
    )
    assert result.result == [accepted]
    assert len(seen) == 3
    assert len(seen[0].messages) == 1
    for request in seen[1:]:
        assert request.messages[:-1] == seen[0].messages
        reminder = request.messages[-1]
        assert isinstance(reminder, HumanMessage)
        assert reminder.text == prompt
        assert request.tools == seen[0].tools
        assert request.tool_choice == seen[0].tool_choice
    assert len(caplog.records) == 2
    for record in caplog.records:
        retry = record.args
        assert retry["request_adjusted"] is True
        assert retry["retry_prompt"] == prompt


@pytest.mark.parametrize("asynchronous", [False, True])
def test_retrospective_length_response_gets_no_submission_reminder(asynchronous):
    truncated = _response("length")
    result, seen = _call(ConnectionRetryMiddleware(), [truncated], asynchronous)
    assert result.result == [truncated]
    assert len(seen) == 1


@pytest.mark.parametrize("reason", ["error", "length"])
def test_native_recovers_and_saves_the_model(tmp_path, reason):
    config = _config(tmp_path, "model_retries=1")
    script = "result = None\n"
    model = ScriptedChatModel(
        responses=(
            _response(reason),
            tool_call(
                "run_shell", {"command": "echo 'result = None' > model.py"}, "write"
            ),
            AIMessage(content="Saved model.py."),
            AIMessage(content="A retrospective."),
        )
    )
    result = runner.run(config, model=model)
    directory = Path(config.artifact_root) / config.sample.sample_id
    assert (directory / "workspace/model.py").read_text() == script
    assert (directory / "reasoning_traj.md").read_text() == "A retrospective.\n"
    assert len(model.received_messages) == 4
    if reason == "error":
        assert model.received_messages[1] == model.received_messages[0]
    else:
        assert "/work/model.py now" in model.received_messages[1][-1].text
    assert all(
        message.response_metadata.get("finish_reason") != reason
        for message in result["messages"]
        if isinstance(message, AIMessage)
    )
    assert _events(config)[-1]["event"] == "run_completed"


@pytest.mark.parametrize("reason", ["error", "length"])
def test_native_response_retry_exhaustion_reports_the_finish_reason(
    tmp_path, reason, caplog
):
    config = _config(tmp_path, "model_retries=1")
    model = ScriptedChatModel(responses=(_response(reason), _response(reason)))
    with pytest.raises(ModelResponseError, match=f"finish_reason='{reason}'"):
        runner.run(config, model=model)
    assert len(model.received_messages) == 2
    failed = _events(config)[-1]
    assert failed["event"] == "run_failed"
    assert failed["data"]["error_type"] == "ModelResponseError"
    assert "gen-incomplete" in failed["data"]["error"]
    retry = caplog.records[-1].args
    assert retry["retrying"] is False
    assert retry["request_adjusted"] is False
    assert "retry_prompt" not in retry


def test_native_response_retries_still_count_toward_the_turn_limit(tmp_path):
    config = _config(tmp_path, "max_turns=2", "model_retries=5")
    model = ScriptedChatModel(responses=(_response("length"), _response("length")))
    result = runner.run(config, model=model)
    assert result["status"] == "turn_limit_exceeded"
    assert len(model.received_messages) == 2
    assert any(event["event"] == "turn_limit_exceeded" for event in _events(config))
