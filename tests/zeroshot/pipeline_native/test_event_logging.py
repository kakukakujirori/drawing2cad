import json
from hashlib import sha256
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from zeroshot.pipeline_native.event_logging import EventLog, safe_value, write_json


def protocol(method, data, seq):
    return {
        "type": "event",
        "method": method,
        "seq": seq,
        "params": {"timestamp": 1000 + seq, "namespace": ["generator"], "data": data},
    }


def test_reasoning_events_preserve_parts_metadata_and_order(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    metadata = {"run_id": "call-1", "langgraph_step": 1, "langgraph_node": "model"}
    summary = [
        {"type": "summary_text", "text": "First summary."},
        {"type": "summary_text", "text": "Second summary."},
    ]
    payloads = [
        {"event": "message-start", "id": "response-1", "role": "ai"},
        {
            "event": "content-block-delta",
            "index": 0,
            "delta": {"type": "reasoning", "reasoning": "First summary."},
        },
        {
            "event": "content-block-delta",
            "index": 1,
            "delta": {"type": "reasoning", "reasoning": "Second summary."},
        },
        {
            "event": "content-block-finish",
            "index": 0,
            "content": {"type": "reasoning", "id": "rs_1", "summary": summary},
        },
        {
            "event": "message-finish",
            "metadata": {"finish_reason": "stop"},
            "additional_kwargs": {"reasoning_content": "GLM reasoning."},
        },
    ]
    with EventLog(path) as log:
        for seq, payload in enumerate(payloads):
            log.record_protocol(protocol("messages", (payload, metadata), seq))
        log.record_protocol(protocol("values", {"messages": []}, 5))

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["event_index"] for r in records] == list(range(5))
    assert [r["seq"] for r in records] == list(range(5))
    assert all(r["namespace"] == ["generator"] for r in records)
    assert [r["data"][0] for r in records] == payloads
    assert all(r["data"][1] == metadata for r in records)
    assert records[0]["timestamp"] == 1000


def test_complete_nonstream_messages_keep_provider_fields(tmp_path: Path) -> None:
    ai = AIMessage(
        content="done",
        id="response-2",
        additional_kwargs={"reasoning_content": "GLM reasoning", "provider": {"id": 2}},
        response_metadata={"finish_reason": "stop", "model_name": "glm"},
        usage_metadata={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
    )
    tool = ToolMessage(content="result", tool_call_id="tool-1", name="run_shell")
    path = tmp_path / "events.jsonl"
    with EventLog(path) as log:
        log.record_protocol(protocol("messages", (ai, {"langgraph_step": 2}), 10))
        log.record_protocol(
            protocol("tools", {"event": "tool-finished", "output": tool}, 11)
        )
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[0]["data"] == [ai.model_dump(), {"langgraph_step": 2}]
    assert records[1]["data"]["output"] == tool.model_dump()


def test_partial_events_are_flushed_even_without_console(
    tmp_path: Path, capsys
) -> None:
    path = tmp_path / "events.jsonl"
    partial = {
        "event": "content-block-delta",
        "index": 0,
        "delta": {"type": "reasoning", "reasoning": "partial"},
    }
    with (
        pytest.raises(RuntimeError, match="stream failed"),
        EventLog(path, console=False) as log,
    ):
        log.record_protocol(protocol("messages", (partial, {"run_id": "r"}), 0))
        assert json.loads(path.read_text())["data"][0] == partial
        raise RuntimeError("stream failed")
    assert len(path.read_text().splitlines()) == 1
    assert capsys.readouterr().out == ""

    def failed_console(*args, **kwargs):
        assert json.loads(console_path.read_text())["event"] == "saved"
        raise BrokenPipeError("console failed")

    console_path = tmp_path / "console.jsonl"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("builtins.print", failed_console)
        with (
            pytest.raises(BrokenPipeError),
            EventLog(console_path, console=True) as log,
        ):
            log.write("saved", {})


def test_redaction_keeps_reasoning_and_image_hashes(tmp_path: Path) -> None:
    data_url = "data:image/png;base64,YWJj"
    value = {
        "config": {"api_key": "never-write-this", "Authorization": "secret-header"},
        "messages": (
            HumanMessage(
                content=[
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "image", "base64": "YWJj"},
                ]
            ),
            AIMessage(content="", additional_kwargs={"reasoning_content": "keep me"}),
        ),
    }
    redacted = safe_value(value)
    assert redacted["config"] == {
        "api_key": "<redacted>",
        "Authorization": "<redacted>",
    }
    image = redacted["messages"][0]["content"][0]["image_url"]["url"]
    assert image == {
        "omitted": "image_data_url",
        "size_bytes": len(data_url),
        "sha256": sha256(data_url.encode()).hexdigest(),
    }
    assert redacted["messages"][0]["content"][1]["base64"]["omitted"] == "base64"
    assert (
        redacted["messages"][1]["additional_kwargs"]["reasoning_content"] == "keep me"
    )
    path = tmp_path / "messages.json"
    write_json(path, value)
    assert json.loads(path.read_text()) == redacted
    assert "never-write-this" not in path.read_text()
