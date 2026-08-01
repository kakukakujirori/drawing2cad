import json
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from zeroshot.pipeline.event_logging.jsonl import JsonlEventWriter
from zeroshot.pipeline.event_logging.normalizer import _safe_value


def test_event_serialization_redacts_images_and_secrets() -> None:
    serialized = _safe_value(
        {
            "message": HumanMessage(
                content=[{"type": "image", "base64": "raw-image-data"}]
            ),
            "openai_api_key": "do-not-log",
        }
    )
    image = serialized["message"]["content"][0]["base64"]
    assert image["omitted"] == "base64"
    assert image["size_bytes"] == len("raw-image-data")
    assert serialized["openai_api_key"] == "<redacted>"


def test_writer_flushes_events_and_records_failure(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"

    with pytest.raises(RuntimeError, match="expected failure"):
        with JsonlEventWriter(
            path,
            run_id="run-1",
            sample_id="sample-1",
        ) as writer:
            writer.write(
                {
                    "event": "tool_started",
                    "timestamp_ms": 1000,
                    "namespace": [],
                    "data": {"tool_name": "run_shell"},
                }
            )

            lines_before_failure = path.read_text(encoding="utf-8").splitlines()
            assert len(lines_before_failure) == 2
            raise RuntimeError("expected failure")

    events = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in events] == [
        "run_started",
        "tool_started",
        "run_failed",
    ]
    assert [event["event_index"] for event in events] == [0, 1, 2]
    assert events[-1]["data"]["error_type"] == "RuntimeError"
