import json
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from zeroshot.pipeline.event_logging.jsonl import JsonlEventWriter, has_run_completed
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

    with (
        pytest.raises(RuntimeError, match="expected failure"),
        JsonlEventWriter(
            path,
            run_id="run-1",
            sample_id="sample-1",
        ) as writer,
    ):
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


def test_has_run_completed_separates_the_three_terminal_states(
    tmp_path: Path,
) -> None:
    """A sweep resumes on this predicate, so failed must not read as done."""
    missing = tmp_path / "absent.jsonl"

    completed = tmp_path / "completed.jsonl"
    with JsonlEventWriter(completed, run_id="r", sample_id="s"):
        pass

    failed = tmp_path / "failed.jsonl"
    try:
        with JsonlEventWriter(failed, run_id="r", sample_id="s"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    interrupted = tmp_path / "interrupted.jsonl"
    interrupted.write_text(
        json.dumps({"event": "run_started", "data": {}}) + "\n", encoding="utf-8"
    )

    assert has_run_completed(completed) is True
    assert has_run_completed(failed) is False
    assert has_run_completed(interrupted) is False
    assert has_run_completed(missing) is False
