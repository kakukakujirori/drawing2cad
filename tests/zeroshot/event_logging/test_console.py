import json
from io import StringIO

from langchain_core.language_models.chat_model_stream import ChatModelStream
from rich.console import Console

from zeroshot.pipeline.event_logging import ConsoleReporter


def test_console_reporter_renders_full_prompt_model_stream_and_tool_output() -> None:
    output = StringIO()
    reporter = ConsoleReporter(
        Console(
            file=output,
            color_system=None,
            force_terminal=False,
            highlight=False,
        )
    )
    prompt = "PROMPT_BEGIN\n" + "p" * 500 + "\nPROMPT_END"
    reasoning = "REASONING_BEGIN\n" + "r" * 500 + "\nREASONING_END"
    answer = "ANSWER_BEGIN\n" + "a" * 500 + "\nANSWER_END"
    command = "COMMAND_BEGIN\n" + "c" * 500 + "\nCOMMAND_END"
    tool_args = json.dumps({"command": command})
    stdout = "STDOUT_BEGIN\n" + "o" * 500 + "\nSTDOUT_END"
    secret = "DO_NOT_PRINT"

    message = ChatModelStream(node="agent", message_id="message-1")
    for event in (
        {"event": "message-start", "id": "message-1", "role": "assistant"},
        {
            "event": "content-block-delta",
            "index": 0,
            "delta": {"type": "reasoning-delta", "reasoning": reasoning},
        },
        {
            "event": "content-block-finish",
            "index": 0,
            "content": {"type": "reasoning", "reasoning": reasoning},
        },
        {
            "event": "content-block-delta",
            "index": 1,
            "delta": {"type": "text-delta", "text": answer},
        },
        {
            "event": "content-block-finish",
            "index": 1,
            "content": {"type": "text", "text": answer},
        },
        {
            "event": "content-block-delta",
            "index": 2,
            "delta": {
                "type": "block-delta",
                "fields": {
                    "type": "tool_call_chunk",
                    "id": "call-1",
                    "name": "run_shell",
                    "args": tool_args[: len(tool_args) // 2],
                    "index": 2,
                },
            },
        },
        {
            "event": "content-block-delta",
            "index": 2,
            "delta": {
                "type": "block-delta",
                "fields": {
                    "type": "tool_call_chunk",
                    "id": "call-1",
                    "name": "run_shell",
                    "args": tool_args,
                    "index": 2,
                },
            },
        },
        {
            "event": "content-block-finish",
            "index": 2,
            "content": {
                "type": "tool_call",
                "id": "call-1",
                "name": "run_shell",
                "args": {"command": command},
            },
        },
        {
            "event": "content-block-delta",
            "index": 3,
            "delta": {
                "type": "future-delta",
                "payload": "UNKNOWN_MODEL_DELTA",
                "api_key": secret,
            },
        },
        {
            "event": "future-message-event",
            "payload": "UNKNOWN_MODEL_EVENT",
        },
        {"event": "message-finish", "usage": None, "metadata": {}},
    ):
        message.dispatch(event)

    with reporter.run_context(run_id="sample-1:run-1", sample_id="sample-1"):
        reporter.render_event(
            {
                "event": "input",
                "timestamp_ms": 1,
                "namespace": [],
                "data": {
                    "messages": [
                        {"type": "system", "content": prompt},
                    ]
                },
            }
        )
        reporter.render_message(message)
        reporter.render_event(
            {
                "event": "message",
                "timestamp_ms": 2,
                "namespace": [],
                "data": {"messages": ["NORMALIZED_MESSAGE_DUPLICATE"]},
            }
        )
        reporter.render_event(
            {
                "event": "tool_finished",
                "timestamp_ms": 3,
                "namespace": [],
                "data": {
                    "tool_name": "run_shell",
                    "output": {
                        "content": json.dumps(
                            {
                                "status": "COMPLETED",
                                "returncode": 0,
                                "stdout": stdout,
                                "stderr": "",
                            }
                        )
                    },
                },
            }
        )
        reporter.render_event(
            {
                "event": "future_run_event",
                "timestamp_ms": 4,
                "namespace": [],
                "data": {
                    "payload": "UNKNOWN_RUN_EVENT",
                    "api_key": secret,
                },
            }
        )

    rendered = output.getvalue()
    for marker in (
        "PROMPT_BEGIN",
        "PROMPT_END",
        "REASONING_BEGIN",
        "REASONING_END",
        "ANSWER_BEGIN",
        "ANSWER_END",
        "COMMAND_BEGIN",
        "COMMAND_END",
        "STDOUT_BEGIN",
        "STDOUT_END",
    ):
        assert marker in rendered
    assert rendered.count("COMMAND_BEGIN") == 1
    assert rendered.count("COMMAND_END") == 1
    assert rendered.count("tool call: run_shell") == 1
    assert rendered.count("REASONING_BEGIN") == 1
    assert rendered.count("ANSWER_BEGIN") == 1
    assert "[unhandled model delta] future-delta" in rendered
    assert "UNKNOWN_MODEL_DELTA" in rendered
    assert "[unhandled model event] future-message-event" in rendered
    assert "UNKNOWN_MODEL_EVENT" in rendered
    assert "[unhandled event] future_run_event" in rendered
    assert "UNKNOWN_RUN_EVENT" in rendered
    assert "<redacted>" in rendered
    assert secret not in rendered
    assert "NORMALIZED_MESSAGE_DUPLICATE" not in rendered
    assert "run completed" in rendered
